# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of motion samples from a model and save them as
separate numpy files (per-sample), keeping the HumanML3D vector representation
as (T, 263) float32.
"""

from utils.fixseed import fixseed
import os
import csv
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import collate
import shutil
from datetime import datetime


def main():
    args = generate_args()
    fixseed(args.seed)

    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    # 「データを使うか（=テストセットからサンプリングか）」判定
    is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])

    dist_util.setup_dist(args.device)

    if out_path == '':
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            f'samples_{name}_{niter}_seed{args.seed}'
        )
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # ===== 入力テキスト/アクションの読み込み =====
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r', encoding='utf-8') as fr:
            texts = [s.strip('\n') for s in fr.readlines()]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r', encoding='utf-8') as fr:
            action_text = [s.strip('\n') for s in fr.readlines()]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    args.batch_size = args.num_samples  # GPU保護のため1バッチのみ

    print('Loading dataset...')
    data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # classifier-free guidance用ラッパ

    model.to(dist_util.dev())
    model.eval()  # disable random masking

    # ===== モデル入力（model_kwargs）の準備 =====
    if is_using_data:
        iterator = iter(data)
        _, model_kwargs = next(iterator)
    else:
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        is_t2m = any([args.input_text, args.text_prompt])
        if is_t2m:
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        else:
            action = data.dataset.action_name_to_action(action_text)
            collate_args = [dict(arg, action=one_action, action_text=one_action_text)
                            for arg, one_action, one_action_text in zip(collate_args, action, action_text)]
        _, model_kwargs = collate(collate_args)

    # ===== 出力ディレクトリ準備 =====
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    # インデックスCSV
    index_csv = os.path.join(out_path, 'index.csv')
    with open(index_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'saved_file', 'sample_idx', 'repetition_idx', 'length',
            'text_or_action', 'model_path', 'seed', 'dataset',
            'motion_length_sec', 'fps', 'saved_at'
        ])

    print('### Start sampling')
    sample_fn = diffusion.p_sample_loop

    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetition #{rep_i}]')

        # CFGスケール
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        # 生成（max_frames形状で生成）
        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        # テキスト or アクション
        if args.unconstrained:
            text_list = ['unconstrained'] * args.num_samples
        else:
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            text_list = list(model_kwargs['y'][text_key])

        lengths = model_kwargs['y']['lengths'].detach().cpu().numpy()

        # ===== サンプルごとに保存 =====
        for sample_i in range(args.num_samples):
            length = int(lengths[sample_i])
            caption = text_list[sample_i] if sample_i < len(text_list) else ''

            # (C1, C2, T) → (T, 263)
            one_motion = sample[sample_i, :, :, :length].detach().cpu().numpy()
            C1, C2, T = one_motion.shape
            D = C1 * C2
            assert D == 263, f"Expected 263D but got {D}D from shape {one_motion.shape}"
            motion_263 = one_motion.reshape(D, T).T.astype(np.float32)

            save_name = f'sample{sample_i:02d}_rep{rep_i:02d}.npy'
            save_path = os.path.join(out_path, save_name)
            np.save(save_path, motion_263)

            print(f'[Saved] {save_path} | shape={motion_263.shape} | dtype={motion_263.dtype} | text="{caption}"')

            # CSV追記
            with open(index_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    save_name, sample_i, rep_i, length,
                    caption, args.model_path, args.seed, args.dataset,
                    args.motion_length, fps, datetime.now().isoformat(timespec='seconds')
                ])

    abs_path = os.path.abspath(out_path)
    print(f'[Done] {total_num_samples} files saved under: {abs_path}')
    print(f'Index: {index_csv}')


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        split='test',
        hml_mode='text_only'
    )
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()