# This code is based on https://github.com/openai/guided-diffusion
"""
Generate motion samples as (T, 263) float32 .npy files ONLY, using text prompts
loaded from individual files in a directory, where each filename (without extension)
is listed line-by-line in a name list. For each prompt file, only the substring
before the first '#' is used as the actual text prompt.
"""

from utils.fixseed import fixseed
import os
import sys
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.tensors import collate

def main():
    args = generate_args()
    fixseed(args.seed)

    # ---- 追加引数の柔軟対応（parser未対応でも落ちない） ----
    if not hasattr(args, 'name_list'):
        setattr(args, 'name_list', '')
    if not hasattr(args, 'prompt_dir'):
        setattr(args, 'prompt_dir', '')

    # ====== 必須: 出力先フォルダ（.npyのみ出力） ======
    out_dir = args.output_dir
    if not out_dir:
        raise ValueError("--output_dir を必ず指定してください（この版は .npy をそのフォルダにのみ出力します）")
    os.makedirs(out_dir, exist_ok=True)

    # ====== 入力: 名前リスト と プロンプトフォルダ ======
    if not args.name_list or not os.path.exists(args.name_list):
        raise ValueError("--name_list にベース名の列挙ファイル（1行1つ、拡張子なし）を指定してください")
    if not args.prompt_dir or not os.path.isdir(args.prompt_dir):
        raise ValueError("--prompt_dir に {name}.txt が置かれたフォルダを指定してください")

    # HumanML3D/kitに準拠した長さ計算（従来どおり）
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length * fps))

    # 分散設定
    dist_util.setup_dist(args.device)

    # ====== モデル準備 ======
    print("Loading dataset (for shapes/metadata)...")
    data = load_dataset(args, max_frames, n_frames)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # CFG wrapper

    model.to(dist_util.dev())
    model.eval()

    # ====== 名前リストを読み取り、対応するテキストをロード ======
    base_names = []
    prompts = []
    with open(args.name_list, 'r', encoding='utf-8') as fr:
        for line in fr:
            nm = line.strip()
            if not nm:
                continue
            txt_path = os.path.join(args.prompt_dir, nm + '.txt')
            if not os.path.exists(txt_path):
                print(f"[WARN] prompt file not found, skip: {txt_path}")
                continue
            with open(txt_path, 'r', encoding='utf-8') as tf:
                raw = tf.read().strip()
            # 最初の '#' より前だけ使用
            sharp_idx = raw.find('#')
            if sharp_idx != -1:
                raw = raw[:sharp_idx]
            raw = raw.strip()
            if not raw:
                print(f"[WARN] empty prompt after '#' stripping, skip: {txt_path}")
                continue
            base_names.append(nm)
            prompts.append(raw)

    if not base_names:
        print("[INFO] No valid prompts. Nothing to do.")
        return

    # ====== バッチ処理で生成（.npyのみ保存） ======
    # ユーザ指定の batch_size を尊重
    batch_size = max(1, int(args.batch_size))
    sample_fn = diffusion.p_sample_loop

    total = len(base_names)
    print(f"### Start sampling for {total} prompts (batch_size={batch_size}, repetitions={args.num_repetitions})")

    # 繰り返し（repetitions）はファイル名サフィックスで区別
    for rep_i in range(args.num_repetitions):
        print(f"### Repetition #{rep_i}")
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            names_chunk = base_names[start:end]
            texts_chunk = prompts[start:end]

            # collate入力の準備
            collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * len(texts_chunk)
            collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts_chunk)]
            _, model_kwargs = collate(collate_args)

            # CFGスケール付与
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(len(texts_chunk), device=dist_util.dev()) * args.guidance_param

            # 生成（max_frames形状）
            sample = sample_fn(
                model,
                (len(texts_chunk), model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            lengths = model_kwargs['y']['lengths'].detach().cpu().numpy()  # [B]

            # 保存：各サンプルごとに (T,263)/float32 を {name}[ _repXX ].npy で出力
            for i, nm in enumerate(names_chunk):
                length = int(lengths[i])

                # (C1, C2, T) -> (T, 263)
                one_motion = sample[i, :, :, :length].detach().cpu().numpy()
                C1, C2, T = one_motion.shape
                D = C1 * C2
                if D != 263:
                    raise RuntimeError(f"Expected 263D but got {D}D from shape {one_motion.shape}")
                motion_263 = one_motion.reshape(D, T).T.astype(np.float32)  # (T,263)

                # 繰り返しが1回なら上書きしないようそのまま / 複数回ならサフィックス付与
                if args.num_repetitions <= 1:
                    save_name = f"{nm}.npy"
                else:
                    save_name = f"{nm}_rep{rep_i:02d}.npy"

                save_path = os.path.join(out_dir, save_name)
                np.save(save_path, motion_263)
                print(f"[Saved] {save_path} | shape={motion_263.shape} | dtype={motion_263.dtype}")

    print(f"[Done] Generated .npy files under: {os.path.abspath(out_dir)}")


def load_dataset(args, max_frames, n_frames):
    data = get_dataset_loader(
        name=args.dataset,
        batch_size=max(1, int(args.batch_size)),
        num_frames=max_frames,
        split='test',
        hml_mode='text_only'
    )
    if args.dataset in ['kit', 'humanml']:
        data.dataset.t2m_dataset.fixed_length = n_frames
    return data


if __name__ == "__main__":
    main()
