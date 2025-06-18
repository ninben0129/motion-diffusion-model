import gradio as gr
import subprocess
import uuid
import os

def call_model_and_get_video(prompt):
    # output_path = f"outputsfordemo/{uuid.uuid4()}.mp4"
    # model_path = "./save/finetune_0524_person/model000400000.pt"
    output_dir = os.path.abspath("outputsfordemo")
    output_filename = f"{uuid.uuid4()}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    # model_path = os.path.abspath("./save/finetune_0524_person/model000400000.pt")
    # model_path = "/home/icd/motion-diffusion-model/save/finetune_0524_person/model000100000.pt"
    model_path = "/home/icd/motion-diffusion-model/save/finetune_0401/model000200000.pt"
    motion_diffusion_root = "/home/icd/motion-diffusion-model"

    try:
        os.makedirs("outputsfordemo", exist_ok=True)

        # subprocess コマンド
        # subprocess.run(
        #     [
        #         "conda", "run", "-n", "mdm", "python", "-m", "sample.generatefordemo",
        #         "--model_path", model_path,
        #         "--text_prompt", prompt,
        #         "--pathfordemo", output_path
        #     ],
        #     check=True
        # )
        # subprocess.run(
        #     [
        #         "conda", "run", "-n", "mdm", "python", "-m", "sample.generatefordemo",
        #         "--model_path", model_path,
        #         "--text_prompt", prompt,
        #         "--pathfordemo", output_path
        #     ],
        #     check=True
        # )

        cmd = (
            f'conda run -n mdm python -m sample.generatefordemo '
            f'--model_path "{model_path}" '
            f'--text_prompt "{prompt}" '
            f'--pathfordemo "{output_path}"'
        )

        subprocess.run(cmd, shell=True, check=True,cwd=motion_diffusion_root)

        if os.path.exists(output_path):
            return output_path
        else:
            return "Error: 出力ファイルが存在しません"
    except subprocess.CalledProcessError as e:
        return f"実行エラー:\n{e}"

# Gradio で mp4 を表示
gr.Interface(
    fn=call_model_and_get_video,
    inputs=gr.Textbox(label="プロンプト"),
    outputs=gr.Video(label="生成動画", height=480, width=360),
    examples=[
        ["A person is walking in surprise."],
        ["The Person happily stretches his arms outward."],
        ["The person angrily crosses his arms over his chest."]
    ],
    title="emotional text2motion generation"
).launch()