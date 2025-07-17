import os

# userstudy_list.txt からファイル名を取得（空行やラベル行を除外）
with open("userstudy_list.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip() and not line.endswith(":")]

# 出力先ファイル
with open("userstudy_prompt.txt", "w", encoding="utf-8") as out_file:
    for filename in lines:
        txt_path = os.path.join("HumanML3D_0524_person/texts", filename + ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                first_line = f.readline().strip()
                prompt = first_line.split("#")[0]
                out_file.write(prompt + "\n")
        else:
            print(f"[警告] ファイルが見つかりません: {txt_path}")
