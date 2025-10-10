# -*- coding: utf-8 -*-
"""
output_finetuned に存在するファイル名と同名のファイルだけを
groundtruth から groundtruth_test へコピーするスクリプト
（サブフォルダ内のファイルも対象。ファイル名で一致判定）
"""

import shutil
from pathlib import Path

# ====== パスを必要に応じて書き換えてください ======
GROUNDTRUTH_DIR     = Path("groundtruth")
OUTPUT_FINETUNED_DIR = Path("output_finetuned")
DEST_DIR            = Path("groundtruth_test")

# 既に同名ファイルがある場合に上書きするなら True、スキップするなら False
OVERWRITE = True

def collect_filenames(root: Path) -> set[str]:
    """ディレクトリ配下の全ファイル名（拡張子込み）の集合を作る"""
    return {p.name for p in root.rglob("*") if p.is_file()}

def main():
    if not GROUNDTRUTH_DIR.exists():
        raise FileNotFoundError(f"GROUNDTRUTH_DIR が見つかりません: {GROUNDTRUTH_DIR}")
    if not OUTPUT_FINETUNED_DIR.exists():
        raise FileNotFoundError(f"OUTPUT_FINETUNED_DIR が見つかりません: {OUTPUT_FINETUNED_DIR}")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # output_finetuned にある全ファイル名（重複は集合に）
    target_names = collect_filenames(OUTPUT_FINETUNED_DIR)
    if not target_names:
        print("output_finetuned にファイルが見つかりません。処理を終了します。")
        return

    copied = 0
    skipped = 0
    missing = 0

    # groundtruth 側で、対象名に一致するファイルを探してコピー
    # 同名が複数（別サブフォルダ）にある場合はすべてコピーします（保存先は直下）
    name_to_paths: dict[str, list[Path]] = {}
    for p in GROUNDTRUTH_DIR.rglob("*"):
        if p.is_file() and p.name in target_names:
            name_to_paths.setdefault(p.name, []).append(p)

    # output_finetuned にあったのに groundtruth に無いファイルも把握
    missing_names = target_names - set(name_to_paths.keys())
    if missing_names:
        missing = len(missing_names)

    for name, paths in name_to_paths.items():
        for src in paths:
            dst = DEST_DIR / name
            if dst.exists() and not OVERWRITE:
                skipped += 1
                continue
            shutil.copy2(src, dst)
            copied += 1

    print("=== 完了 ===")
    print(f"対象ファイル名数（output_finetuned 起点）：{len(target_names)}")
    print(f"groundtruth からコピーしたファイル数       ：{copied}")
    print(f"既存でスキップしたファイル数               ：{skipped}")
    print(f"groundtruth に見つからなかったファイル名数 ：{missing}")

if __name__ == "__main__":
    main()
