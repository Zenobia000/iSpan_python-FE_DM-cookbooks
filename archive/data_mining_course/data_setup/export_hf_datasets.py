#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""把課程用到的 HuggingFace 資料集落地成 datasets/raw/ 底下的一般檔案。

課程有兩個資料來源：Kaggle 走 data_download.py 落到 datasets/raw/，
HuggingFace 則是 notebook 執行時才抓、只存在使用者的 ~/.cache/huggingface。
上課時後者會變成問題——學生現場下載 700 MB，網路一塞整堂就停擺。

這支腳本把它們轉成可直接開的檔案，放到同一個 datasets/raw/ 底下。
老師先跑一次，資料就能隨隨身碟發下去；notebook 會優先讀 raw，沒有才回頭抓 HF。

用法：
    uv run python data_mining_course/data_setup/export_hf_datasets.py
    uv run python data_mining_course/data_setup/export_hf_datasets.py --per-class 500
    uv run python data_mining_course/data_setup/export_hf_datasets.py --only imdb
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COURSE_DIR = SCRIPT_DIR.parent
RAW_DIR = COURSE_DIR / "datasets" / "raw"

# 下載過程的暫存也留在課程樹下，不要散到 ~/.cache/huggingface。
# 匯出完成後 datasets/.hf_cache 可以整個刪掉，notebook 只讀 raw/。
# 已經有既有快取的人可以自己覆寫：HF_HOME=~/.cache/huggingface 執行，避免重抓。
os.environ.setdefault("HF_HOME", str(COURSE_DIR / "datasets" / ".hf_cache"))

IMDB_DIR = RAW_DIR / "imdb_hf"
CATS_DOGS_DIR = RAW_DIR / "dogs_vs_cats"
CLASS_NAMES = {0: "cat", 1: "dog"}


def _require_datasets():
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        sys.exit("❌ 找不到 datasets 套件。請先 `uv sync`，或 `pip install datasets`。")


def export_imdb(force: bool = False) -> None:
    """stanfordnlp/imdb -> raw/imdb_hf/{train,test}.csv（欄位 text, label）"""
    from datasets import load_dataset

    train_csv = IMDB_DIR / "train.csv"
    test_csv = IMDB_DIR / "test.csv"
    if train_csv.exists() and test_csv.exists() and not force:
        print(f"⏭️  IMDB 已存在，跳過：{IMDB_DIR}（--force 可覆寫）")
        return

    print("📥 載入 stanfordnlp/imdb …")
    imdb = load_dataset("stanfordnlp/imdb")
    IMDB_DIR.mkdir(parents=True, exist_ok=True)

    for split, path in (("train", train_csv), ("test", test_csv)):
        imdb[split].to_csv(path, index=False)
        size_mb = path.stat().st_size / 1e6
        print(f"✅ {split}: {len(imdb[split]):,} 列 -> {path} ({size_mb:.1f} MB)")


def export_cats_vs_dogs(per_class: int | None = None, force: bool = False) -> None:
    """microsoft/cats_vs_dogs -> raw/dogs_vs_cats/{cat,dog}/*.jpg

    存成 imagefolder 版面，notebook 可用 load_dataset("imagefolder", data_dir=...)
    讀回來，資料夾名即類別名（cat=0, dog=1，與 HF 的標籤順序一致）。
    """
    from datasets import load_dataset

    existing = sum(1 for _ in CATS_DOGS_DIR.glob("*/*.jpg")) if CATS_DOGS_DIR.exists() else 0
    if existing and not force:
        print(f"⏭️  貓狗影像已存在 {existing:,} 張，跳過：{CATS_DOGS_DIR}（--force 可覆寫）")
        return

    print("📥 載入 microsoft/cats_vs_dogs …")
    ds = load_dataset("microsoft/cats_vs_dogs", split="train")
    for name in CLASS_NAMES.values():
        (CATS_DOGS_DIR / name).mkdir(parents=True, exist_ok=True)

    written = {name: 0 for name in CLASS_NAMES.values()}
    skipped = 0
    total = len(ds)
    for i, example in enumerate(ds):
        name = CLASS_NAMES[example["labels"]]
        if per_class is not None and written[name] >= per_class:
            if all(c >= per_class for c in written.values()):
                break
            continue
        try:
            # 原始資料含極少數損毀檔，轉 RGB 時才會炸；略過並計數，不中斷匯出
            example["image"].convert("RGB").save(
                CATS_DOGS_DIR / name / f"{name}_{written[name]:05d}.jpg", quality=92
            )
            written[name] += 1
        except Exception:
            skipped += 1
        if (i + 1) % 2000 == 0:
            print(f"   … {i + 1:,}/{total:,}（cat={written['cat']:,} dog={written['dog']:,}）")

    print(f"✅ 貓狗影像：cat={written['cat']:,}、dog={written['dog']:,} -> {CATS_DOGS_DIR}")
    if skipped:
        print(f"⚠️  略過 {skipped} 個損毀檔（原始資料本來就有幾個壞檔）")


def main() -> None:
    parser = argparse.ArgumentParser(description="把 HuggingFace 資料集落地到 datasets/raw/")
    parser.add_argument("--only", choices=["imdb", "cats_vs_dogs"],
                        help="只匯出其中一份，預設兩份都匯")
    parser.add_argument("--per-class", type=int, default=None, metavar="N",
                        help="貓狗每類只匯出 N 張（預設全部 23,410 張，約 700 MB）")
    parser.add_argument("--force", action="store_true", help="已存在也重新匯出")
    args = parser.parse_args()

    _require_datasets()
    print(f"📁 目標目錄：{RAW_DIR}")
    print(f"🗃️  下載暫存：{os.environ['HF_HOME']}（匯出完成後可刪）")

    if args.only in (None, "imdb"):
        export_imdb(force=args.force)
    if args.only in (None, "cats_vs_dogs"):
        export_cats_vs_dogs(per_class=args.per_class, force=args.force)

    print("\n🎉 完成。notebook 會自動優先讀這裡的檔案，不再走 HuggingFace。")


if __name__ == "__main__":
    main()
