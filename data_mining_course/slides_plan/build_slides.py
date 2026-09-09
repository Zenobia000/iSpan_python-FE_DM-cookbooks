#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把各章的 slides_content.md 生成投影片 PNG（版型 W，可續跑）。

一頁一張圖。已經生成過的頁會跳過，所以撞到額度上限時直接重跑即可，
不會重複計費。失敗的請求 OpenAI 不計費。

    python3 data_mining_course/slides_plan/build_slides.py M04_類別編碼
    python3 data_mining_course/slides_plan/build_slides.py --all
    python3 data_mining_course/slides_plan/build_slides.py --all --dry-run
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DRAW = Path.home()/".claude"/"skills"/"draw"/"draw.py"

STYLE = ("一張 16:9 教學投影片。純白色背景，乾淨無底紋無漸層。"
         "字體規定：全頁所有文字，包含中文、英文與阿拉伯數字，一律使用無襯線黑體，"
         "類似思源黑體或 Helvetica 的字形。標題、內文、標籤、數字全部都是無襯線。"
         "絕對不要出現襯線體、明體、宋體、楷體或任何有裝飾筆畫的字形，標題也不例外。"
         "主文字為近黑色，次要說明為中灰色，分隔線與線框為淺灰色。"
         "標題靠左，字重明顯較粗，標題下方一條淺灰色水平細線橫貫內容區寬度。"
         "強調只用兩種手段：把字加粗，或在字下方加一條粗的黑色底線。"
         "需要標示錯誤時才用朱紅色，其餘一律黑白灰。"
         "版面留白充足，元素之間對齊嚴謹，整體像印刷品而不是網頁。"
         "邊界嚴格：上下各留白約畫面高度的百分之九，左右各留白約畫面寬度的百分之七，"
         "所有內容的左邊界對齊同一條垂直線，多欄並排時欄距一致、最左欄貼齊左邊界、最右欄貼齊右邊界。")

# 模型會自己補署名、頁碼與統計數字。這段固定接在每頁提示詞尾端。
NO_FABRICATION = ("只畫我指定的文字，不要加入任何其他文字。"
                  "右下角不要頁碼、不要章名標籤、不要任何頁尾。"
                  "不要寫來源、不要寫公司名或署名、不要加浮水印。"
                  "不要自行加入百分比或統計數字。不要使用藍色。")

# slides_content.md 的一頁：## 3 · 決策表   後面接內容描述段落
PAGE = re.compile(r'^##\s+(\d+)\s+·\s+(.+?)\s*$', re.M)


def to_16x9(png: Path) -> tuple[int, int]:
    """把 1536x1024（3:2）補成 16:9，不裁切內容。

    版型 M 的底是純色海軍藍，所以左右補上從角落取樣到的實際色值，
    接縫看不出來。回傳補完的尺寸。
    """
    from PIL import Image

    img = Image.open(png).convert("RGB")
    w, h = img.size
    target_w = round(h * 16 / 9)
    if w >= target_w:
        return w, h
    # 四個角各取一點，取眾數當底色，避免壓到內容
    corners = [img.getpixel(c) for c in ((2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3))]
    bg = max(set(corners), key=corners.count)
    canvas = Image.new("RGB", (target_w, h), bg)
    canvas.paste(img, ((target_w - w) // 2, 0))
    canvas.save(png)
    return target_w, h


def parse(chapter_dir: Path) -> tuple[str, list[tuple[int, str, str]]]:
    f = chapter_dir/"slides_content.md"
    if not f.exists():
        sys.exit(f"❌ 找不到 {f}，請先寫該章的 slides_content.md")
    text = f.read_text(encoding="utf-8")
    m = re.search(r'^章名：(.+)$', text, re.M)
    if not m:
        sys.exit(f"❌ {f} 缺少「章名：」那一行")
    chapter = m.group(1).strip()
    hits = list(PAGE.finditer(text))
    pages = []
    for i, h in enumerate(hits):
        body = text[h.end(): hits[i+1].start() if i+1 < len(hits) else len(text)].strip()
        # 「頁型：C 密度」這行只給人看，用來排節奏，不送進提示詞
        body = re.sub(r'^頁型：.*\n+', '', body).strip()
        pages.append((int(h.group(1)), h.group(2).strip(), body))
    return chapter, pages


def build(chapter_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    chapter, pages = parse(chapter_dir)
    out = chapter_dir/"_generated"
    out.mkdir(exist_ok=True)
    total = len(pages)
    made = skipped = 0
    for num, title, body in pages:
        name = f"p{num:02d}"
        if list(out.glob(f"{name}_*.png")):
            skipped += 1
            continue
        prompt = (f"{STYLE}\n\n"
                  f"左上角大標題寫「{title}」。\n"
                  f"頁面內容：{body}\n\n{NO_FABRICATION}")
        cmd = ["python3", str(DRAW), prompt, "--name", name,
               "--size", "1536x1024", "--quality", "low", "--outdir", str(out)]
        print(f"  [{num:2d}/{total}] {title}")
        if dry_run:
            made += 1
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            for f in out.glob(f"{name}_*.png"):
                w, h = to_16x9(f)
                print(f"       -> 16:9 {w}x{h}")
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["未知錯誤"]
            print(f"  ⚠️ 第 {num} 頁失敗：{tail[0]}")
            print(f"     已完成 {made} 頁；補額度後重跑同一個指令即可接續。")
            return made, skipped
        made += 1
    return made, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="生成投影片 PNG（版型 W）")
    ap.add_argument("chapters", nargs="*", help="章資料夾名，例如 M04_類別編碼")
    ap.add_argument("--all", action="store_true", help="跑全部有 slides_content.md 的章")
    ap.add_argument("--dry-run", action="store_true", help="只列出要生哪幾頁，不呼叫 API")
    args = ap.parse_args()

    if args.all:
        targets = sorted(d for d in SCRIPT_DIR.iterdir()
                         if d.is_dir() and (d/"slides_content.md").exists())
    else:
        targets = [SCRIPT_DIR/c for c in args.chapters]
    if not targets:
        sys.exit("沒有指定章節，也沒有任何章寫好 slides_content.md")

    grand_made = grand_skip = 0
    for d in targets:
        print(f"\n=== {d.name} ===")
        made, skipped = build(d, args.dry_run)
        grand_made += made
        grand_skip += skipped
    print(f"\n本次生成 {grand_made} 頁，跳過已存在 {grand_skip} 頁。")


if __name__ == "__main__":
    main()
