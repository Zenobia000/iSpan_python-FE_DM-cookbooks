#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把各章的 slides_content.md 生成投影片 PNG（版型 M，可續跑）。

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

STYLE = ("一張 16:9 顧問級商業簡報投影片。深海軍藍純色背景（近乎 #051C2C），無漸層無雜訊。"
         "左上角白色粗襯線體大標題靠左，標題下方一條細白色水平分隔線橫貫版面。"
         "內容區用等寬多欄或流程排列，欄與欄之間留白充足。"
         "只用一個強調色：亮天藍（近乎 #00A9F4），用在小標、數字與重點圖示；其餘一律白色或淺灰。"
         "圖示一律是白色細線條、畫在細線圓框內，線條簡潔不填色。"
         "繁體中文字大而清晰，版面乾淨專業。")

# 模型會自己補「來源：McKinsey & Company」這種署名，或自行編造百分比。
# 這段固定接在每頁提示詞尾端，把捏造率壓下來。
NO_FABRICATION = ("只畫我指定的文字，不要加入任何其他文字。特別是：不要寫來源、不要寫公司名或署名、"
                  "不要自行加入百分比或統計數字、不要加浮水印。左下角保持空白。")

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
                  f"左上角白色襯線大標題寫「{title}」。"
                  f"右下角小字頁碼寫「{chapter} · {num} / {total}」。\n"
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
    ap = argparse.ArgumentParser(description="生成投影片 PNG（版型 M）")
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
