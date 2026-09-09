#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把各章的 slides_content.md 生成投影片 PNG（版型 C，可續跑）。

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

STYLE = ("一張 16:9 教學投影片，牛皮紙速寫本風格。淺牛皮紙質感背景，內容像用鉛筆與細字筆手繪的教學筆記，"
         "帶手繪箭頭、圈選與底線強調。左上角手寫章名並畫一條手繪橫線，右下角手寫頁碼。"
         "配色為莫蘭迪低飽和：赭石、灰藍、苔綠、暗玫瑰，紙張暖色調。"
         "繁體中文字要大而清晰，整頁文字量少，不要出現小字段落。")

# slides_content.md 的一頁：## 3 · 決策表   後面接內容描述段落
PAGE = re.compile(r'^##\s+(\d+)\s+·\s+(.+?)\s*$', re.M)


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
                  f"左上角章名寫「{chapter}」。中央大號手寫標題寫「{title}」。"
                  f"右下角頁碼寫「{num} / {total}」。\n頁面內容：{body}")
        cmd = ["python3", str(DRAW), prompt, "--name", name,
               "--size", "1536x1024", "--quality", "low", "--outdir", str(out)]
        print(f"  [{num:2d}/{total}] {title}")
        if dry_run:
            made += 1
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["未知錯誤"]
            print(f"  ⚠️ 第 {num} 頁失敗：{tail[0]}")
            print(f"     已完成 {made} 頁；補額度後重跑同一個指令即可接續。")
            return made, skipped
        made += 1
    return made, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="生成投影片 PNG（版型 C）")
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
