#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把各章的投影片拼成聯絡表，一張圖看完一整章，方便快速校對。

    python3 data_mining_course/slides_plan/make_contact_sheet.py
    python3 data_mining_course/slides_plan/make_contact_sheet.py M04_類別編碼

輸出到 _contact/<章名>.png，每頁縮圖左上角標頁碼。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR/"_contact"
COLS = 3
THUMB_W = 640
PAD = 14
LABEL_H = 30


def sheet(chapter_dir: Path) -> Path | None:
    pngs = sorted(chapter_dir.glob("_generated/p*.png"))
    if not pngs:
        return None
    thumbs = []
    for f in pngs:
        im = Image.open(f).convert("RGB")
        h = round(im.height * THUMB_W / im.width)
        thumbs.append((f.name[:3], im.resize((THUMB_W, h), Image.LANCZOS)))
    tw, th = THUMB_W, max(t.height for _, t in thumbs)
    rows = (len(thumbs) + COLS - 1) // COLS
    W = COLS*tw + (COLS+1)*PAD
    H = rows*(th+LABEL_H) + (rows+1)*PAD
    canvas = Image.new("RGB", (W, H), (235, 235, 235))
    d = ImageDraw.Draw(canvas)
    for i, (tag, im) in enumerate(thumbs):
        r, c = divmod(i, COLS)
        x = PAD + c*(tw+PAD)
        y = PAD + r*(th+LABEL_H+PAD)
        d.text((x+4, y+6), f"{chapter_dir.name}  {tag}", fill=(20, 20, 20))
        canvas.paste(im, (x, y+LABEL_H))
        d.rectangle([x, y+LABEL_H, x+tw-1, y+LABEL_H+im.height-1], outline=(170, 170, 170))
    OUT.mkdir(exist_ok=True)
    out = OUT/f"{chapter_dir.name}.png"
    canvas.save(out)
    return out


def main() -> None:
    targets = ([SCRIPT_DIR/a for a in sys.argv[1:]] if len(sys.argv) > 1
               else sorted(d for d in SCRIPT_DIR.iterdir()
                           if d.is_dir() and (d/"_generated").exists()))
    for d in targets:
        p = sheet(d)
        if p:
            print(f"{d.name:<22} -> {p.relative_to(SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
