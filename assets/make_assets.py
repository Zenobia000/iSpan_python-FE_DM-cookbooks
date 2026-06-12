#!/usr/bin/env python3
"""產生 README 行銷視覺：hero.png / og-cover.png / showcase.gif
底圖為 gpt-image-2 生成的等距插畫，文字以 PIL + Noto Sans CJK TC 疊上（正確繁中）。"""
import glob, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ASSETS = "assets"
SRC = f"{ASSETS}/hero_source.png"   # gpt-image-2 生成的等距插畫底圖（無文字）
FB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# 配色（與插畫一致）
NAVY   = (10, 14, 38)
WHITE  = (248, 250, 252)
MUTED  = (148, 163, 184)
PURPLE = (167, 139, 250)
BLUE   = (96, 165, 250)
TEAL   = (45, 212, 191)
GOLD   = (245, 158, 11)

def font(sz, bold=True):
    return ImageFont.truetype(FB if bold else FR, sz)

def text_w(draw, s, f):
    b = draw.textbbox((0, 0), s, font=f); return b[2] - b[0]

def left_scrim(img, frac=0.62):
    """在左側加深色漸層遮罩，讓白字清楚。"""
    w, h = img.size
    scrim = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = scrim.load()
    cut = int(w * frac)
    for x in range(cut):
        a = int(225 * (1 - x / cut) ** 1.15)
        for y in range(h):
            px[x, y] = (NAVY[0], NAVY[1], NAVY[2], a)
    return Image.alpha_composite(img.convert("RGBA"), scrim)

def glow_text(base, xy, s, f, fill, glow=None, gr=8):
    if glow:
        lyr = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lyr); d.text(xy, s, font=f, fill=glow + (255,))
        lyr = lyr.filter(ImageFilter.GaussianBlur(gr))
        base.alpha_composite(lyr)
    ImageDraw.Draw(base).text(xy, s, font=f, fill=fill + (255,) if len(fill) == 3 else fill)

# ---------- 共用：載入底圖並裁切到指定 banner 比例（保留方塊+核心，去除多餘空白）----------
art = Image.open(SRC).convert("RGB")
AW, AH = art.size  # 1536x1024

def crop_to(target_w, target_h, top=0.03):
    """從底圖裁出 target 比例的橫幅（取上半保留主視覺）。"""
    ratio = target_h / target_w
    ch = int(AW * ratio)
    y0 = int(AH * top)
    if y0 + ch > AH:
        y0 = AH - ch
    c = art.crop((0, y0, AW, y0 + ch)).resize((target_w, target_h), Image.LANCZOS)
    return c

def compose_banner(W, H, title_y, t1_sz, t2_sz, sub_sz, pill_sz):
    base = crop_to(W, H)
    base = left_scrim(base, frac=0.66).convert("RGBA")
    x = 70
    y = title_y
    glow_text(base, (x, y), "資料探勘與特徵工程", font(t1_sz), WHITE, glow=(30, 40, 90), gr=10)
    y += int(t1_sz * 1.22)
    glow_text(base, (x, y), "2026 大模型實戰", font(t2_sz), TEAL, glow=TEAL, gr=14)
    y += int(t2_sz * 1.28)
    d = ImageDraw.Draw(base)
    d.text((x + 3, y), "Data Mining × Feature Engineering for Large Models",
           font=font(sub_sz, bold=False), fill=MUTED + (255,))
    y += int(sub_sz * 1.9)
    # 模態流向膠囊
    seg = [("文字", PURPLE), ("圖像", BLUE), ("聲音", TEAL), ("影片", GOLD)]
    fx = x + 3
    fpill = font(pill_sz)
    for i, (lbl, col) in enumerate(seg):
        d.text((fx, y), lbl, font=fpill, fill=col + (255,))
        fx += text_w(d, lbl, fpill) + 10
        if i < 3:
            d.text((fx, y), "·", font=fpill, fill=MUTED + (255,)); fx += text_w(d, "·", fpill) + 10
    d.text((fx + 6, y), "→  大模型", font=fpill, fill=WHITE + (255,))
    return base, d, y

# ---------- HERO 1280x460 ----------
W, H = 1280, 460
hero, d, y = compose_banner(W, H, title_y=64, t1_sz=62, t2_sz=58, sub_sz=27, pill_sz=30)
# 底部統計列（膠囊）
stats = ["11 模組", "67 實戰筆記本", "PyTorch + HuggingFace", "真實資料 · CPU 可跑"]
sf = font(24, bold=False); sx = 73; sy = H - 66
for s in stats:
    tw = text_w(d, s, sf)
    d.rounded_rectangle([sx, sy, sx + tw + 34, sy + 42], radius=21,
                        fill=(255, 255, 255, 18), outline=(255, 255, 255, 60), width=1)
    d.text((sx + 17, sy + 8), s, font=sf, fill=WHITE + (230,))
    sx += tw + 34 + 14
hero.convert("RGB").save(f"{ASSETS}/hero.png")
print("✓ assets/hero.png", hero.size)

# ---------- OG 社群圖 1280x640 ----------
W, H = 1280, 640
og, d, y = compose_banner(W, H, title_y=150, t1_sz=72, t2_sz=66, sub_sz=30, pill_sz=34)
stats = ["11 模組", "67 實戰筆記本", "PyTorch + HuggingFace"]
sf = font(27, bold=False); sx = 73; sy = H - 96
for s in stats:
    tw = text_w(d, s, sf)
    d.rounded_rectangle([sx, sy, sx + tw + 38, sy + 48], radius=24,
                        fill=(255, 255, 255, 18), outline=(255, 255, 255, 60), width=1)
    d.text((sx + 19, sy + 9), s, font=sf, fill=WHITE + (230,))
    sx += tw + 38 + 16
og.convert("RGB").save(f"{ASSETS}/og-cover.png")
print("✓ assets/og-cover.png", og.size)

# ================= GIF：4 模態依序點亮 → 大模型 =================
GW, GH = 1200, 380

def grad_bg(w, h):
    bg = Image.new("RGB", (w, h))
    px = bg.load()
    for yy in range(h):
        for xx in range(0, w, 2):
            t = (xx / w * 0.6 + yy / h * 0.4)
            r = int(10 + 30 * t); g = int(14 + 16 * t); b = int(38 + 60 * t)
            px[xx, yy] = (r, g, b)
            if xx + 1 < w: px[xx + 1, yy] = (r, g, b)
    return bg

def icon(d, kind, cx, cy, col, a):
    c = col + (a,)
    if kind == "text":
        for i, wdt in enumerate([46, 38, 44, 30]):
            yy = cy - 30 + i * 18
            d.rounded_rectangle([cx - 24, yy, cx - 24 + wdt, yy + 8], radius=4, fill=c)
    elif kind == "image":
        d.rounded_rectangle([cx - 30, cy - 24, cx + 30, cy + 24], radius=6, outline=c, width=4)
        d.ellipse([cx + 6, cy - 16, cx + 20, cy - 2], fill=c)
        d.polygon([(cx - 26, cy + 20), (cx - 6, cy - 6), (cx + 14, cy + 20)], fill=c)
    elif kind == "audio":
        for i, hh in enumerate([14, 30, 20, 40, 24, 12]):
            xx = cx - 30 + i * 11
            d.rounded_rectangle([xx, cy - hh // 2, xx + 6, cy + hh // 2], radius=3, fill=c)
    elif kind == "video":
        d.rounded_rectangle([cx - 30, cy - 22, cx + 30, cy + 22], radius=6, outline=c, width=4)
        d.polygon([(cx - 8, cy - 12), (cx - 8, cy + 12), (cx + 14, cy)], fill=c)

def draw_model(layer, cx, cy, lit):
    d = ImageDraw.Draw(layer)
    R = 78
    acc = TEAL if lit else (70, 90, 110)
    a = 255 if lit else 150
    # hexagon
    pts = [(cx + R * math.cos(math.radians(60 * k - 90)),
            cy + R * math.sin(math.radians(60 * k - 90))) for k in range(6)]
    d.polygon(pts, outline=acc + (a,), width=5)
    # 內部節點（mini neural net）
    rng = [(-34, -20), (-30, 18), (0, -36), (4, 4), (34, -16), (30, 24), (0, 34)]
    for (dx, dy) in rng:
        d.ellipse([cx + dx - 6, cy + dy - 6, cx + dx + 6, cy + dy + 6], fill=acc + (a,))
    for i in range(len(rng)):
        for j in range(i + 1, len(rng)):
            if (i + j) % 2 == 0:
                d.line([cx + rng[i][0], cy + rng[i][1], cx + rng[j][0], cy + rng[j][1]],
                       fill=acc + (90 if lit else 50,), width=2)

CARDS = [("文字", "text", PURPLE), ("圖像", "image", BLUE),
         ("聲音", "audio", TEAL), ("影片", "video", GOLD)]
cardW, cardH, gap = 150, 150, 22
x0 = 56
cy = GH // 2 + 6
model_cx, model_cy = GW - 150, cy
lab = font(30); cap = font(26, bold=False)

def make_frame(active):
    """active: 0..3 點亮第幾張卡, 4 點亮模型"""
    base = grad_bg(GW, GH).convert("RGBA")
    glow = Image.new("RGBA", (GW, GH), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    # 連接線（卡片 → 模型）
    for i, (_, _, col) in enumerate(CARDS):
        x = x0 + i * (cardW + gap)
        lit_line = (active >= 4)
        gd.line([x + cardW, cy, model_cx - 70, cy], fill=col + (70 if lit_line else 30,), width=3)
    base.alpha_composite(glow.filter(ImageFilter.GaussianBlur(6)))
    d = ImageDraw.Draw(base)
    # 標題
    cap_s = "從非結構化資料到大模型 · Unstructured Data → LLMs"
    d.text(((GW - text_w(d, cap_s, cap)) // 2, 26), cap_s, font=cap, fill=MUTED + (255,))
    # 卡片
    for i, (name, kind, col) in enumerate(CARDS):
        x = x0 + i * (cardW + gap)
        lit = (active == i) or (active >= 4)
        if lit:
            gl = Image.new("RGBA", (GW, GH), (0, 0, 0, 0))
            ImageDraw.Draw(gl).rounded_rectangle([x, cy - cardH // 2, x + cardW, cy + cardH // 2],
                                                 radius=22, fill=col + (130,))
            base.alpha_composite(gl.filter(ImageFilter.GaussianBlur(16)))
        d = ImageDraw.Draw(base)
        fillc = col + (60,) if lit else col + (24,)
        outc = col + (255,) if lit else col + (80,)
        d.rounded_rectangle([x, cy - cardH // 2, x + cardW, cy + cardH // 2], radius=22,
                            fill=fillc, outline=outc, width=4 if lit else 2)
        icon(d, kind, x + cardW // 2, cy - 14,
             WHITE if lit else (120, 140, 160), 255 if lit else 130)
        d.text((x + (cardW - text_w(d, name, lab)) // 2, cy + 40), name, font=lab,
               fill=WHITE + (255,) if lit else MUTED + (200,))
    # 箭頭
    d.text((model_cx - 138, cy - 22), "→", font=font(48), fill=WHITE + (220,))
    # 模型節點
    ml = Image.new("RGBA", (GW, GH), (0, 0, 0, 0))
    draw_model(ml, model_cx, model_cy, lit=(active >= 4))
    if active >= 4:
        base.alpha_composite(ml.filter(ImageFilter.GaussianBlur(14)))
    base.alpha_composite(ml)
    d = ImageDraw.Draw(base)
    mcol = WHITE if active >= 4 else MUTED
    d.text((model_cx - text_w(d, "大模型", lab) // 2, model_cy + 84), "大模型", font=lab, fill=mcol + (255,))
    return base.convert("P", palette=Image.ADAPTIVE, colors=128)

frames = [make_frame(i) for i in [0, 1, 2, 3, 4, 4]]
frames[0].save(f"{ASSETS}/showcase.gif", save_all=True, append_images=frames[1:],
               duration=[850, 850, 850, 850, 1100, 1100], loop=0, optimize=False, disposal=1)
print("✓ assets/showcase.gif", (GW, GH))
