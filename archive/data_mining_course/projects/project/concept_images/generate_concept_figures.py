"""Generate bilingual (zh-TW + en) concept figures for car_market_eda.ipynb."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge

OUT_DIR = Path(__file__).resolve().parent
FONT = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msjh.ttc")
FONT_BOLD = font_manager.FontProperties(fname=r"C:\Windows\Fonts\msjhbd.ttc")

NAVY = "#1B365D"
GREEN = "#1F6B4A"
RED = "#C0392B"
INK = "#1A2332"
MUTED = "#5B6573"
LINE = "#D5DBE3"
CARD = "#FFFFFF"
PAGE = "#F7F8FA"
BLUE = "#2F6FED"
ORANGE = "#E67E22"
PURPLE = "#7B61FF"
TEAL = "#1AA6A6"
Z_TEXT = 10


def text(ax, x, y, s, *, size=11, bold=False, color=INK, ha="left", va="center", **kwargs):
    return ax.text(
        x,
        y,
        s,
        fontproperties=FONT_BOLD if bold else FONT,
        fontsize=size,
        color=color,
        ha=ha,
        va=va,
        zorder=Z_TEXT,
        **kwargs,
    )


def card(ax, x, y, w, h, *, fc=CARD, ec=LINE, lw=1.2, radius=0.018):
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.006, y - 0.008),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor="#E6E9EE",
            edgecolor="none",
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=2,
        )
    )


def chip(ax, x, y, w, h, fc="#EEF3FA"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=fc,
            edgecolor="none",
            zorder=4,
        )
    )


def badge(ax, x, y, w, h, text_zh, text_en, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=color,
            edgecolor="none",
            zorder=5,
        )
    )
    text(ax, x + w / 2, y + h * 0.62, text_zh, size=12, bold=True, color="white", ha="center")
    text(ax, x + w / 2, y + h * 0.28, text_en, size=8.5, color="#E8EEF5", ha="center")


def new_fig(w=14.6, h=8.5):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(PAGE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PAGE)
    return fig, ax


def save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", facecolor=PAGE, pad_inches=0.14)
    plt.close(fig)
    return out


def fig1_two_sources():
    fig, ax = new_fig(14.8, 8.8)

    text(ax, 0.50, 0.955, "兩份資料，角色不同", size=20, bold=True, ha="center")
    text(
        ax,
        0.50,
        0.912,
        "Two sources, two roles    |    不要 concat    /    Do not concatenate",
        size=11,
        color=MUTED,
        ha="center",
    )

    card(ax, 0.03, 0.075, 0.405, 0.805, ec="#C5D4EA")
    ax.add_patch(Rectangle((0.03, 0.780), 0.405, 0.008, facecolor=NAVY, edgecolor="none", zorder=4))

    ax.add_patch(
        FancyBboxPatch(
            (0.048, 0.800),
            0.036,
            0.022,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor="#F4C430",
            edgecolor="none",
            zorder=5,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.048, 0.782),
            0.050,
            0.026,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor="#F7D45C",
            edgecolor="none",
            zorder=5,
        )
    )
    text(ax, 0.112, 0.805, "car_data/", size=15, bold=True, color=NAVY)
    text(ax, 0.112, 0.772, "99,187 筆英國上架    |    99,187 UK listings", size=9.2, color=MUTED)

    files = [
        "audi.csv",
        "bmw.csv",
        "ford.csv",
        "hyundai.csv",
        "merc.csv",
        "skoda.csv",
        "toyota.csv",
        "vauxhall.csv",
        "vw.csv",
    ]
    fields = [
        ("model", "車型"),
        ("year", "年份"),
        ("price", "價格"),
        ("transmission", "變速箱"),
        ("mileage", "里程"),
        ("fuelType", "燃料"),
        ("tax", "稅金"),
        ("mpg", "油耗"),
        ("engineSize", "排量"),
    ]

    text(ax, 0.050, 0.738, "九個品牌檔    |    9 brand files", size=9, bold=True, color=NAVY)
    for i, name in enumerate(files):
        y = 0.685 - i * 0.052
        chip(ax, 0.048, y - 0.016, 0.158, 0.034, "#EEF3FA")
        text(ax, 0.060, y, "-", size=10, color=BLUE)
        text(ax, 0.078, y, name, size=9.4, color=INK)

    text(ax, 0.230, 0.738, "欄位（每個檔相同）", size=9, bold=True, color=NAVY)
    text(ax, 0.230, 0.714, "Fields (same per file)", size=8, color=MUTED)
    for i, (en, zh) in enumerate(fields):
        y = 0.680 - i * 0.048
        text(ax, 0.230, y, en, size=9.6, bold=True, color=INK)
        text(ax, 0.355, y, zh, size=9.6, color=MUTED)

    badge(ax, 0.048, 0.098, 0.370, 0.070, "主市場 EDA", "MAIN MARKET EDA", NAVY)

    card(ax, 0.565, 0.075, 0.405, 0.805, ec="#C9E4D4")
    ax.add_patch(Rectangle((0.565, 0.780), 0.405, 0.008, facecolor=GREEN, edgecolor="none", zorder=4))
    ax.add_patch(
        FancyBboxPatch(
            (0.585, 0.782),
            0.042,
            0.046,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor="#E8F6EE",
            edgecolor="#B7DCC6",
            zorder=5,
        )
    )
    text(ax, 0.606, 0.805, "CSV", size=7, bold=True, color=GREEN, ha="center")
    text(ax, 0.640, 0.805, "Automobile_data.csv", size=13.2, bold=True, color=GREEN)
    text(ax, 0.640, 0.772, "205 輛    |    205 cars    |    UCI 規格 / 保險表", size=9.2, color=MUTED)

    text(ax, 0.588, 0.722, "本 notebook 只用規格欄", size=10.5, bold=True, color=GREEN)
    text(ax, 0.588, 0.696, "Used here as a spec sheet only", size=9.2, color=MUTED)

    spec_fields = [
        ("bore", "缸徑"),
        ("stroke", "衝程"),
        ("num-of-cylinders", "缸數"),
        ("engine-size", "排量"),
        ("symboling", "保險風險等級"),
        ("normalized-losses", "標準化損失"),
    ]
    for i, (en, zh) in enumerate(spec_fields):
        y = 0.632 - i * 0.062
        chip(ax, 0.585, y - 0.022, 0.365, 0.048, "#F3FAF6")
        text(ax, 0.602, y, en, size=11, bold=True, color=INK)
        text(ax, 0.930, y, zh, size=11, color=MUTED, ha="right")

    badge(ax, 0.585, 0.098, 0.365, 0.070, "僅規格表", "SPEC SHEET ONLY", GREEN)

    ax.add_patch(Circle((0.50, 0.52), 0.040, facecolor="#FDECEC", edgecolor="#F5C6C2", lw=1.5, zorder=5))
    text(ax, 0.50, 0.528, "X", size=20, bold=True, color=RED, ha="center")
    text(ax, 0.50, 0.430, "不要 concat", size=11, bold=True, color=RED, ha="center")
    text(ax, 0.50, 0.398, "DO NOT CONCAT", size=8.5, color=RED, ha="center")

    return save(fig, "fig1_two_sources_zh_en.png")


def icon_tag(ax, x, y, color):
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.016, y - 0.016),
            0.024,
            0.032,
            boxstyle="round,pad=0,rounding_size=0.004",
            facecolor=color,
            edgecolor="none",
            zorder=6,
        )
    )
    ax.add_patch(Circle((x + 0.014, y + 0.004), 0.012, facecolor="#EEF3FA", edgecolor=color, lw=1.6, zorder=6))


def icon_cal(ax, x, y, color):
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.020, y - 0.016),
            0.040,
            0.034,
            boxstyle="round,pad=0,rounding_size=0.005",
            facecolor=color,
            edgecolor="none",
            zorder=6,
        )
    )
    text(ax, x, y + 0.001, "31", size=8, bold=True, color="white", ha="center")


def icon_engine(ax, x, y, color):
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.024, y - 0.012),
            0.048,
            0.024,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=color,
            edgecolor="none",
            zorder=6,
        )
    )
    ax.add_patch(Circle((x - 0.010, y), 0.006, facecolor="white", edgecolor="none", zorder=7))
    ax.add_patch(Circle((x + 0.010, y), 0.006, facecolor="white", edgecolor="none", zorder=7))


def icon_pie(ax, x, y):
    ax.add_patch(Wedge((x, y), 0.020, 0, 130, facecolor="#3CB371", edgecolor="white", lw=1, zorder=6))
    ax.add_patch(Wedge((x, y), 0.020, 130, 210, facecolor=PURPLE, edgecolor="white", lw=1, zorder=6))
    ax.add_patch(Wedge((x, y), 0.020, 210, 300, facecolor=BLUE, edgecolor="white", lw=1, zorder=6))
    ax.add_patch(Wedge((x, y), 0.020, 300, 360, facecolor=ORANGE, edgecolor="white", lw=1, zorder=6))


def icon_bars(ax, x, y, color):
    for i, h in enumerate((0.012, 0.018, 0.026, 0.034)):
        ax.add_patch(Rectangle((x - 0.020 + i * 0.011, y - 0.016), 0.008, h, facecolor=color, edgecolor="none", zorder=6))


def fig2_five_questions():
    fig, ax = new_fig(14.9, 8.3)

    text(ax, 0.50, 0.940, "英國二手車市場 — 什麼驅動上架價？", size=20, bold=True, ha="center")
    text(
        ax,
        0.50,
        0.892,
        "UK used-car market  —  what drives listing price?",
        size=12,
        color=MUTED,
        ha="center",
    )

    items = [
        {
            "color": BLUE,
            "zh": "品牌價差",
            "en": "Brand price gap",
            "zh_body": "豪華品牌掛價\n高於主流品牌。",
            "en_body": "Premium brands list\nhigher than mainstream.",
            "icon": "tag",
        },
        {
            "color": "#2E8B57",
            "zh": "車齡與里程折舊",
            "en": "Age and mileage",
            "zh_body": "越舊、里程越高，\n上架價越低。",
            "en_body": "Older cars and higher\nmileage mean lower prices.",
            "icon": "cal",
        },
        {
            "color": ORANGE,
            "zh": "引擎 / 燃料 / 變速箱",
            "en": "Engine, fuel, gearbox",
            "zh_body": "動力總成影響需求、\n使用成本與殘值。",
            "en_body": "Powertrain affects demand,\nrunning cost and resale.",
            "icon": "engine",
        },
        {
            "color": PURPLE,
            "zh": "燃料結構逐年變化",
            "en": "Fuel mix over years",
            "zh_body": "燃料偏好改變，\n會拉動需求與掛價。",
            "en_body": "Shifting fuel mix\ninfluences demand and prices.",
            "icon": "pie",
        },
        {
            "color": TEAL,
            "zh": "最強價格訊號",
            "en": "Strongest price signal",
            "zh_body": "這份資料裡，\n排量是最強正向訊號。",
            "en_body": "Here, engine size is\nthe strongest positive signal.",
            "icon": "bars",
        },
    ]

    left, gap = 0.026, 0.012
    w = (1 - 2 * left - 4 * gap) / 5
    y0, h = 0.085, 0.745

    for i, item in enumerate(items):
        x = left + i * (w + gap)
        cx = x + w / 2
        card(ax, x, y0, w, h, ec="#E1E5EC")
        ax.add_patch(Rectangle((x, y0 + h - 0.008), w, 0.008, facecolor=item["color"], edgecolor="none", zorder=4))

        ax.add_patch(Circle((cx, 0.755), 0.026, facecolor=item["color"], edgecolor="none", zorder=6))
        text(ax, cx, 0.755, str(i + 1), size=15, bold=True, color="white", ha="center")

        text(ax, cx, 0.680, item["zh"], size=12, bold=True, ha="center")
        text(ax, cx, 0.638, item["en"], size=8.6, color=MUTED, ha="center")

        if item["icon"] == "tag":
            icon_tag(ax, cx, 0.555, item["color"])
        elif item["icon"] == "cal":
            icon_cal(ax, cx, 0.555, item["color"])
        elif item["icon"] == "engine":
            icon_engine(ax, cx, 0.555, item["color"])
        elif item["icon"] == "pie":
            icon_pie(ax, cx, 0.555)
        else:
            icon_bars(ax, cx, 0.555, item["color"])

        ax.plot([x + 0.022, x + w - 0.022], [0.490, 0.490], color=LINE, lw=1, zorder=4)
        text(ax, cx, 0.375, item["zh_body"], size=10.6, color=INK, ha="center", va="center", linespacing=1.45)
        text(ax, cx, 0.230, item["en_body"], size=8.5, color=MUTED, ha="center", va="center", linespacing=1.4)

    return save(fig, "fig2_five_questions_zh_en.png")


def dim_arrow(ax, x1, y1, x2, y2, color):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.7),
        zorder=7,
    )


def fig3_engine_displacement():
    fig, ax = new_fig(14.9, 8.7)

    text(ax, 0.50, 0.955, "引擎排量 — 單缸", size=20, bold=True, ha="center")
    text(ax, 0.50, 0.912, "Engine displacement  —  one cylinder", size=12, color=MUTED, ha="center")

    card(ax, 0.03, 0.07, 0.46, 0.81, ec="#D5DBE3")

    cx = 0.175
    bore = 0.050
    cyl_top, cyl_bot = 0.78, 0.36
    piston_y, piston_h = 0.52, 0.068

    # outer block
    ax.add_patch(
        FancyBboxPatch(
            (cx - bore - 0.018, cyl_bot),
            2 * bore + 0.036,
            cyl_top - cyl_bot + 0.02,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor="#E4E9F0",
            edgecolor="#7B8794",
            lw=2.0,
            zorder=3,
        )
    )
    # bore cavity
    ax.add_patch(
        Rectangle(
            (cx - bore, cyl_bot + 0.012),
            2 * bore,
            cyl_top - cyl_bot - 0.018,
            facecolor="#F4F7FB",
            edgecolor="#8A96A8",
            lw=1.4,
            zorder=4,
        )
    )
    # swept volume (TDC to current piston top) — the pedagogical core
    ax.add_patch(
        Rectangle(
            (cx - bore + 0.002, piston_y + piston_h),
            2 * bore - 0.004,
            0.70 - (piston_y + piston_h),
            facecolor="#D7E8FF",
            edgecolor="none",
            zorder=4.5,
        )
    )
    # spark plug
    ax.add_patch(Rectangle((cx - 0.005, cyl_top + 0.012), 0.010, 0.028, facecolor="#6B7280", edgecolor="none", zorder=5))
    ax.add_patch(Circle((cx, cyl_top + 0.044), 0.009, facecolor="#9CA3AF", edgecolor="#4B5563", lw=1, zorder=5))

    # piston
    ax.add_patch(
        FancyBboxPatch(
            (cx - bore + 0.003, piston_y),
            2 * bore - 0.006,
            piston_h,
            boxstyle="round,pad=0,rounding_size=0.005",
            facecolor="#6B7280",
            edgecolor="#374151",
            lw=1.2,
            zorder=5,
        )
    )
    ax.add_patch(Circle((cx, piston_y + piston_h / 2), 0.010, facecolor="#D1D5DB", edgecolor="#374151", lw=1, zorder=6))

    crank_y = 0.250
    ax.plot([cx, cx], [piston_y, crank_y + 0.028], color="#374151", lw=3.4, zorder=5, solid_capstyle="round")
    ax.add_patch(Circle((cx, crank_y), 0.034, facecolor="#E5E7EB", edgecolor="#374151", lw=2, zorder=5))
    ax.add_patch(Circle((cx, crank_y), 0.010, facecolor="#9CA3AF", edgecolor="#374151", lw=1, zorder=6))

    # Bore dimension: above cylinder, not overlapping walls
    by = 0.835
    dim_arrow(ax, cx - bore, by, cx + bore, by, BLUE)
    text(ax, cx, 0.860, "缸徑   Bore", size=11, bold=True, color=BLUE, ha="center")
    text(ax, 0.050, 0.640, "掃過體積", size=8.5, bold=True, color=BLUE)
    text(ax, 0.050, 0.618, "swept volume", size=7.5, color=MUTED)

    # Stroke dimension to the right of cylinder
    tdc_y = 0.70
    bdc_y = 0.43
    sx = cx + bore + 0.035
    ax.plot([cx + bore, sx + 0.008], [tdc_y, tdc_y], color=GREEN, lw=1.0, zorder=6)
    ax.plot([cx + bore, sx + 0.008], [bdc_y, bdc_y], color=GREEN, lw=1.0, zorder=6)
    dim_arrow(ax, sx, tdc_y, sx, bdc_y, GREEN)

    lx = 0.30
    text(ax, lx, 0.700, "上死點  TDC", size=10, bold=True, color=GREEN)
    text(ax, lx, 0.675, "Top Dead Center", size=8, color=MUTED)
    text(ax, lx, 0.575, "衝程  Stroke", size=11, bold=True, color=GREEN)
    text(ax, lx, 0.548, "活塞行程  /  piston travel", size=8.2, color=MUTED)
    text(ax, lx, 0.445, "下死點  BDC", size=10, bold=True, color=GREEN)
    text(ax, lx, 0.420, "Bottom Dead Center", size=8, color=MUTED)

    text(ax, 0.050, 0.145, "排量 = 活塞從 TDC 走到 BDC 掃過的體積", size=9.2, color=INK)
    text(ax, 0.050, 0.112, "Displacement = volume swept by the piston from TDC to BDC", size=8.2, color=MUTED)

    card(ax, 0.52, 0.50, 0.45, 0.38, ec="#C5D4EA")
    text(ax, 0.545, 0.835, "單缸排量", size=13.5, bold=True, color=NAVY)
    text(ax, 0.545, 0.800, "Displacement of one cylinder", size=9.5, color=MUTED)
    text(ax, 0.745, 0.735, r"$V_1=\pi\times(Bore/2)^2\times Stroke$", size=14.5, ha="center", color=NAVY)
    text(ax, 0.545, 0.655, "Bore     缸徑 = 汽缸內徑", size=10.5, color=BLUE)
    text(ax, 0.545, 0.615, "Stroke   衝程 = 活塞行程", size=10.5, color=GREEN)
    text(ax, 0.545, 0.575, "V1       單缸排量 = displacement of one cylinder", size=10.2, color=INK)
    text(ax, 0.545, 0.535, "pi  ~  3.14159", size=10.2, color=MUTED)

    card(ax, 0.52, 0.07, 0.45, 0.39, ec="#C9E4D4")
    text(ax, 0.545, 0.415, "總排量", size=13.5, bold=True, color=GREEN)
    text(ax, 0.545, 0.380, "Total engine displacement", size=9.5, color=MUTED)
    text(ax, 0.745, 0.315, r"$V_{total}=V_1\times N$", size=16, ha="center", color=GREEN)
    text(ax, 0.545, 0.240, "V_total   總排量 = total displacement", size=10.2, color=INK)
    text(ax, 0.545, 0.200, "V1        單缸排量 = displacement of one cylinder", size=10.2, color=INK)
    text(ax, 0.545, 0.160, "N         缸數 = number of cylinders", size=10.2, color=INK)
    text(ax, 0.545, 0.115, "對應欄位：bore  x  stroke  x  num-of-cylinders", size=9.4, color=MUTED)

    return save(fig, "fig3_engine_displacement_zh_en.png")


if __name__ == "__main__":
    for path in (fig1_two_sources(), fig2_five_questions(), fig3_engine_displacement()):
        print(path.name, path.stat().st_size)
