"""Render a portable research report; all analysis remains in Python artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def render_report(payload: dict, output_path: Path) -> None:
    """Write one standalone HTML file without replacing an existing artifact."""
    from plotly.offline import get_plotlyjs

    rows = payload.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) or "row_id" not in row for row in rows):
        raise ValueError("Report rows must be dictionaries with stable row_id values")
    if len({str(row["row_id"]) for row in rows}) != len(rows):
        raise ValueError("Report row_id values must be unique")
    if not payload.get("run_id") or not payload.get("data_hash"):
        raise ValueError("Report requires run_id and data_hash provenance")
    data = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False, default=str)
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    data = data.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (Path(__file__).resolve().parents[1] / "assets" / "report.html").read_text()
    replacements = {"__DATA__": data, "__PLOTLY__": get_plotlyjs()}
    html = re.sub(r"__DATA__|__PLOTLY__", lambda match: replacements[match.group()], template)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(html)
