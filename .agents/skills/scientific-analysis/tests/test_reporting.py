"""Check offline exploration and provenance through an actual browser."""

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reporting", ROOT / "scripts" / "reporting.py")
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def launch_browser(runtime):
    executable = os.environ.get("CHROME_EXECUTABLE") or next(
        (path for name in ("google-chrome", "chromium", "chromium-browser")
         if (path := shutil.which(name))), None
    )
    if not executable and not Path(runtime.chromium.executable_path).exists():
        pytest.skip("No Chromium browser: run playwright install chromium or set CHROME_EXECUTABLE")
    return runtime.chromium.launch(
        executable_path=executable, headless=True, args=["--no-sandbox"]
    )


@pytest.fixture
def payload():
    return {
        "run_id": "test-run",
        "data_hash": "sha256-example",
        "target": "price",
        "numeric_columns": ["mileage", "price"],
        "column_units": {"price": "GBP"},
        "findings": ["觀察不是因果證據", "</script><script>window.pwned=true</script>"],
        "metrics": {"primary_metric": "mae", "experiments": [
            {"model": "ridge", "variant": "baseline", "aggregate": {"mae": {"mean": 2.5, "std": 0.2}}},
            {"model": "ridge", "variant": "v1", "aggregate": {"mae": {"mean": 2.3, "std": 0.3}}},
        ]},
        "interpretations": [
            {"model": "linear", "variant": "baseline", "fold_id": 1,
             "limitation": "Fold-specific, standardized input; not causal.",
             "coefficients": [{"class_index": 0, "terms": [{"feature": "mileage", "coefficient": -3.0}]}]},
            {"model": "shallow_tree", "variant": "baseline", "fold_id": 1,
             "rules": "|--- mileage <= 20\n|   |--- value: 90"},
        ],
        "rows": [
            {"row_id": "r1", "mileage": 10, "price": 100, "brand": "A"},
            {"row_id": "r2", "mileage": 20, "price": 90, "brand": "B"},
            {"row_id": "r3", "mileage": 30, "price": 80, "brand": "A"},
        ],
    }


def test_report_refuses_duplicate_identity_and_overwrite(tmp_path, payload):
    target = tmp_path / "report.html"
    reporting.render_report(payload, target)
    with pytest.raises(FileExistsError):
        reporting.render_report(payload, target)
    payload["rows"].append(payload["rows"][0])
    with pytest.raises(ValueError, match="unique"):
        reporting.render_report(payload, tmp_path / "bad.html")


def test_report_embeds_data_safely_and_normalizes_nonfinite(tmp_path, payload):
    payload["rows"][0]["price"] = float("nan")
    target = tmp_path / "report.html"
    reporting.render_report(payload, target)
    html = target.read_text()
    start = html.index('<script type="application/json" id="report-data">')
    data = html[start:].split(">", 1)[1].split("</script>", 1)[0]
    decoded = json.loads(data)
    assert decoded["rows"][0]["price"] is None
    assert "<script>" not in data
    assert decoded["findings"] == payload["findings"]


def test_offline_linking_filter_export_and_reset(tmp_path, payload):
    from playwright.sync_api import sync_playwright

    target = tmp_path / "report.html"
    reporting.render_report(payload, target)
    with sync_playwright() as runtime:
        browser = launch_browser(runtime)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, external_requests = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("http://**/*", lambda route: (external_requests.append(route.request.url), route.abort()))
        page.route("https://**/*", lambda route: (external_requests.append(route.request.url), route.abort()))
        page.goto(target.as_uri())
        page.wait_for_function("window.researchReport?.ready === true")
        assert page.evaluate("window.pwned") is None
        assert "完整展示 3 列" in page.locator("#status").inner_text()
        assert page.locator("#scatter .main-svg").count() > 0
        assert page.evaluate("document.getElementById('scatter').layout.yaxis.title.text") == "price (GBP)"
        assert page.evaluate("document.getElementById('scatter').layout.xaxis.title.text") == "mileage (單位未提供)"
        assert "price (GBP)" in page.evaluate("document.getElementById('scatter').data[0].hovertemplate")
        assert "mileage (單位未提供)" in page.evaluate("document.getElementById('scatter').data[0].hovertemplate")
        assert page.locator("#model-panel").is_visible()
        assert page.evaluate("document.getElementById('model-comparison').data[0].x") == [2.5, 2.3]
        assert page.evaluate("document.getElementById('coefficient-plot').data[0].x") == [-3.0]
        page.select_option("#interpretation-select", "1")
        assert "mileage <= 20" in page.locator("#tree-rules").inner_text()
        page.select_option("#category-column", "brand")
        page.select_option("#category-value", "A")
        page.click("#apply-filter")
        assert "目前篩選 2 列" in page.locator("#status").inner_text()
        assert "原研究結論未重新驗證" in page.locator("#status").inner_text()
        page.evaluate("document.getElementById('scatter').emit('plotly_selected', {points:[{customdata:'r3'}]})")
        assert "選取 1 列" in page.locator("#status").inner_text()
        assert page.locator("#data-table tr.selected").count() == 1
        assert page.evaluate("document.getElementById('distribution').data[2].x") == [30]
        page.fill("#notes", "追查 A 品牌的低價樣本")
        with page.expect_download() as download_info:
            page.click("#export-state")
        exported = json.loads(Path(download_info.value.path()).read_text())
        assert exported["data_hash"] == payload["data_hash"]
        assert exported["run_id"] == payload["run_id"]
        assert exported["selected_row_ids"] == ["r3"]
        assert exported["filters"]["category"] == {"column": "brand", "value": "A"}
        assert exported["notes"] == "追查 A 品牌的低價樣本"
        page.click("#reset")
        assert "目前篩選 3 列 ｜選取 0 列" in page.locator("#status").inner_text()
        page.select_option("#numeric-column", "mileage")
        page.fill("#filter-min", "999")
        page.click("#apply-filter")
        assert "目前篩選 0 列" in page.locator("#status").inner_text()
        assert page.evaluate("window.researchReport.exportState().scope.filtered_row_count") == 0
        page.screenshot(path=str(tmp_path / "report-browser.png"), full_page=True)
        assert not external_requests
        assert not errors
        browser.close()


def test_resumed_report_restores_exploratory_state(tmp_path, payload):
    from playwright.sync_api import sync_playwright

    coefficients = payload["interpretations"][0]["coefficients"][0]
    coefficients["terms"] = [{"feature": f"x{i}", "coefficient": float(i)} for i in range(25)]
    coefficients["total_terms"] = 25
    payload["parent_run_id"] = "parent-run"
    payload["resume_state"] = {
        "run_id": "parent-run", "data_hash": payload["data_hash"],
        "selected_row_ids": ["r3"], "notes": "這是探索，不是新驗證",
        "filters": {"numeric": {"column": "mileage", "min": 20, "max": None},
                    "category": {"column": "brand", "value": "A"}},
        "view": {"x": "price", "y": "mileage", "group": "brand"},
    }
    target = tmp_path / "resumed.html"
    reporting.render_report(payload, target)
    with sync_playwright() as runtime:
        browser = launch_browser(runtime)
        page = browser.new_page()
        page.goto(target.as_uri())
        page.wait_for_function("window.researchReport?.ready")
        assert page.locator("#notes").input_value() == "這是探索，不是新驗證"
        assert "parent-run，這不是獨立驗證" in page.locator("#provenance").inner_text()
        assert "前 20 項／25 項完整模型係數" in page.locator("#interpretation-caption").inner_text()
        assert page.evaluate("document.getElementById('coefficient-plot').data[0].x") == list(range(24, 4, -1))
        assert "目前篩選 1 列 ｜選取 1 列" in page.locator("#status").inner_text()
        state = page.evaluate("window.researchReport.exportState()")
        assert state["filtered_row_ids"] == ["r3"]
        assert state["selected_row_ids"] == ["r3"]
        assert state["view"]["x"] == "price"
        browser.close()
