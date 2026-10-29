"""Protocol tests use fixture prose, not genuine analytical or model reviews."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "ds_harness.py"


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = self.root / "case"
        self.document = self.root / "evidence.md"
        self.document.write_text("Protocol fixture only. No scientific claim.", encoding="utf-8")
        self.report = self.root / "decision.md"
        self.report.write_text("Protocol delivery fixture; not a real decision.", encoding="utf-8")
        self.review = self.root / "review.json"
        self.write_review()
        self.call("init", "--goal", "eda")

    def call(self, command, *args, expected=0):
        process = subprocess.run([sys.executable, str(CLI), command, "--case", str(self.case),
                                  *map(str, args)], capture_output=True, text=True)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return json.loads(process.stdout if expected == 0 else process.stderr)

    def session(self, skill, host="codex"):
        return self.call("start", "--skill", skill, "--host", host)["session_id"]

    def record(self, stage, inputs=()):
        args = ["--stage", stage, "--session", self.session("ds-" + stage), "--file", self.document]
        for path in inputs:
            args.extend(["--input", path])
        return self.call("record", *args)

    def ready(self):
        self.record("frame")
        self.record("eda")

    def write_review(self, findings=None, result="pass"):
        self.review.write_text(json.dumps({
            "summary": "Protocol fixture, not an independent or scientific review.",
            "checks": {key: {"result": result, "evidence": "Protocol fixture only"}
                       for key in ("scope", "leakage", "statistics", "business")},
            "findings": findings or [],
        }), encoding="utf-8")

    def begin_review(self):
        return self.call("review-start", "--session", self.session("ds-review", "claude"),
                         "--file", self.report)["review_token"]

    def finish_review(self, receipt=None, verdict="pass", expected=0):
        receipt = receipt or self.begin_review()
        return self.call("review-finish", "--token", receipt, "--verdict", verdict,
                         "--file", self.review, expected=expected)

    def test_cross_host_resume_and_eda_only_delivery(self):
        self.ready()
        state = self.call("status")
        self.assertEqual(state["stages"]["proxy"]["state"], "missing")
        self.assertEqual(state["stages"]["rules"]["state"], "missing")
        self.assertFalse(state["review_current"])
        self.finish_review()
        result = self.call("deliver", "--session", self.session("ds-deliver", "claude"),
                           "--file", self.report)
        self.assertTrue((self.case / result["delivery"]["path"]).is_file())
        resumed = self.call("status")
        self.assertTrue(resumed["delivery_current"])
        self.assertEqual({s["host"] for s in resumed["sessions"].values()}, {"claude", "codex"})

    def test_required_upstream_cannot_be_bypassed(self):
        result = self.call("record", "--stage", "rules", "--session", self.session("ds-rules"),
                           "--file", self.document, expected=2)
        self.assertEqual(result["code"], "UPSTREAM_NOT_READY")
        self.assertEqual(self.call("status")["stages"]["rules"]["state"], "missing")

    def test_wrong_role_cannot_record_stage(self):
        result = self.call("record", "--stage", "frame", "--session", self.session("ds-eda"),
                           "--file", self.document, expected=2)
        self.assertEqual(result["code"], "WRONG_SESSION")

    def test_input_changes_invalidate_review_and_delivery(self):
        raw = self.root / "data.txt"
        raw.write_text("version one")
        self.record("frame")
        self.record("eda", [raw])
        self.finish_review()
        raw.write_text("version two")
        state = self.call("status")
        self.assertEqual(state["stages"]["eda"]["state"], "stale")
        self.assertFalse(state["review_current"])
        result = self.call("deliver", "--session", self.session("ds-deliver"),
                           "--file", self.report, expected=2)
        self.assertEqual(result["code"], "NO_CURRENT_REVIEW")

    def test_upstream_changes_invalidate_transitive_evidence(self):
        self.ready()
        self.record("rules")
        self.record("impact")
        self.record("frame")
        state = self.call("status")
        self.assertEqual(state["stages"]["frame"]["state"], "current")
        for key in ("eda", "rules", "impact"):
            self.assertEqual(state["stages"][key]["state"], "stale")
        self.assertEqual(state["next_skill"], "ds-eda")

    def test_adding_optional_proxy_invalidates_previous_eda(self):
        self.ready()
        self.record("proxy")
        self.assertEqual(self.call("status")["stages"]["eda"]["state"], "stale")

    def test_review_cannot_finish_on_changed_candidate(self):
        self.ready()
        receipt = self.begin_review()
        self.record("eda")
        result = self.finish_review(receipt, expected=2)
        self.assertEqual(result["code"], "STALE_REVIEW")

    def test_changed_final_report_requires_new_review(self):
        self.ready()
        self.finish_review()
        self.report.write_text("New unreviewed conclusion.")
        result = self.call("deliver", "--session", self.session("ds-deliver"),
                           "--file", self.report, expected=2)
        self.assertEqual(result["code"], "REPORT_NOT_REVIEWED")

    def test_open_major_and_failed_check_cannot_receive_pass(self):
        self.ready()
        self.write_review(findings=[{"detail": "Fixture unresolved problem", "severity": "major", "resolution": "open"}])
        self.assertEqual(self.finish_review(expected=2)["code"], "OPEN_FINDINGS")
        self.write_review(result="fail")
        self.assertEqual(self.finish_review(expected=2)["code"], "OPEN_FINDINGS")

    def test_later_revise_revokes_previous_pass(self):
        self.ready()
        self.finish_review()
        self.assertTrue(self.call("status")["review_current"])
        self.finish_review(verdict="revise")
        self.assertFalse(self.call("status")["review_current"])

    def test_receipt_cannot_be_replayed(self):
        self.ready()
        receipt = self.begin_review()
        self.finish_review(receipt)
        self.assertEqual(self.finish_review(receipt, expected=2)["code"], "BAD_REVIEW_TOKEN")

    def test_modified_snapshot_invalidates_review(self):
        self.ready()
        self.finish_review()
        state = self.call("status")
        (self.case / state["stages"]["eda"]["path"]).write_text("edited snapshot")
        self.assertFalse(self.call("status")["review_current"])

    def test_goal_and_read_only_status(self):
        before = (self.case / "state.sqlite").read_bytes()
        self.assertEqual(self.call("status")["next_skill"], "ds-frame")
        self.assertEqual(before, (self.case / "state.sqlite").read_bytes())
        self.call("init", expected=2)

    def test_failed_review_does_not_consume_token(self):
        self.ready()
        receipt = self.begin_review()
        self.review.write_text('{"summary": "invalid schema"}')
        self.assertEqual(self.finish_review(receipt, expected=2)["code"], "BAD_REVIEW")
        self.write_review()
        self.finish_review(receipt)
        self.assertTrue(self.call("status")["review_current"])

    def test_impact_goal_requires_impact_but_not_rules(self):
        self.case = self.root / "impact-case"
        self.call("init", "--goal", "impact")
        self.ready()
        result = self.call("review-start", "--session", self.session("ds-review"),
                           "--file", self.report, expected=2)
        self.assertEqual(result["code"], "CANDIDATE_NOT_READY")
        self.record("impact")
        self.finish_review()
        self.assertEqual(self.call("status")["stages"]["rules"]["state"], "missing")
        self.assertTrue(self.call("status")["review_current"])

    def test_malformed_review_is_actionable_and_does_not_receive_pass(self):
        self.ready()
        self.review.write_text("[]")
        error = self.finish_review(expected=2)
        self.assertEqual(error["code"], "BAD_REVIEW")
        self.assertIn("recovery", error)
        self.assertFalse(self.call("status")["review_current"])


class RenderTests(unittest.TestCase):
    def test_host_render_and_drift_detection(self):
        renderer = ROOT / "tools" / "render_skills.py"
        with tempfile.TemporaryDirectory() as temp:
            command = [sys.executable, str(renderer), "--output-root", temp]
            subprocess.run(command + ["--write"], check=True, capture_output=True)
            subprocess.run(command + ["--check"], check=True, capture_output=True)
            claude = Path(temp) / ".claude/skills/ds-frame/SKILL.md"
            codex = Path(temp) / ".agents/skills/ds-frame/SKILL.md"
            self.assertTrue(claude.is_file() and codex.is_file())
            claude.write_text("local drift")
            failed = subprocess.run(command + ["--check"], capture_output=True, text=True)
            self.assertEqual(failed.returncode, 1)
            self.assertIn("ds-frame", failed.stderr)
            self.assertEqual(claude.read_text(), "local drift")


if __name__ == "__main__":
    unittest.main()
