"""Verify host generation and read-only drift detection without an analysis runtime."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.render_skills import ROOT, render


class RenderSkillsTests(unittest.TestCase):
    def run_generator(self, output_root, mode, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "tools/render_skills.py"), mode,
             "--output-root", str(output_root), *args],
            capture_output=True, text=True, check=False,
        )

    def test_both_hosts_and_read_only_drift_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_generator(root, "--write")
            self.assertEqual(result.returncode, 0, result.stderr)
            for host in ("codex", "claude"):
                for path, expected in render(host).items():
                    self.assertEqual((root / path).read_text(), expected)
            result = self.run_generator(root, "--check")
            self.assertEqual(result.returncode, 0, result.stderr)
            path = root / ".agents/skills/ds-frame/SKILL.md"
            changed = path.read_text() + "\nLocal drift\n"
            path.write_text(changed)
            result = self.run_generator(root, "--check")
            self.assertEqual(result.returncode, 1)
            self.assertIn(".agents/skills/ds-frame/SKILL.md", result.stderr)
            self.assertEqual(path.read_text(), changed)

    def test_missing_output_check_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_generator(tmp, "--check")
            self.assertEqual(result.returncode, 1)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_selected_host_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_file = root / ".agents/skills/README.md"
            user_file.parent.mkdir(parents=True)
            user_file.write_text("User guide\n")
            result = self.run_generator(root, "--write", "--host", "codex")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(user_file.read_text(), "User guide\n")
            self.assertFalse((root / ".claude").exists())

    def test_discovers_templates_and_adapts_host_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "skills/custom/SKILL.md.tmpl"
            source.parent.mkdir(parents=True)
            source.write_text(
                "---\nname: custom\ndescription: Test role\n---\n"
                "{{SESSION_START}}\nUse {{INVOKE}}custom.\n"
            )
            for host, folder, prefix in (
                ("codex", ".agents", "$"), ("claude", ".claude", "/")
            ):
                result = render(host, root)
                self.assertEqual(len(result), 1)
                text = result[Path(folder) / "skills/custom/SKILL.md"]
                self.assertIn(f"Use {prefix}custom.", text)
                self.assertNotIn("{{", text)

    def test_invalid_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                render("codex", root)
            source = root / "skills/custom/SKILL.md.tmpl"
            source.parent.mkdir(parents=True)
            source.write_text("---\nname: wrong\n---\nBody\n")
            with self.assertRaisesRegex(ValueError, "名稱與資料夾不符"):
                render("codex", root)
            source.write_text("---\nname: custom\n---\n{{UNKNOWN}}\n")
            with self.assertRaisesRegex(ValueError, "未解析"):
                render("codex", root)


if __name__ == "__main__":
    unittest.main()
