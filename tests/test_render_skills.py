"""Verify host generation and read-only drift detection without an analysis runtime."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.render_skills import ROOT, render


class RenderSkillsTests(unittest.TestCase):
    def test_native_command_names_are_rejected(self):
        for name in ('init', 'model', 'review', 'models', 'help'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'skills' / name / 'SKILL.md.tmpl'
                source.parent.mkdir(parents=True)
                source.write_text(f'---\nname: {name}\ndescription: Conflict\n---\nBody\n')
                with self.assertRaisesRegex(ValueError, '原生指令衝突'):
                    render('codex', root)

    def test_manual_policy_and_retired_names_absent(self):
        for host, folder in (("codex", ".agents"), ("claude", ".claude")):
            outputs = render(host)
            names = {path.parent.name for path in outputs if path.name == "SKILL.md"}
            self.assertEqual(names, {"lab-init", "research", "eda", "model-fit", "analysis-review", "report"})
            for name in names:
                base = Path(folder) / "skills" / name
                self.assertEqual("disable-model-invocation: true" in outputs[base / "SKILL.md"], name == "lab-init")
                if host == "codex":
                    expected = "false" if name == "lab-init" else "true"
                    self.assertIn("allow_implicit_invocation: " + expected, outputs[base / "agents/openai.yaml"])
            self.assertFalse(any(part.startswith("ds-") for path in outputs for part in path.parts))

    def run_generator(self, output_root, mode, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "tools/render_skills.py"), mode,
             "--output-root", str(output_root), *args],
            capture_output=True, text=True, check=False,
        )

    def test_all_hosts_and_read_only_drift_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_generator(root, "--write")
            self.assertEqual(result.returncode, 0, result.stderr)
            for host in ("codex", "claude", "opencode"):
                for path, expected in render(host).items():
                    self.assertEqual((root / path).read_text(), expected)
            result = self.run_generator(root, "--check")
            self.assertEqual(result.returncode, 0, result.stderr)
            path = root / ".agents/skills/research/SKILL.md"
            changed = path.read_text() + "\nLocal drift\n"
            path.write_text(changed)
            result = self.run_generator(root, "--check")
            self.assertEqual(result.returncode, 1)
            self.assertIn(".agents/skills/research/SKILL.md", result.stderr)
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
            self.assertFalse((root / ".opencode").exists())

    def test_opencode_commands_and_shared_skills_are_installed_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_command = root / ".opencode/commands/custom.md"
            user_command.parent.mkdir(parents=True)
            user_command.write_text("User command\n")
            result = self.run_generator(root, "--write", "--host", "opencode")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / ".claude").exists())
            self.assertFalse((root / ".opencode/skills").exists())
            self.assertEqual(user_command.read_text(), "User command\n")
            for path, expected in render("codex").items():
                self.assertEqual((root / path).read_text(), expected)
            commands = list((root / ".opencode/commands").glob("*.md"))
            self.assertEqual(len(commands), sum(path.name == "SKILL.md" for path in render("codex")) + 1)
            command = root / ".opencode/commands/research.md"
            self.assertIn('$ARGUMENTS', command.read_text())
            result = self.run_generator(root, "--check", "--host", "opencode")
            self.assertEqual(result.returncode, 0, result.stderr)
            changed = command.read_text() + "\nLocal command drift\n"
            command.write_text(changed)
            result = self.run_generator(root, "--check", "--host", "opencode")
            self.assertEqual(result.returncode, 1)
            self.assertIn(".opencode/commands/research.md", result.stderr)
            self.assertEqual(command.read_text(), changed)

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
                ("codex", ".agents", ""), ("claude", ".claude", "/"),
                ("opencode", ".agents", ""),
            ):
                result = render(host, root)
                self.assertEqual(len(result), {"opencode": 3, "codex": 2, "claude": 1}[host])
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
            source.write_text("---\nname: wrong\ndescription: Test role\n---\nBody\n")
            with self.assertRaisesRegex(ValueError, "名稱與資料夾不符"):
                render("codex", root)
            source.write_text("---\nname: custom\ndescription: Test role\n---\n{{UNKNOWN}}\n")
            with self.assertRaisesRegex(ValueError, "未解析"):
                render("codex", root)


if __name__ == "__main__":
    unittest.main()
