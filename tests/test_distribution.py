"""Exercise the shipped skill in a separate install and workspace location."""

from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

from tools.build_release import build


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.release = Path(cls.temporary.name) / 'release'
        build(cls.release)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def call_init(self, root, *args, installed=None):
        skill = installed or self.release / 'skills/ds-init'
        return subprocess.run(
            [sys.executable, str(skill / 'scripts/init_project.py'), '--root', str(root), *args],
            cwd=root.parent, text=True, capture_output=True, check=False,
        )

    def snapshot(self, root):
        return {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}

    def test_release_is_complete_and_local_resource_links_resolve(self):
        self.assertEqual(len(list((self.release / 'skills').glob('*/SKILL.md'))), 9)
        self.assertEqual({p.name for p in self.release.iterdir()}, {'skills', 'README.md', 'VERSION', 'LICENSE'})
        for path in self.release.rglob('*.md'):
            text = path.read_text(encoding='utf-8')
            for target in re.findall(r'(?<!!)\[[^\]]*\]\(([^\s)]+)\)', text):
                if '://' in target or target.startswith('#'):
                    continue
                self.assertTrue((path.parent / target.split('#')[0]).exists(), (path, target))
        for entry in (self.release / 'skills').glob('*/SKILL.md'):
            for resource in re.findall(r'`(resources/[^`]+\.md)`', entry.read_text(encoding='utf-8')):
                self.assertTrue((entry.parent / resource).is_file(), (entry, resource))

    def test_relocated_skill_initializes_unicode_workspace_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            installed = base / 'installed skills/ds-init'
            shutil.copytree(self.release / 'skills/ds-init', installed)
            project = base / '分析題目 A'
            result = self.call_init(project, installed=installed)
            self.assertEqual(result.returncode, 0, result.stderr)
            for relative in ('research/question.md', 'research/plan.md', 'research/handoff.md',
                             'AGENTS.md', 'CLAUDE.md', '.gitignore'):
                self.assertTrue((project / relative).is_file(), relative)
            for relative in ('data/raw', 'data/interim', 'data/processed', 'analysis', 'config',
                             'reports/explorations', 'reports/figures', 'reports/tables'):
                self.assertTrue((project / relative).is_dir(), relative)
            (project / 'research/question.md').write_text('Human confirmed scope\n')
            (project / 'data/raw/private.csv').write_text('Keep original data\n')
            before = self.snapshot(project)
            result = self.call_init(project, installed=installed)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(before, self.snapshot(project))
            self.assertEqual(len(json.loads(result.stdout)['created']), 0)

    def test_existing_guidance_ignore_rules_and_commands_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'workspace'
            root.mkdir()
            (root / 'AGENTS.md').write_text('Existing policy\n')
            (root / '.gitignore').write_text('personal/')
            commands = root / '.opencode/commands'
            commands.mkdir(parents=True)
            (commands / 'ds-frame.md').write_text('Existing custom command\n')
            result = self.call_init(root, '--opencode-commands')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / 'AGENTS.md').read_text(), 'Existing policy\n')
            self.assertTrue((root / '.gitignore').read_text().startswith('personal/\n'))
            self.assertEqual((commands / 'ds-frame.md').read_text(), 'Existing custom command\n')
            self.assertEqual(len(list(commands.glob('*.md'))), 9)
            self.assertIn('$ARGUMENTS', (commands / 'ds-eda.md').read_text())
            before = self.snapshot(root)
            result = self.call_init(root, '--opencode-commands')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(before, self.snapshot(root))

    def test_path_conflict_is_detected_before_creating_any_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'workspace'
            root.mkdir()
            (root / 'analysis').write_text('Existing file\n')
            before = self.snapshot(root)
            result = self.call_init(root)
            self.assertEqual(result.returncode, 1)
            self.assertIn('analysis', result.stderr)
            self.assertEqual(before, self.snapshot(root))
            self.assertFalse((root / 'research').exists())

    def test_symlink_to_external_data_is_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / 'outside'
            outside.mkdir()
            (outside / 'source.txt').write_text('Private data\n')
            root = base / 'workspace'
            (root / 'data').mkdir(parents=True)
            (root / 'data/raw').symlink_to(outside, target_is_directory=True)
            result = self.call_init(root)
            self.assertEqual(result.returncode, 1)
            self.assertIn('data/raw', result.stderr)
            self.assertEqual(list(outside.iterdir()), [outside / 'source.txt'])
            self.assertFalse((root / 'research').exists())

    def test_incomplete_ignore_block_is_reported_without_partial_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'workspace'
            root.mkdir()
            (root / '.gitignore').write_text('# BEGIN ds-analysis-skills outputs\n')
            before = self.snapshot(root)
            result = self.call_init(root)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(before, self.snapshot(root))
            self.assertFalse((root / 'research').exists())

    def test_release_build_refuses_nonempty_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'keep.txt').write_text('Keep\n')
            with self.assertRaises(ValueError):
                build(root)
            self.assertEqual(self.snapshot(root), {Path('keep.txt'): b'Keep\n'})


if __name__ == '__main__':
    unittest.main()
