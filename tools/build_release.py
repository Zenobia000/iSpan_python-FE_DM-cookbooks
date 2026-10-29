"""Build a lightweight, installable Agent Skills repository in an empty directory."""

import argparse
from pathlib import Path
import shutil
import sys

if __package__:
    from .render_skills import ROOT, render
else:
    from render_skills import ROOT, render


def build(destination):
    destination = Path(destination).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f'請指定不存在或空白的發行目錄：{destination}')
    outputs = {
        Path('skills') / path.relative_to('.agents/skills'): text
        for path, text in render('codex').items()
    }
    for target, source in (
        ('README.md', 'README.md'), ('README.en.md', 'README.en.md'),
        ('LICENSE', 'LICENSE'), ('VERSION', 'VERSION')
    ):
        outputs[Path(target)] = (ROOT / source).read_text(encoding='utf-8')
    artwork = [ROOT / 'assets/brand' / name for name in ('icon.png', 'hero.png', 'workflow.svg')]
    for source in artwork:
        if not source.is_file():
            raise ValueError(f'缺少品牌資產：{source}')
    for relative, text in outputs.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    for source in artwork:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return len(outputs) + len(artwork)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    try:
        count = build(args.out)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f'Built {count} release files in {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
