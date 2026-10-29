"""Build a lightweight, installable Agent Skills repository in an empty directory."""

import argparse
from pathlib import Path
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
        ('README.md', 'docs/distribution-readme.md'), ('LICENSE', 'LICENSE'), ('VERSION', 'VERSION')
    ):
        outputs[Path(target)] = (ROOT / source).read_text(encoding='utf-8')
    for relative, text in outputs.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    return len(outputs)


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
