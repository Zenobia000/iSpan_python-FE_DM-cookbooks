"""Create one analysis workspace without overwriting existing project files."""

import argparse
import json
from pathlib import Path
import sys

DIRECTORIES = (
    'research', 'materials/external', 'materials/generated',
    'data/raw', 'data/processed', 'data/features',
    'analysis/notebooks', 'analysis/scripts', 'config', 'reports', 'artifacts',
)
BEGIN = '# BEGIN ds-analysis-skills outputs'
END = '# END ds-analysis-skills outputs'
IGNORE_BLOCK = f'''{BEGIN}
/data/raw/*
/data/interim/*
/data/processed/*
/data/features/*
/materials/external/*
/materials/generated/*
/artifacts/*
/reports/*
!/reports/README.md
{END}
'''


def initialize(root, *, opencode_commands=False, skill_root=None):
    root = Path(root).expanduser().resolve()
    skill_root = Path(skill_root) if skill_root else Path(__file__).resolve().parents[1]
    assets = skill_root / 'assets/project'
    if not assets.is_dir():
        raise ValueError(f'找不到專案骨架：{assets}')
    files = {p.relative_to(assets): p.read_bytes() for p in sorted(assets.rglob('*')) if p.is_file()}
    if not files:
        raise ValueError('專案骨架沒有文件。')
    directories = set(Path(p) for p in DIRECTORIES)
    if opencode_commands:
        commands = skill_root / 'assets/commands'
        sources = sorted(commands.glob('*.md'))
        if not sources:
            raise ValueError(f'找不到指令轉接檔：{commands}')
        files.update({Path('.opencode/commands') / p.name: p.read_bytes() for p in sources})
        directories.add(Path('.opencode/commands'))
    for relative in files:
        directories.update(p for p in relative.parents if p != Path('.'))
    if root.exists() and not root.is_dir():
        raise ValueError(f'專案位置不是目錄：{root}')
    for relative in sorted(directories):
        target = root / relative
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise ValueError(f'目錄衝突或符號連結：{target}')
    for relative in sorted([*files, Path('.gitignore')]):
        target = root / relative
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f'文件衝突或符號連結：{target}')
    ignore_path = root / '.gitignore'
    ignore_existed = ignore_path.exists()
    previous = ignore_path.read_text(encoding='utf-8') if ignore_existed else ''
    if (BEGIN in previous) != (END in previous):
        raise ValueError(f'忽略規則區塊不完整：{ignore_path}')
    # All deterministic path conflicts are checked before the first mutation.
    root.mkdir(parents=True, exist_ok=True)
    result = {'root': str(root), 'created': [], 'updated': [], 'skipped': []}
    for relative in sorted(directories, key=lambda p: (len(p.parts), str(p))):
        target = root / relative
        if not target.exists():
            target.mkdir()
            result['created'].append(str(relative) + '/')
    for relative, content in sorted(files.items()):
        target = root / relative
        if target.exists():
            result['skipped'].append(str(relative))
            continue
        with target.open('xb') as stream:
            stream.write(content)
        result['created'].append(str(relative))
    if BEGIN not in previous:
        with ignore_path.open('a', encoding='utf-8') as stream:
            if previous and not previous.endswith('\n'):
                stream.write('\n')
            stream.write(IGNORE_BLOCK)
        result['updated' if ignore_existed else 'created'].append('.gitignore')
    else:
        result['skipped'].append('.gitignore')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--opencode-commands', action='store_true')
    args = parser.parse_args()
    try:
        result = initialize(args.root, opencode_commands=args.opencode_commands)
    except (OSError, ValueError) as error:
        print(f'初始化未完成：{error}', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
