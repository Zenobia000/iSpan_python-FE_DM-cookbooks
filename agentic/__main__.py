"""Run from the repository root with python -m agentic."""

import argparse
import json
from pathlib import Path

from .pipeline import make_demo, run


def main():
    parser = argparse.ArgumentParser(description="AI 協作決策分析工具")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="使用明確標記的合成資料執行四環節")
    demo.add_argument("--out", type=Path, required=True)
    analyze = sub.add_parser("analyze", help="根據 JSON 設定分析單一設備的時間資料")
    analyze.add_argument("--data", type=Path, required=True)
    analyze.add_argument("--config", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        # A run owns a new directory. Never silently replace earlier evidence.
        args.out.mkdir(parents=True, exist_ok=False)
        if args.command == "demo":
            data, config = make_demo(args.out)
        else:
            data, config = args.data, args.config
        result = run(data, config, args.out)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"分析未完成：{error}\n")
    print(f"完成：{result / 'report.md'}")


if __name__ == "__main__":
    main()
