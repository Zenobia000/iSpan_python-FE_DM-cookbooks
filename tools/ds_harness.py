"""Local evidence lifecycle for host-native skills. Python standard library only."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import uuid

STAGES = ("frame", "proxy", "eda", "rules", "impact")
GOALS = ("eda", "rules", "impact")
SKILLS = ("decision-analysis",) + tuple("ds-" + stage for stage in STAGES) + ("ds-review", "ds-deliver")
CHECKS = ("scope", "leakage", "statistics", "business")


class HarnessError(Exception):
    def __init__(self, code, message, recovery):
        super().__init__(message)
        self.code, self.recovery = code, recovery


def require(condition, code, message, recovery):
    if not condition:
        raise HarnessError(code, message, recovery)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def token():
    return uuid.uuid4().hex


def initial_state(goal):
    return {"version": 1, "goal": goal, "sessions": {}, "stages": {},
            "review_starts": {}, "reviews": [], "deliveries": [], "events": []}


@contextmanager
def state_at(case, write=False):
    database = Path(case) / "state.sqlite"
    require(database.is_file(), "NO_CASE", "找不到任務狀態。", "先用 init --case <新目錄> --goal eda 建立任務。")
    mode = "rw" if write else "ro"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=" + mode, uri=True, timeout=2)
    try:
        if write:
            connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT payload FROM state WHERE id = 1").fetchone()
        require(row is not None, "BAD_STATE", "任務狀態不完整。", "保留目錄，從已核對的證據重建任務。")
        state = json.loads(row[0])
        yield state
        if write:
            connection.execute("UPDATE state SET payload = ? WHERE id = 1", (json.dumps(state, ensure_ascii=False),))
            connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_case(case, goal):
    case = Path(case)
    require(not case.exists(), "CASE_EXISTS", "任務目錄已存在。", "使用 status 續作，或選擇新的 case 目錄。")
    case.mkdir(parents=True)
    (case / "artifacts").mkdir()
    with sqlite3.connect(case / "state.sqlite") as connection:
        connection.execute("CREATE TABLE state (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("INSERT INTO state VALUES (1, ?)", (json.dumps(initial_state(goal)),))
    return {"status": "DONE", "case": str(case), "goal": goal, "next_skill": "ds-frame"}


def session_for(state, session_id, skill):
    session = state["sessions"].get(session_id)
    require(session is not None and session["skill"] == skill, "WRONG_SESSION",
            f"此操作需要 {skill} session。", f"先用 start --skill {skill} --host <claude|codex> 建立 session。")
    return session


def start_session(state, skill, host):
    session_id = token()
    state["sessions"][session_id] = {"skill": skill, "host": host, "started_at": now()}
    state["events"].append({"type": "session_start", "session": session_id, "at": now()})
    return {"status": "DONE", "session_id": session_id, "host": host, "skill": skill}


def snapshot_file(case, path, prefix):
    path = Path(path)
    content = path.read_bytes()
    require(0 < len(content) <= 2 * 1024 * 1024, "BAD_ARTIFACT", "證據文件須為 1 byte 至 2 MiB。",
            "大型資料用 record --input 引用，文件只保留方法與結果摘要。")
    content.decode("utf-8")
    require(path.suffix in (".md", ".json", ".txt"), "BAD_ARTIFACT", "證據文件須為 Markdown、JSON 或文字。",
            "將分析方法、結果與證據路徑整理成 .md 或 .json。")
    relative = Path("artifacts") / (prefix + "-" + token() + path.suffix)
    with (Path(case) / relative).open("xb") as stream:
        stream.write(content)
    return {"path": relative.as_posix(), "sha256": hashlib.sha256(content).hexdigest()}


def upstream(stage):
    return STAGES[:STAGES.index(stage)]


def current_dependencies(state, stage):
    return {key: state["stages"].get(key, {}).get("id") for key in upstream(stage)}


def file_valid(path, expected):
    try:
        return digest(path) == expected
    except OSError:
        return False


def stage_states(case, state):
    results = {}
    for stage in STAGES:
        artifact = state["stages"].get(stage)
        if artifact is None:
            results[stage] = {"state": "missing"}
            continue
        problems = []
        if not file_valid(Path(case) / artifact["path"], artifact["sha256"]):
            problems.append("artifact_changed_or_missing")
        for item in artifact["inputs"]:
            if not file_valid(item["path"], item["sha256"]):
                problems.append("input_changed_or_missing:" + item["path"])
        for dependency, old_id in artifact["dependencies"].items():
            if old_id != state["stages"].get(dependency, {}).get("id"):
                problems.append("upstream_revised:" + dependency)
            elif old_id is not None and results[dependency]["state"] != "current":
                problems.append("upstream_stale:" + dependency)
        results[stage] = {"state": "stale" if problems else "current", "id": artifact["id"],
                          "path": artifact["path"], "reasons": problems}
    return results


def record(case, state, stage, path, session_id, inputs):
    session_for(state, session_id, "ds-" + stage)
    statuses = stage_states(case, state)
    needed = [] if stage == "frame" else ["frame"]
    if stage in ("rules", "impact"):
        needed.append("eda")
    # Optional stages may be omitted, but an existing stale upstream must be repaired.
    needed += [key for key in upstream(stage) if key in state["stages"]]
    require(all(statuses[key]["state"] == "current" for key in needed), "UPSTREAM_NOT_READY",
            "上游證據缺失或已過期。", "先用 status 查看原因，從最早的缺失／過期環節重做並 record。")
    references = [{"path": str(Path(path).resolve()), "sha256": digest(path)} for path in inputs]
    artifact = {**snapshot_file(case, path, stage), "id": token(), "stage": stage,
                "session": session_id, "inputs": references,
                "dependencies": current_dependencies(state, stage), "recorded_at": now()}
    state["stages"][stage] = artifact
    state["events"].append({"type": "record", **artifact})
    return {"status": "DONE", "artifact": artifact}


def candidate(case, state):
    statuses = stage_states(case, state)
    required = ["frame", state["goal"]]
    require(all(statuses[key]["state"] == "current" for key in required)
            and all(value["state"] != "stale" for value in statuses.values()),
            "CANDIDATE_NOT_READY", "任務目標尚未完成，或有過期證據。",
            "執行 status，補齊目標並重做過期環節；不需要的環節可保持 missing。")
    return {key: {"id": value["id"], "sha256": value["sha256"]}
            for key, value in state["stages"].items()}


def review_start(case, state, session_id, report_path):
    session_for(state, session_id, "ds-review")
    receipt_id = token()
    captured = candidate(case, state)
    report = snapshot_file(case, report_path, "report-candidate")
    receipt = {"id": receipt_id, "session": session_id, "candidate": captured, "report": report, "at": now()}
    state["review_starts"][receipt_id] = receipt
    state["events"].append({"type": "review_start", **receipt})
    return {"status": "DONE", "review_token": receipt_id, "candidate": receipt["candidate"], "report": report,
            "review_origin": "self_reported", "note": "session 與 host 是宣告資料，不證明獨立審查。"}


def review_finish(case, state, receipt_id, verdict, path):
    receipt = state["review_starts"].get(receipt_id)
    require(receipt is not None and not receipt.get("finished"), "BAD_REVIEW_TOKEN", "review token 不存在或已使用。",
            "先用 review-start 擷取目前候選，再檢查證據。")
    require(candidate(case, state) == receipt["candidate"], "STALE_REVIEW", "審查期間候選證據已改變。",
            "重新 review-start 並檢查新候選，不重用舊的 pass。")
    require(file_valid(Path(case) / receipt["report"]["path"], receipt["report"]["sha256"]),
            "STALE_REVIEW", "被審查的報告快照已變更。", "重新 review-start 並檢查報告。")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(report, dict), "BAD_REVIEW", "review 須為 JSON object。",
            "依 docs/harness.md 的 review 格式完成文件。")
    checks = report.get("checks", {})
    require(isinstance(report.get("summary"), str) and bool(report["summary"].strip())
            and isinstance(report.get("findings"), list) and isinstance(checks, dict),
            "BAD_REVIEW", "review 需要 summary、checks 與 findings。", "依 docs/harness.md 的 review 格式完成文件。")
    for key in CHECKS:
        check = checks.get(key, {})
        require(isinstance(check, dict) and check.get("result") in ("pass", "fail", "not_applicable")
                and isinstance(check.get("evidence"), str) and bool(check["evidence"].strip()),
                "BAD_REVIEW", f"缺少有效的 {key} 檢查與證據說明。", "補齊 review checks；不適用也需說明理由。")
    for finding in report["findings"]:
        require(isinstance(finding, dict) and isinstance(finding.get("detail"), str) and bool(finding["detail"].strip())
                and finding.get("severity") in ("blocker", "major", "minor")
                and finding.get("resolution") in ("open", "resolved"),
                "BAD_REVIEW", "finding 需要 detail、severity 與 resolution。", "填入具體問題與處理狀態。")
    if verdict == "pass":
        require(all(checks[key]["result"] != "fail" for key in CHECKS)
                and not any(item["severity"] in ("blocker", "major") and item["resolution"] != "resolved"
                            for item in report["findings"]),
                "OPEN_FINDINGS", "仍有失敗檢查或未解決的重要問題。", "記錄 revise／blocked，修正證據後重新審查。")
    review = {**snapshot_file(case, path, "review"), "id": token(), "token": receipt_id,
              "session": receipt["session"], "candidate": receipt["candidate"], "report": receipt["report"], "verdict": verdict,
              "review_origin": "self_reported", "at": now()}
    receipt["finished"] = True
    state["reviews"].append(review)
    state["events"].append({"type": "review_finish", **review})
    return {"status": "DONE" if verdict == "pass" else "DONE_WITH_CONCERNS", "review": review}


def valid_review(case, state):
    if not state["reviews"]:
        return None
    review = state["reviews"][-1]
    try:
        matches = review["candidate"] == candidate(case, state)
    except HarnessError:
        return None
    return review if matches and review["verdict"] == "pass" and file_valid(
        Path(case) / review["path"], review["sha256"]) and file_valid(
        Path(case) / review["report"]["path"], review["report"]["sha256"]) else None


def deliver(case, state, path, session_id):
    session_for(state, session_id, "ds-deliver")
    review = valid_review(case, state)
    require(review is not None, "NO_CURRENT_REVIEW", "目前候選沒有有效的 pass review。",
            "先完成 ds-review；修改證據後必須重新審查。")
    require(digest(path) == review["report"]["sha256"], "REPORT_NOT_REVIEWED", "交付報告與已審查版本不同。",
            "交付 review token 指向的原版報告，或對修改後報告重新 review-start。")
    delivery = {**snapshot_file(case, path, "delivery"), "id": token(), "session": session_id,
                "candidate": review["candidate"], "review_id": review["id"], "at": now()}
    state["deliveries"].append(delivery)
    state["events"].append({"type": "deliver", **delivery})
    return {"status": "DONE", "delivery": delivery,
            "note": "本機交付快照已建立；此 receipt 不代表已部署或已驗證商業成效。"}


def status(case, state):
    stages = stage_states(case, state)
    next_stage = next((key for key in STAGES if stages[key]["state"] == "stale"), None)
    if next_stage is None:
        next_stage = "frame" if stages["frame"]["state"] == "missing" else (
            "eda" if stages["eda"]["state"] == "missing" else (
                state["goal"] if stages[state["goal"]]["state"] == "missing" else None))
    review = valid_review(case, state)
    latest = state["deliveries"][-1] if state["deliveries"] else None
    delivery_current = bool(latest and review and latest["review_id"] == review["id"]
                            and file_valid(Path(case) / latest["path"], latest["sha256"]))
    return {"status": "DONE", "goal": state["goal"], "stages": stages,
            "review_current": review is not None, "delivery_current": delivery_current,
            "next_skill": "ds-" + next_stage if next_stage else "ds-deliver",
            "sessions": state["sessions"], "events": len(state["events"])}


def build_parser():
    parser = argparse.ArgumentParser(description="角色技能共用的本機證據與 review harness；不選擇統計方法。")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "status", "start", "record", "review-start", "review-finish", "deliver"):
        item = sub.add_parser(command)
        item.add_argument("--case", required=True, type=Path)
        if command == "init":
            item.add_argument("--goal", choices=GOALS, default="eda")
        elif command == "start":
            item.add_argument("--skill", choices=SKILLS, required=True)
            item.add_argument("--host", choices=("claude", "codex"), required=True)
        elif command in ("record", "review-start", "deliver"):
            item.add_argument("--session", required=True)
        if command in ("record", "review-start", "review-finish", "deliver"):
            item.add_argument("--file", required=True, type=Path)
        if command == "record":
            item.add_argument("--stage", choices=STAGES, required=True)
            item.add_argument("--input", action="append", default=[], type=Path)
        if command == "review-finish":
            item.add_argument("--token", required=True)
            item.add_argument("--verdict", choices=("pass", "revise", "blocked"), required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            result = init_case(args.case, args.goal)
        else:
            with state_at(args.case, write=args.command != "status") as state:
                if args.command == "status":
                    result = status(args.case, state)
                elif args.command == "start":
                    result = start_session(state, args.skill, args.host)
                elif args.command == "record":
                    result = record(args.case, state, args.stage, args.file, args.session, args.input)
                elif args.command == "review-start":
                    result = review_start(args.case, state, args.session, args.file)
                elif args.command == "review-finish":
                    result = review_finish(args.case, state, args.token, args.verdict, args.file)
                else:
                    result = deliver(args.case, state, args.file, args.session)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except HarnessError as error:
        result = {"status": "BLOCKED", "code": error.code, "reason": str(error), "recovery": error.recovery}
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        result = {"status": "BLOCKED", "code": "IO_OR_FORMAT", "reason": str(error),
                  "recovery": "檢查本機路徑、文件格式或另一個執行中的 writer，再用 status 檢查任務。"}
    print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
