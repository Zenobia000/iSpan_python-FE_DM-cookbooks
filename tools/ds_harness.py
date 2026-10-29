"""Local evidence lifecycle for host-native skills. Python standard library only."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

STAGES = ("frame", "plan", "proxy", "eda", "rules", "impact")
GOALS = ("eda", "rules", "impact")
SKILLS = ("decision-analysis",) + tuple("ds-" + stage for stage in STAGES if stage != "plan") + ("ds-review", "ds-deliver")
CHECKS = ("scope", "leakage", "statistics", "business")
PROJECT_TEMPLATE = Path(__file__).resolve().parents[1] / "skills/ds-frame/assets/project"
PROJECT_DIRS = ("research/confirmations", "data/raw", "data/interim", "data/processed",
                "analysis", "config", "reports/explorations", "reports/figures", "reports/tables", "artifacts")


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
    return {"version": 2, "goal": goal, "sessions": {}, "stages": {}, "confirmations": [],
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
    require(not any((p / "state.sqlite").is_file() for p in case.resolve().parents),
            "NESTED_CASE", "不能在另一個分析專案內建立 case。", "使用 projects/ 下另一個同層目錄。")
    case.mkdir(parents=True)
    shutil.copytree(PROJECT_TEMPLATE, case, dirs_exist_ok=True)
    for relative in PROJECT_DIRS:
        (case / relative).mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(case / "state.sqlite") as connection:
        connection.execute("CREATE TABLE state (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        connection.execute("INSERT INTO state VALUES (1, ?)", (json.dumps(initial_state(goal)),))
    return {"status": "DONE", "case": str(case), "goal": goal, "next_skill": "ds-frame",
            "next_action": "define_problem", "directories": list(PROJECT_DIRS)}


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
    require(Path(case).resolve() in path.resolve().parents, "OUTSIDE_PROJECT",
            "證據文件必須位於目前分析專案內。", "先將草稿、報告或確認紀錄保存到本 case，再登錄。")
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
        dependencies = dict.fromkeys((*upstream(stage), *artifact["dependencies"]))
        for dependency in dependencies:
            old_id = artifact["dependencies"].get(dependency)
            if old_id != state["stages"].get(dependency, {}).get("id"):
                problems.append("upstream_revised:" + dependency)
            elif old_id is not None and results[dependency]["state"] != "current":
                problems.append("upstream_stale:" + dependency)
        results[stage] = {"state": "stale" if problems else "current", "id": artifact["id"],
                          "path": artifact["path"], "reasons": problems}
    return results


def frame_confirmation(case, state):
    frame = state["stages"].get("frame")
    if frame is None or stage_states(case, state)["frame"]["state"] != "current":
        return None
    for receipt in reversed(state.get("confirmations", [])):
        if (receipt["revision"] == frame["id"] and receipt["frame_sha256"] == frame["sha256"]
                and file_valid(Path(case) / receipt["path"], receipt["sha256"])):
            return receipt
    return None


def confirm_frame(case, state, revision, path, session_id):
    session_for(state, session_id, "ds-frame")
    frame = state["stages"].get("frame")
    require(frame is not None and frame["id"] == revision
            and stage_states(case, state)["frame"]["state"] == "current",
            "STALE_CONFIRMATION", "確認的問題版本不存在或已失效。", "重新呈現目前問題版本，取得使用者明確確認。")
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(evidence, dict) and evidence.get("decision") == "confirmed"
            and evidence.get("source") in ("user_message", "user_file")
            and all(isinstance(evidence.get(key), str) and bool(evidence[key].strip())
                    for key in ("statement", "reference")),
            "BAD_CONFIRMATION", "確認紀錄需有明確使用者陳述與來源。",
            "依 docs/research-workflow.md 保存 decision、source、statement、reference；不可代替使用者確認。")
    receipt = {**snapshot_file(case, path, "confirmation"), "id": token(),
               "revision": revision, "frame_sha256": frame["sha256"], "session": session_id,
               "confirmation_origin": "recorded_user_statement", "at": now()}
    state.setdefault("confirmations", []).append(receipt)
    state["events"].append({"type": "confirm_frame", **receipt})
    return {"status": "DONE", "confirmation": receipt, "next_skill": "ds-frame",
            "next_action": "write_research_plan",
            "note": "僅驗證來源紀錄與版本綁定，不認證使用者身分或陳述真偽。"}


def input_reference(case, path):
    path = Path(path).resolve()
    owner = next((p for p in path.parents if (p / "state.sqlite").is_file()), None)
    require(owner is None or owner == Path(case).resolve(), "CROSS_PROJECT_INPUT",
            "不能直接引用另一個分析專案的工作檔或資料。",
            "將經允許的來源快照複製到本專案 data/，記錄來源與版本，再登錄。")
    return {"path": str(path), "sha256": digest(path)}


def record(case, state, stage, path, session_id, inputs):
    session_for(state, session_id, "ds-frame" if stage == "plan" else "ds-" + stage)
    statuses = stage_states(case, state)
    needed = [] if stage == "frame" else ["frame"]
    if stage in ("rules", "impact"):
        needed.append("eda")
    # Optional stages may be omitted, but an existing stale upstream must be repaired.
    needed += [key for key in upstream(stage) if key in state["stages"]]
    require(all(statuses[key]["state"] == "current" for key in needed), "UPSTREAM_NOT_READY",
            "上游證據缺失或已過期。", "先用 status 查看原因，從最早的缺失／過期環節重做並 record。")
    if stage != "frame":
        require(frame_confirmation(case, state) is not None, "FRAME_NOT_CONFIRMED",
                "研究問題尚未取得目前版本的使用者確認。", "先呈現問題定義並取得明確回覆，再使用 confirm 登錄。")
    if stage not in ("frame", "plan"):
        require(statuses["plan"]["state"] == "current", "PLAN_NOT_READY",
                "分析前需要有效研究計畫。", "由 ds-frame 根據已確認問題完成假設、反證、方法與順序，以 --stage plan 登錄。")
    references = [input_reference(case, path) for path in inputs]
    artifact = {**snapshot_file(case, path, stage), "id": token(), "stage": stage,
                "session": session_id, "inputs": references,
                "dependencies": current_dependencies(state, stage), "recorded_at": now()}
    state["stages"][stage] = artifact
    state["events"].append({"type": "record", **artifact})
    return {"status": "DONE", "artifact": artifact}


def candidate(case, state):
    statuses = stage_states(case, state)
    required = ["frame", "plan", state["goal"]]
    require(all(statuses[key]["state"] == "current" for key in required)
            and frame_confirmation(case, state) is not None
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
    confirmation = frame_confirmation(case, state)
    if stages["frame"]["state"] != "current":
        next_stage, next_action = "frame", "define_problem"
    elif confirmation is None:
        next_stage, next_action = "frame", "await_problem_confirmation"
    elif stages["plan"]["state"] != "current":
        next_stage, next_action = "frame", "write_research_plan"
    else:
        next_stage = next((key for key in STAGES if stages[key]["state"] == "stale"), None)
        if next_stage is None:
            next_stage = "eda" if stages["eda"]["state"] == "missing" else (
                state["goal"] if stages[state["goal"]]["state"] == "missing" else None)
        next_action = "analyze" if next_stage else "review_and_deliver"
    review = valid_review(case, state)
    latest = state["deliveries"][-1] if state["deliveries"] else None
    delivery_current = bool(latest and review and latest["review_id"] == review["id"]
                            and file_valid(Path(case) / latest["path"], latest["sha256"]))
    return {"status": "DONE", "goal": state["goal"], "stages": stages,
            "review_current": review is not None, "delivery_current": delivery_current,
            "problem_confirmed": confirmation is not None, "next_action": next_action,
            "next_skill": "ds-" + next_stage if next_stage else "ds-deliver",
            "sessions": state["sessions"], "events": len(state["events"])}


def build_parser():
    parser = argparse.ArgumentParser(description="角色技能共用的本機證據與 review harness；不選擇統計方法。")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "status", "start", "record", "confirm", "review-start", "review-finish", "deliver"):
        item = sub.add_parser(command)
        item.add_argument("--case", required=True, type=Path)
        if command == "init":
            item.add_argument("--goal", choices=GOALS, default="eda")
        elif command == "start":
            item.add_argument("--skill", choices=SKILLS, required=True)
            item.add_argument("--host", choices=("claude", "codex"), required=True)
        elif command in ("record", "confirm", "review-start", "deliver"):
            item.add_argument("--session", required=True)
        if command in ("record", "confirm", "review-start", "review-finish", "deliver"):
            item.add_argument("--file", required=True, type=Path)
        if command == "record":
            item.add_argument("--stage", choices=STAGES, required=True)
            item.add_argument("--input", action="append", default=[], type=Path)
        if command == "confirm":
            item.add_argument("--revision", required=True)
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
                elif args.command == "confirm":
                    result = confirm_frame(args.case, state, args.revision, args.file, args.session)
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
