"""Stage 5 (eval): measure whether a generated skill actually helps.

For a skill built by stage 4, sample real (redacted) prompts from its
cluster's arcs and run each twice through headless Claude Code
(`claude -p ... --output-format json --permission-mode acceptEdits`):

  A. with the skill installed in ~/.claude/skills/<name>
  B. with the skill dir temporarily moved out of the skills dir

Each run happens in a throwaway SEEDED workdir (a small toy project with an
in-progress task) so resume/document-style prompts have something real to
act on. Per run we record:
  - num_turns / tokens / cost from the CLI's JSON output
  - skill_loaded: whether the skill's name appears in the run's debug log
    (--debug-file), i.e. the skill was actually visible/consulted
  - success: an optional mechanical criterion (--check claude-md: does
    workdir/CLAUDE.md contain plausible resume state AFTER the run — the
    seed CLAUDE.md is deliberately too thin to pass on its own)
  - corrections_recurred: LLM-judged against mined corrections (skipped
    when the cluster produced none)

The skill dir move is guarded by try/finally so an interrupt can't leave
the skill uninstalled.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import Config
from .parser import load_sessions
from .redaction import redact
from . import llm

_DEBUG_LOG = ".sm-debug.log"

# Deliberately thin seed: no resume keywords, so the claude-md success
# criterion can only be satisfied by what the run itself writes.
_FIXTURE = {
    "README.md": "# orders-etl\n\nToy CSV-to-parquet pipeline. Work in progress.\n",
    "CLAUDE.md": "# orders-etl\n\nToy data pipeline project.\n",
    "app.py": (
        "\"\"\"orders-etl: load orders.csv, clean, write parquet.\"\"\"\n\n"
        "import csv\n\n\n"
        "def load_orders(path):\n"
        "    with open(path) as f:\n"
        "        return list(csv.DictReader(f))\n\n\n"
        "def clean(rows):\n"
        "    # TODO: drop rows with missing customer_id\n"
        "    # TODO: parse order_date to ISO\n"
        "    raise NotImplementedError\n\n\n"
        "def write_parquet(rows, path):\n"
        "    raise NotImplementedError\n"
    ),
    "orders.csv": "order_id,customer_id,order_date,total\n1,c1,2026-01-05,19.99\n2,,01/07/2026,5.00\n",
}

_RESUME_KEYWORDS = ("continue", "next", "state", "resume", "step", "status",
                    "todo", "progress", "remaining")


def _sample_prompts(cfg: Config, meta: dict, n: int) -> list[dict]:
    """Prefer arc opening prompts (the sub-workflow the skill encodes);
    fall back to session openings for pre-arc provenance files."""
    from .scoring import _content_words
    by_id = {s["session_id"]: s for s in load_sessions(cfg)}
    arcs = meta.get("arcs", [])
    sigs = [_content_words(a.get("signature", "")) for a in arcs]

    def centrality(k: int) -> float:
        """Avg signature similarity to the OTHER arcs: outlier arcs the
        clusterer lumped in (real failure: a scaffolding arc inside a
        session-resume cluster) rank last and don't get sampled."""
        others = [s for j, s in enumerate(sigs) if j != k and (s or sigs[k])]
        if not others or not sigs[k]:
            return 0.0
        return sum(len(sigs[k] & o) / len(sigs[k] | o) for o in others) / len(others)

    candidates = []
    for k, a in enumerate(sorted(range(len(arcs)), key=lambda k: -centrality(k))):
        arc = arcs[a]
        s = by_id.get(arc["session_id"])
        if s:
            prompts = s["user_prompts"][arc["start"]: arc["end"] + 1]
            if prompts:
                candidates.append((arc["session_id"], prompts))
    if not candidates:
        candidates = [(sid, by_id[sid]["user_prompts"])
                      for sid in meta.get("session_ids", []) if sid in by_id]
    samples, seen_sessions = [], set()
    for prefer_new in (True, False):
        for sid, prompts in candidates:
            if len(samples) >= n:
                break
            if prefer_new == (sid in seen_sessions):
                continue
            first = next((p["text"] for p in prompts if not p["is_correction"]),
                         prompts[0]["text"])
            # Junk filter by content, not length: a raw length cutoff threw
            # away terse imperatives ("Commit and push") that are exactly
            # the trigger prompts of commit/close-style skills.
            if (len(_content_words(first)) >= 2
                    and all(first != s["prompt"] for s in samples)):
                samples.append({"session_id": sid, "prompt": first})
                seen_sessions.add(sid)
    return samples[:n]


def _git(workdir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=workdir, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=60, shell=True)


def _seed_workdir(workdir: Path, check: str | None = None) -> None:
    for name, content in _FIXTURE.items():
        (workdir / name).write_text(content, encoding="utf-8", newline="\n")
    if check == "git-clean":
        # A repo mid-work: one clean commit behind, then uncommitted edits
        # and an untracked file. Deliberately NO remote — a good commit
        # skill must handle "nowhere to push" without flailing. Identity is
        # repo-local so the machine's global git config never leaks in.
        _git(workdir, "init", "-b", "main")
        _git(workdir, "config", "user.name", "Eval Fixture")
        _git(workdir, "config", "user.email", "fixture@example.com")
        _git(workdir, "add", "-A")
        _git(workdir, "commit", "-q", "-m", "initial project state")
        with open(workdir / "app.py", "a", encoding="utf-8") as f:
            f.write("\n\ndef parse_date(raw):\n    # WIP: handle both ISO and US formats\n    raise NotImplementedError\n")
        (workdir / "scratch_notes.md").write_text(
            "- parse_date started, not finished\n- ask about timezone handling\n",
            encoding="utf-8")


def _check_claude_md(workdir: Path) -> dict:
    """Success = CLAUDE.md exists and now carries plausible resume state."""
    path = workdir / "CLAUDE.md"
    if not path.is_file():
        return {"success": False, "reason": "no CLAUDE.md"}
    text = path.read_text(encoding="utf-8", errors="replace")
    grew = len(text) >= len(_FIXTURE["CLAUDE.md"]) + 150
    kw = [k for k in _RESUME_KEYWORDS if re.search(rf"\b{k}", text, re.I)]
    ok = grew and len(kw) >= 2
    return {"success": ok,
            "reason": f"len={len(text)} keywords={kw[:4]}" + ("" if grew else " (did not grow)"),
            "excerpt": redact(text)[:400]}

def _check_git_clean(workdir: Path) -> dict:
    """Success = at least one NEW commit beyond the seed, and a clean
    working tree (nothing uncommitted left behind)."""
    count = _git(workdir, "rev-list", "--count", "HEAD").stdout.strip()
    porcelain = _git(workdir, "status", "--porcelain").stdout.strip()
    try:
        commits = int(count)
    except ValueError:
        return {"success": False, "reason": f"not a git repo? ({count[:60]})"}
    ok = commits > 1 and not porcelain
    log = _git(workdir, "log", "--oneline", "-3").stdout.strip()
    return {"success": ok,
            "reason": f"commits={commits} dirty_files={len(porcelain.splitlines()) if porcelain else 0}",
            "excerpt": redact(log)[:400]}


_CHECKS = {"claude-md": _check_claude_md, "git-clean": _check_git_clean}


def _run_headless(cfg: Config, prompt: str, workdir: Path, skill_name: str,
                  check: str | None) -> dict:
    # The prompt goes through STDIN, never through the command line: on
    # Windows the claude shim needs shell=True, and cmd.exe truncates
    # multi-line argv elements at the first newline (real failure we hit).
    # Read/search/exec tools must be allowlisted explicitly: headless runs
    # can't answer permission prompts, and acceptEdits alone left skill-led
    # runs flailing against Read/Glob/PowerShell denials until the turn cap
    # (real failure we hit). Exec scope is deliberately narrow (git/python/
    # echo prefixes only): historical prompts can mention things like
    # `claude update`, and a blanket Bash allowance would let the eval run
    # execute them for real.
    allowed = ("Read Glob Grep Edit Write "
               "Bash(git *) Bash(python *) Bash(echo *) Bash(ls *) Bash(cat *)")
    cmd = [
        "claude", "-p",
        "--output-format", "json",
        "--max-turns", str(cfg.eval_max_turns),
        "--permission-mode", "acceptEdits",
        "--allowedTools", allowed,
        "--debug-file", str(workdir / _DEBUG_LOG),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=workdir, input=prompt, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=cfg.eval_timeout_s,
            shell=True)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {cfg.eval_timeout_s}s"}
    if proc.returncode != 0 and not proc.stdout.strip():
        return {"error": f"exit {proc.returncode}: {proc.stderr[-500:]}"}
    stdout = proc.stdout.strip()
    data = None
    for candidate in (stdout, stdout.splitlines()[-1] if stdout else ""):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        return {"error": f"unparseable output: {stdout[-300:]}"}

    skill_loaded = False
    tools_dispatched: dict[str, int] = {}
    denials: list[str] = []
    log = workdir / _DEBUG_LOG
    if log.is_file():
        log_text = log.read_text(encoding="utf-8", errors="replace")
        skill_loaded = skill_name.lower() in log_text.lower()
        for m in re.finditer(r"tool_dispatch_start tool=(\w+)", log_text):
            tools_dispatched[m.group(1)] = tools_dispatched.get(m.group(1), 0) + 1
        denials = sorted({m.group(1) for m in
                          re.finditer(r"(\w+) tool permission denied", log_text)})

    usage = data.get("usage") or {}
    run = {
        "num_turns": data.get("num_turns"),
        "duration_ms": data.get("duration_ms"),
        "cost_usd": data.get("total_cost_usd"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "result_text": redact(str(data.get("result", "")))[:4000],
        "is_error": data.get("is_error", False),
        # loaded = name visible in debug log (includes the mere skills
        # listing); invoked = the Skill tool actually dispatched.
        "skill_loaded": skill_loaded,
        "skill_invoked": tools_dispatched.get("Skill", 0) > 0,
        "tools_dispatched": tools_dispatched,
        "denials": denials,
    }
    if check in _CHECKS:
        run["check"] = _CHECKS[check](workdir)
        run["failure_mode"] = _failure_mode(run)
    return run


def _failure_mode(run: dict) -> str:
    """Distinguish HOW a run missed the success criterion — 'did nothing'
    and 'attempted but blocked' demand opposite skill fixes."""
    if run.get("check", {}).get("success"):
        return "succeeded"
    dispatched = run.get("tools_dispatched", {})
    wrote = any(t in dispatched for t in ("Write", "Edit", "NotebookEdit"))
    blocked = bool(run.get("denials"))
    if wrote:
        return "attempted_but_blocked" if blocked else "attempted_but_failed"
    if blocked:
        return "blocked_before_attempt"
    return "did_nothing"


def _judge_corrections(cfg: Config, corrections: list[dict], result_text: str) -> list[str]:
    if not corrections or not result_text:
        return []
    rules = [f"{i}: {c['correction'][:200]}" for i, c in enumerate(corrections[:15])]
    data = llm.complete_json(
        "A user historically had to make these corrections during this kind of task:\n"
        + "\n".join(rules) + "\n\n"
        "Below is the final output of a new automated run of a similar task. "
        "Which corrections (by index) would the user likely have to repeat, "
        "judging only from this output? Be conservative.\n"
        'Reply ONLY JSON: {"recurred": [indices], "notes": "one sentence"}\n\n'
        f"Output:\n{result_text[:3000]}",
        model=cfg.model, max_tokens=512)
    idx = [i for i in data.get("recurred", []) if isinstance(i, int) and 0 <= i < len(corrections)]
    return [corrections[i]["correction"][:120] for i in idx]


def _skill_description(skill_dir: Path) -> str:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^---\s*\n(.*?)\n---", text, re.S)
    return (m.group(1) if m else text)[:2000]


def _classify_trigger(cfg: Config, description: str, prompt: str) -> bool:
    """Is this prompt the skill's core trigger scenario? A skill can win on
    its trigger case and merely add overhead elsewhere — the verdict must
    see the difference."""
    data = llm.complete_json(
        "A Claude Code skill has this frontmatter description (including "
        "when it should and should not trigger):\n---\n" + description +
        "\n---\n\nIs the following user prompt an example of the skill's "
        "CORE trigger scenario (not merely related to its topic)?\n"
        f"Prompt: {prompt[:500]}\n\n"
        'Reply ONLY JSON: {"on_trigger": true/false, "reason": "one clause"}',
        model=cfg.model, max_tokens=128)
    return bool(data.get("on_trigger"))


def estimate(cfg: Config, n_prompts: int) -> str:
    n_runs = n_prompts * 2
    return (
        f"Planned: {n_runs} headless runs ({n_prompts} prompts x with/without skill), "
        f"each capped at {cfg.eval_max_turns} turns / {cfg.eval_timeout_s}s.\n"
        f"Expected wall time ~{n_runs * 1}-{n_runs * 4} min (runs are sequential). "
        "Token cost is billed to whatever `claude -p` is logged into "
        "(subscription usage, not API dollars, if you use a Claude plan); "
        f"rough order: 20k-80k tokens per run, {n_runs * 20}k-{n_runs * 80}k total. "
        "API cost: only the corrections-recurrence judge (skipped when the "
        "cluster has no mined corrections)."
    )


def evaluate_skill(cfg: Config, skill_name: str, n_prompts: int = 3,
                   check: str | None = None) -> dict:
    skill_dir = cfg.skills_dir / skill_name
    meta_path = skill_dir / ".skill-miner.json"
    if not skill_dir.is_dir():
        raise SystemExit(f"Skill not found: {skill_dir}")
    if not meta_path.is_file():
        raise SystemExit(f"{skill_dir} was not built by skill-miner "
                         "(missing .skill-miner.json provenance)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    corrections = meta.get("mined_corrections", [])

    n_prompts = max(3, min(5, n_prompts))
    samples = _sample_prompts(cfg, meta, n_prompts)
    if not samples:
        raise SystemExit("No usable prompts found in this skill's cluster history.")

    print(estimate(cfg, len(samples)))
    out_dir = cfg.out_dir / "eval" / skill_name
    out_dir.mkdir(parents=True, exist_ok=True)
    disabled_parent = cfg.claude_dir / ".skill-miner-disabled"
    disabled_parent.mkdir(exist_ok=True)

    description = _skill_description(skill_dir)
    rows = []
    for i, sample in enumerate(samples):
        on_trigger = _classify_trigger(cfg, description, sample["prompt"])
        row = {"session_id": sample["session_id"], "prompt": sample["prompt"][:300],
               "on_trigger": on_trigger}
        for condition in ("with_skill", "without_skill"):
            workdir = Path(tempfile.mkdtemp(prefix=f"sm-eval-{condition}-"))
            moved = False
            try:
                _seed_workdir(workdir, check)
                if condition == "without_skill":
                    shutil.move(str(skill_dir), str(disabled_parent / skill_name))
                    moved = True
                print(f"  [{i + 1}/{len(samples)}] {condition} ...")
                run = _run_headless(cfg, sample["prompt"], workdir, skill_name, check)
            finally:
                if moved:
                    shutil.move(str(disabled_parent / skill_name), str(skill_dir))
                shutil.rmtree(workdir, ignore_errors=True)
            if "error" not in run and corrections:
                run["corrections_recurred"] = _judge_corrections(
                    cfg, corrections, run["result_text"])
            status = run.get("error") or (
                f"turns={run.get('num_turns')} skill_loaded={run.get('skill_loaded')}"
                + (f" success={run['check']['success']}" if "check" in run else ""))
            print(f"      -> {status}")
            row[condition] = run
        rows.append(row)

    report = _summarize(skill_name, rows)
    (out_dir / "eval.json").write_text(
        json.dumps({"skill": skill_name, "check": check, "rows": rows,
                    "summary": report}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / "report.md").write_text(_render(skill_name, rows, report), encoding="utf-8")
    print(f"Report: {out_dir / 'report.md'}")
    return report


def _summarize(skill_name: str, rows: list[dict]) -> dict:
    def runs(cond):
        return [r[cond] for r in rows
                if isinstance(r.get(cond), dict) and "error" not in r[cond]]

    def avg(cond, key):
        vals = [r.get(key) for r in runs(cond) if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    def successes(cond):
        checked = [r for r in runs(cond) if "check" in r]
        return sum(1 for r in checked if r["check"]["success"]), len(checked)

    sw, sw_n = successes("with_skill")
    so, so_n = successes("without_skill")
    summary = {
        "prompts": len(rows),
        "avg_turns_with": avg("with_skill", "num_turns"),
        "avg_turns_without": avg("without_skill", "num_turns"),
        "avg_out_tokens_with": avg("with_skill", "output_tokens"),
        "avg_out_tokens_without": avg("without_skill", "output_tokens"),
        "skill_loaded_with": sum(1 for r in runs("with_skill") if r.get("skill_loaded")),
        "skill_loaded_without": sum(1 for r in runs("without_skill") if r.get("skill_loaded")),
        "skill_invoked_with": sum(1 for r in runs("with_skill") if r.get("skill_invoked")),
        "skill_invoked_on_trigger": sum(
            1 for r in rows if r.get("on_trigger")
            and isinstance(r.get("with_skill"), dict)
            and r["with_skill"].get("skill_invoked")),
        "skill_invoked_off_trigger": sum(
            1 for r in rows if not r.get("on_trigger")
            and isinstance(r.get("with_skill"), dict)
            and r["with_skill"].get("skill_invoked")),
        "success_with": f"{sw}/{sw_n}" if sw_n else None,
        "success_without": f"{so}/{so_n}" if so_n else None,
        "corrections_recurred_with": sum(
            len(r.get("corrections_recurred", [])) for r in runs("with_skill")),
        "corrections_recurred_without": sum(
            len(r.get("corrections_recurred", [])) for r in runs("without_skill")),
    }
    improvements = []
    if sw_n and so_n and sw > so:
        improvements.append("more success-criterion passes")
    tw, two = summary["avg_turns_with"], summary["avg_turns_without"]
    if tw is not None and two is not None and tw < two:
        improvements.append("fewer turns")
    if summary["corrections_recurred_with"] < summary["corrections_recurred_without"]:
        improvements.append("fewer recurring corrections")
    ow, owo = summary["avg_out_tokens_with"], summary["avg_out_tokens_without"]
    if ow is not None and owo is not None and ow < owo:
        improvements.append("fewer output tokens")
    summary["improvements"] = improvements

    # Per-prompt breakdown: aggregate stats hide a skill that dominates its
    # trigger scenario while adding overhead everywhere else.
    def bucket(rows_subset):
        w = sum(1 for r in rows_subset
                if r.get("with_skill", {}).get("check", {}).get("success"))
        wo = sum(1 for r in rows_subset
                 if r.get("without_skill", {}).get("check", {}).get("success"))
        return {"prompts": len(rows_subset), "success_with": w, "success_without": wo}

    on_rows = [r for r in rows if r.get("on_trigger")]
    off_rows = [r for r in rows if not r.get("on_trigger")]
    summary["on_trigger"] = bucket(on_rows)
    summary["off_trigger"] = bucket(off_rows)

    on, off = summary["on_trigger"], summary["off_trigger"]
    trigger_dominates = (on["prompts"] > 0
                         and on["success_with"] > on["success_without"])
    elsewhere_gain = off["success_with"] > off["success_without"]
    if improvements:
        summary["verdict"] = "improvement"
    elif trigger_dominates and not elsewhere_gain:
        summary["verdict"] = "narrow_the_trigger"
    else:
        summary["verdict"] = "no_improvement"
    summary["flag_no_improvement"] = summary["verdict"] == "no_improvement"
    if summary["skill_loaded_with"] == 0:
        summary["flag_skill_never_loaded"] = True
    return summary


def _render(skill_name: str, rows: list[dict], s: dict) -> str:
    lines = [f"# Eval: {skill_name}", ""]
    if s.get("flag_skill_never_loaded"):
        lines += ["> **FLAG: the skill never appeared in any run's debug log** — "
                  "measurements below compare baseline vs baseline; fix the "
                  "skill description/triggering before trusting them.", ""]
    on, off = s["on_trigger"], s["off_trigger"]
    verdict_text = {
        "improvement": f"**Verdict: improvement** — {', '.join(s['improvements'])}",
        "narrow_the_trigger": (
            "**Verdict: narrow the trigger.** The skill wins where it is "
            "meant to fire but only adds overhead elsewhere — tighten its "
            "description rather than deleting it."),
        "no_improvement": ("**Verdict: no measured improvement.** Consider "
                           "revising or removing this skill."),
    }[s["verdict"]]
    lines += [
        verdict_text,
        "",
        f"Wins on trigger scenarios: {on['success_with']}/{on['prompts']} with "
        f"vs {on['success_without']}/{on['prompts']} without · elsewhere: "
        f"{off['success_with']}/{off['prompts']} with vs "
        f"{off['success_without']}/{off['prompts']} without",
        "",
    ]
    lines += [
        "| metric | with skill | without skill |",
        "|---|---|---|",
        f"| success criterion | {s['success_with']} | {s['success_without']} |",
        f"| avg turns | {s['avg_turns_with']} | {s['avg_turns_without']} |",
        f"| avg output tokens | {s['avg_out_tokens_with']} | {s['avg_out_tokens_without']} |",
        f"| skill loaded (runs) | {s['skill_loaded_with']} | {s['skill_loaded_without']} |",
        f"| skill invoked (runs) | {s.get('skill_invoked_with', '?')} "
        f"(on-trigger {s.get('skill_invoked_on_trigger', '?')}, "
        f"off-trigger {s.get('skill_invoked_off_trigger', '?')}) | — |",
        f"| corrections recurred | {s['corrections_recurred_with']} | {s['corrections_recurred_without']} |",
        "",
    ]
    for i, r in enumerate(rows):
        trig = "on-trigger" if r.get("on_trigger") else "off-trigger"
        lines += [f"## Prompt {i + 1} ({trig}, session {r['session_id'][:8]})", "",
                  f"> {r['prompt'][:250]}", ""]
        for cond in ("with_skill", "without_skill"):
            run = r.get(cond, {})
            if "error" in run:
                lines.append(f"- **{cond}**: ERROR — {run['error']}")
                continue
            chk = run.get("check")
            mode = run.get("failure_mode", "")
            lines.append(
                f"- **{cond}**: {run.get('num_turns')} turns, "
                f"{run.get('output_tokens')} out-tokens, "
                f"skill_loaded={run.get('skill_loaded')}"
                + (f", success={chk['success']}" if chk else "")
                + (f", mode={mode}" if mode else "")
                + (f" ({chk['reason']})" if chk else ""))
            if run.get("denials"):
                lines.append(f"  - permission denials: {', '.join(run['denials'])}")
            if chk and chk.get("excerpt"):
                first = chk["excerpt"].splitlines()
                lines.append(f"  - CLAUDE.md after: `{' / '.join(first[:3])[:150]}`")
        lines.append("")
    return "\n".join(lines)
