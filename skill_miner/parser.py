"""Stage 1 (scan): parse Claude Code session JSONL files into normalized,
redacted session records.

Transcript schema (verified against real files, Claude Code ~2.1.x):
  - one JSON object per line; `type` distinguishes record kinds
  - real content lives in type=="user" / type=="assistant"; other types
    ("mode", "permission-mode", "file-history-snapshot", "ai-title",
    "last-prompt", "attachment", "queue-operation", ...) are session metadata
  - user records: message.content is a STRING for typed prompts, or a LIST of
    tool_result blocks. Noise flags: isMeta, isSidechain, and command wrappers
    like <command-name>/<local-command-stdout>/<local-command-caveat>
  - assistant records: message.content is a list of text/thinking/tool_use
    blocks; tool_use has .name and .input (file tools carry input.file_path)
  - sibling DIRECTORIES named after session ids exist next to the .jsonl
    files; only top-level *.jsonl are transcripts
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path

from .config import Config
from .redaction import redact

# --- noise filtering -------------------------------------------------------

_NOISE_MARKERS = (
    "<command-name>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<system-reminder>",
    "[Request interrupted",
    "<task-notification>",
    "<teammate-message",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
)

# --- correction heuristics -------------------------------------------------

# Strong markers match anywhere; weak markers ("actually", "instead", ...)
# appear mid-sentence in ordinary instructions all the time, so they only
# count within the opening of the prompt where they signal a course-change.
CORRECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("thats-not", re.compile(r"\bthat'?s\s+(not|wrong|incorrect)\b", re.I)),
    ("not-what-i", re.compile(r"\bnot what i (asked|meant|wanted|said)\b", re.I)),
    ("i-said", re.compile(r"\bi (already )?(said|asked|told you|mentioned)\b", re.I)),
    ("undo", re.compile(r"\b(undo|revert|roll ?back)\b", re.I)),
    ("redo", re.compile(r"\b(redo|do it again|try again|start over)\b", re.I)),
    ("still-broken", re.compile(r"\bstill\s+(not|doesn'?t|isn'?t|won'?t|fails?|broken|wrong|missing)\b", re.I)),
    ("you-missed", re.compile(r"\byou\s+(didn'?t|ignored|missed|forgot|removed|broke)\b", re.I)),
    # Prohibitions only count when they reference something already done —
    # "stop doing that" is a correction; "do not touch file X" in a fresh
    # instruction is just guidance (real false positive we hit).
    ("stop-doing", re.compile(r"\bstop\s+(doing|using|changing|touching|adding|creating|making)\b", re.I)),
    ("why-did-you", re.compile(r"\bwhy\s+(did|are|would)\s+you\b", re.I)),
    ("dont-do-that", re.compile(r"\b(don'?t|do not)\s+do\s+(that|this|it)\b", re.I)),
]

_WEAK_WINDOW = 60  # chars from the start of the prompt
WEAK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("leading-no", re.compile(r"^\s*no\b[,.!\s]", re.I)),
    ("nope", re.compile(r"^\s*nope\b", re.I)),
    ("actually", re.compile(r"\bactually\b", re.I)),
    ("instead", re.compile(r"\binstead\b", re.I)),
    ("wrong", re.compile(r"^\s*(wrong|incorrect)\b", re.I)),
]

# Near-duplicate re-instruction: same ask issued twice is friction even
# without an explicit "no".
_SIMILARITY_THRESHOLD = 0.75
_MIN_LEN_FOR_SIMILARITY = 25


def detect_corrections(prompts: list[str]) -> list[list[str]]:
    """Return, per prompt, the list of correction reasons ([] = not a correction)."""
    reasons: list[list[str]] = []
    for i, text in enumerate(prompts):
        hits = [label for label, pat in CORRECTION_PATTERNS if pat.search(text)]
        opening = text[:_WEAK_WINDOW]
        hits += [label for label, pat in WEAK_PATTERNS if pat.search(opening)]
        if i == 0:
            # The first prompt of a session can't be correcting anything.
            reasons.append([])
            continue
        if len(text) >= _MIN_LEN_FOR_SIMILARITY:
            for prev in prompts[max(0, i - 6):i]:
                if len(prev) < _MIN_LEN_FOR_SIMILARITY:
                    continue
                ratio = difflib.SequenceMatcher(None, prev.lower(), text.lower()).ratio()
                if ratio >= _SIMILARITY_THRESHOLD:
                    hits.append("repeated-instruction")
                    break
        reasons.append(sorted(set(hits)))
    return reasons


# --- extraction ------------------------------------------------------------

def _user_prompt_text(record: dict) -> str | None:
    """Extract a genuine typed user prompt, or None for noise/tool results."""
    if record.get("isMeta") or record.get("isSidechain"):
        return None
    msg = record.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # image-paste etc. arrives as a block list; keep text blocks only
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        text = "\n".join(p for p in parts if p)
    else:
        return None
    text = text.strip()
    if not text or any(m in text for m in _NOISE_MARKERS):
        return None
    return text


def parse_session_file(path: Path, cfg: Config) -> dict | None:
    """Parse one session JSONL into a normalized, redacted record."""
    prompts_raw: list[dict] = []
    tools_used: dict[str, int] = {}
    files_touched: set[str] = set()
    project_path = None
    git_branch = None
    first_ts = last_ts = None
    n_assistant = 0

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtype = rec.get("type")
            ts = rec.get("timestamp")
            if ts:
                first_ts = first_ts or ts
                last_ts = ts
            if rec.get("cwd") and not project_path:
                project_path = rec["cwd"]
            if rec.get("gitBranch") and not git_branch:
                git_branch = rec["gitBranch"]

            if rtype == "user":
                text = _user_prompt_text(rec)
                if text is not None:
                    prompts_raw.append({"text": text, "timestamp": ts})
            elif rtype == "assistant":
                if rec.get("isSidechain"):
                    continue
                n_assistant += 1
                content = (rec.get("message") or {}).get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    name = block.get("name") or "?"
                    tools_used[name] = tools_used.get(name, 0) + 1
                    inp = block.get("input") or {}
                    for key in ("file_path", "notebook_path", "path"):
                        v = inp.get(key)
                        if isinstance(v, str) and v:
                            files_touched.add(v)

    if not prompts_raw:
        return None

    texts = [p["text"] for p in prompts_raw]
    correction_reasons = detect_corrections(texts)

    prompts = []
    for i, (p, reasons) in enumerate(zip(prompts_raw, correction_reasons)):
        prompts.append({
            "index": i,
            "text": redact(p["text"])[: cfg.max_prompt_chars],
            "timestamp": p["timestamp"],
            "is_correction": bool(reasons),
            "correction_reasons": reasons,
        })

    record = {
        "session_id": path.stem,
        "project_dir": path.parent.name,
        "project_path": redact(project_path or ""),
        "git_branch": git_branch,
        "started_at": first_ts,
        "ended_at": last_ts,
        "num_user_prompts": len(prompts),
        "num_assistant_messages": n_assistant,
        "correction_count": sum(1 for p in prompts if p["is_correction"]),
        "tools_used": dict(sorted(tools_used.items(), key=lambda kv: -kv[1])),
        "files_touched": sorted(redact(fp) for fp in files_touched)[:200],
        "user_prompts": prompts,
    }
    # Content hash keys the signature cache: same content -> cache hit.
    digest_src = json.dumps(
        {"prompts": [p["text"] for p in prompts], "tools": record["tools_used"]},
        ensure_ascii=False, sort_keys=True)
    record["content_hash"] = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:16]
    return record


# --- scan orchestration ----------------------------------------------------

def scan(cfg: Config, force: bool = False) -> dict:
    """Walk all project dirs; parse new/changed sessions; return summary stats.

    Idempotent: an index of (mtime, size) per jsonl means unchanged files are
    skipped entirely on re-runs.
    """
    cfg.ensure_dirs()
    index: dict = {}
    if cfg.scan_index_path.is_file() and not force:
        index = json.loads(cfg.scan_index_path.read_text(encoding="utf-8"))

    stats = {"parsed": 0, "cached": 0, "skipped_empty": 0, "excluded": 0, "sessions": 0}
    seen_keys = set()

    projects_dir = cfg.projects_dir
    if not projects_dir.is_dir():
        raise FileNotFoundError(f"No Claude projects dir at {projects_dir}")

    for project in sorted(projects_dir.iterdir()):
        if not project.is_dir():
            continue
        for jsonl in sorted(project.glob("*.jsonl")):
            key = str(jsonl)
            seen_keys.add(key)
            st = jsonl.stat()
            entry = index.get(key)
            cache_file = cfg.sessions_cache_dir / f"{jsonl.stem}.json"
            if (entry and entry.get("mtime") == st.st_mtime and entry.get("size") == st.st_size
                    and (cache_file.is_file() or entry.get("empty"))):
                stats["cached"] += 1
                continue

            record = parse_session_file(jsonl, cfg)
            if record is None:
                index[key] = {"mtime": st.st_mtime, "size": st.st_size, "empty": True}
                stats["skipped_empty"] += 1
                continue

            if cfg.is_excluded(record["project_dir"], record["project_path"]):
                index[key] = {"mtime": st.st_mtime, "size": st.st_size, "excluded": True}
                if cache_file.is_file():
                    cache_file.unlink()
                stats["excluded"] += 1
                continue

            cache_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            index[key] = {"mtime": st.st_mtime, "size": st.st_size,
                          "session_id": record["session_id"]}
            stats["parsed"] += 1

    # Drop cache entries for deleted transcripts.
    for key in list(index):
        if key not in seen_keys:
            sid = index[key].get("session_id")
            if sid:
                (cfg.sessions_cache_dir / f"{sid}.json").unlink(missing_ok=True)
            del index[key]

    cfg.scan_index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    stats["sessions"] = len(list(cfg.sessions_cache_dir.glob("*.json")))
    return stats


def load_sessions(cfg: Config) -> list[dict]:
    """Load all cached (already-redacted) session records."""
    sessions = []
    for f in sorted(cfg.sessions_cache_dir.glob("*.json")):
        sessions.append(json.loads(f.read_text(encoding="utf-8")))
    return sessions
