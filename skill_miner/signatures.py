"""Stage 2: segment sessions into topic ARCS and sign each arc.

A session like "build feature -> record GIF demo -> embed in README ->
publish" is 2-3 distinct sub-workflows, not one. One signature per session
buries the tail sub-workflows as trailing clauses, so clustering never sees
them. Instead each session is segmented into 1-4 topic arcs; every arc gets
its own 5-10 word signature and carries its parent session_id, and
clustering runs over arcs.

Cache: signatures.json maps session content_hash -> {"arcs": [{signature,
start, end}]}. Entries in the old single-signature format are treated as
misses and regenerated.
"""

from __future__ import annotations

import json

from .config import Config
from . import llm

_SYSTEM = (
    "You segment coding sessions into topic arcs. An arc is a contiguous run "
    "of user prompts pursuing one sub-workflow (e.g. 'build the feature', "
    "'record demo GIF and embed in README', 'document and publish to GitHub'). "
    "Most sessions have 1-3 arcs; only split where the goal genuinely shifts. "
    "Each arc gets a 5-10 word signature of the workflow pattern that ignores "
    "one-off specifics (no file names, no project names)."
)


def _segmentation_prompt(record: dict, cfg: Config) -> str:
    lines = []
    for p in record["user_prompts"][:30]:
        flag = " [correction]" if p["is_correction"] else ""
        lines.append(f"{p['index']}{flag}: {p['text'][:300]}")
    tools = ", ".join(f"{k}x{v}" for k, v in list(record["tools_used"].items())[:8])
    return (
        f"Session prompts (index: text):\n" + "\n".join(lines) +
        (f"\nTools used: {tools}\n" if tools else "\n") +
        "\nSegment into 1-4 arcs covering all prompt indices in order, "
        "non-overlapping. Reply ONLY JSON:\n"
        '{"arcs": [{"signature": "5-10 word workflow signature", '
        '"start": first_prompt_index, "end": last_prompt_index}]}'
    )


def _validate_arcs(raw: list, n_prompts: int) -> list[dict] | None:
    """Clamp/repair index ranges; reject if not an ordered cover."""
    arcs = []
    for a in raw:
        try:
            start, end = int(a["start"]), int(a["end"])
            sig = str(a["signature"]).strip().strip('"')[:120]
        except (KeyError, TypeError, ValueError):
            return None
        if not sig:
            return None
        arcs.append({"signature": sig, "start": max(0, start),
                     "end": min(n_prompts - 1, end)})
    if not arcs:
        return None
    arcs.sort(key=lambda a: a["start"])
    # Repair gaps/overlaps into a contiguous cover.
    arcs[0]["start"] = 0
    for prev, cur in zip(arcs, arcs[1:]):
        cur["start"] = prev["end"] + 1
        if cur["start"] > cur["end"]:
            return None
    arcs[-1]["end"] = n_prompts - 1
    return arcs


def _segment_session(record: dict, cfg: Config) -> list[dict]:
    n = record["num_user_prompts"]
    if n <= 2:
        # Too short to segment; one arc, one cheap signature call.
        sig = llm.complete(_segmentation_prompt(record, cfg) +
                           "\n(Short session: return exactly one arc.)",
                           model=cfg.model, max_tokens=256, system=_SYSTEM)
        try:
            arcs = _validate_arcs(llm._parse_json(sig).get("arcs", []), n)
            if arcs:
                return arcs[:1] if n <= 2 else arcs
        except Exception:
            pass
        return [{"signature": sig.strip().splitlines()[-1][:120], "start": 0, "end": n - 1}]
    data = llm.complete_json(_segmentation_prompt(record, cfg),
                             model=cfg.model, max_tokens=1024, system=_SYSTEM)
    arcs = _validate_arcs(data.get("arcs", []), n)
    if arcs is None:
        raise llm.LLMError(f"Unrepairable arc segmentation for {record['session_id']}")
    return arcs[:6]


def _load_cache(cfg: Config) -> dict:
    if cfg.signatures_path.is_file():
        return json.loads(cfg.signatures_path.read_text(encoding="utf-8"))
    return {}


def _materialize(record: dict, cached_arcs: list[dict]) -> list[dict]:
    """Attach prompt slices and correction counts to cached arc ranges."""
    out = []
    for k, a in enumerate(cached_arcs):
        prompts = record["user_prompts"][a["start"]: a["end"] + 1]
        if not prompts:
            continue
        out.append({
            "arc_id": f"{record['session_id']}#a{k}",
            "session_id": record["session_id"],
            "project_dir": record["project_dir"],
            "signature": a["signature"],
            "start": a["start"],
            "end": a["end"],
            "prompts": prompts,
            "correction_count": sum(1 for p in prompts if p["is_correction"]),
        })
    return out


def build_arcs(cfg: Config, sessions: list[dict], force: bool = False) -> list[dict]:
    """Return arc records for all sessions; API calls only for cache misses."""
    cache = {} if force else _load_cache(cfg)
    arcs: list[dict] = []
    misses = [s for s in sessions
              if "arcs" not in cache.get(s["content_hash"], {})]

    for i, s in enumerate(misses):
        seg = _segment_session(s, cfg)
        cache[s["content_hash"]] = {"session_id": s["session_id"],
                                    "model": cfg.model, "arcs": seg}
        print(f"  arcs {i + 1}/{len(misses)}: {s['session_id'][:8]} -> "
              + " | ".join(a["signature"] for a in seg))
        # Persist incrementally so an interrupt doesn't lose paid work.
        cfg.signatures_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    for s in sessions:
        arcs.extend(_materialize(s, cache[s["content_hash"]]["arcs"]))
    return arcs
