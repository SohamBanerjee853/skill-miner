"""Stage 3c: orchestrate arcs -> clusters -> scored, evidence-backed proposals.

Eligibility: a cluster becomes a proposal when it spans >= min_cluster_size
distinct sessions, OR >= 2 distinct sessions when its arc signatures are
mechanically very similar (avg pairwise content-word Jaccard >= 0.5) — small
corpora can't produce big clusters, but two near-identical workflows are
already a real signal.

Every proposal carries receipts: verbatim (already-redacted) transcript
quotes with session ids backing the case text, verified mechanically against
the cache — the generator is re-prompted once if a quote doesn't verify, and
unverifiable evidence is dropped rather than shown.

Outputs report.md + proposals.json; the previous proposals.json is kept as
proposals.prev.json for diffing.
"""

from __future__ import annotations

import json
import re
import shutil

from .config import Config
from .parser import load_sessions
from .signatures import build_arcs
from .clustering import get_backend
from .scoring import (score_cluster, corpus_ceiling, WEIGHTS, FORMULAS,
                      _content_words)

_SIMILARITY_FOR_PAIR = 0.5


def _sig_similarity(signatures: list[str]) -> float:
    """Avg pairwise content-word Jaccard (reporting only)."""
    sets = [_content_words(s) for s in signatures]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    vals = [len(a & b) / len(a | b) for a, b in pairs if a | b]
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def _best_cross_session_pair(arcs: list[dict]) -> tuple[float, str, str]:
    """Strongest signature match between arcs of DIFFERENT sessions.

    Clusters often include contextual neighbor arcs, so an average over all
    pairs drowns the core repeating pair; what makes a 2-session cluster
    real is one strong cross-session repeat.
    """
    best, pair = 0.0, ("", "")
    for i, a in enumerate(arcs):
        wa = _content_words(a["signature"])
        for b in arcs[i + 1:]:
            if b["session_id"] == a["session_id"]:
                continue
            wb = _content_words(b["signature"])
            if not (wa | wb):
                continue
            sim = len(wa & wb) / len(wa | wb)
            if sim > best:
                best, pair = sim, (a["arc_id"], b["arc_id"])
    return round(best, 3), pair[0], pair[1]


def _eligible(cluster, arcs: list[dict], cfg: Config) -> tuple[bool, str]:
    n_sessions = len({a["session_id"] for a in arcs})
    if n_sessions >= cfg.min_cluster_size:
        return True, f">= {cfg.min_cluster_size} sessions"
    if n_sessions >= 2:
        sim, a1, a2 = _best_cross_session_pair(arcs)
        if sim >= _SIMILARITY_FOR_PAIR:
            short = lambda aid: f"{aid.split('#')[0][:8]}#{aid.split('#')[1]}"
            return True, (f"2 sessions with matching arc pair "
                          f"{short(a1)} ~ {short(a2)} (similarity {sim})")
    return False, ""


# --- evidence-backed proposal meta ----------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _verify_evidence(evidence: list, arcs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Check each quote is a verbatim substring of some prompt in the claimed
    session (or, failing that, any member session — then fix the id)."""
    prompts_by_sid: dict[str, list[str]] = {}
    for a in arcs:
        prompts_by_sid.setdefault(a["session_id"], []).extend(
            _normalize(p["text"]) for p in a["prompts"])
    ok, bad = [], []
    for e in evidence or []:
        quote = str(e.get("quote", ""))
        claim = str(e.get("claim", ""))[:300]
        sid = str(e.get("session_id", ""))
        nq = _normalize(quote)
        if len(nq) < 15:
            bad.append(e)
            continue
        if any(nq in t for t in prompts_by_sid.get(sid, [])):
            ok.append({"claim": claim, "session_id": sid, "quote": quote[:300]})
            continue
        other = next((s for s, ts in prompts_by_sid.items()
                      if any(nq in t for t in ts)), None)
        if other:
            ok.append({"claim": claim, "session_id": other, "quote": quote[:300]})
        else:
            bad.append(e)
    return ok, bad


def _meta_prompt(cluster, arcs: list[dict]) -> str:
    lines = []
    for a in arcs[:8]:
        first = next((p["text"] for p in a["prompts"] if not p["is_correction"]),
                     a["prompts"][0]["text"])
        lines.append(f"- [{a['session_id']}] {first[:400]}")
    return (
        "A recurring Claude Code sub-workflow was found across sessions.\n"
        f"Cluster: {cluster.name}\nTheme: {cluster.theme}\n"
        "Arc signatures: " + "; ".join(a["signature"] for a in arcs[:8]) + "\n"
        "Opening prompts (session_id + verbatim text):\n" + "\n".join(lines) + "\n\n"
        "Propose a reusable skill. EVERY concrete claim in `case` (named "
        "techniques, formats, constraints, contexts) must be backed by an "
        "evidence entry whose `quote` is copied VERBATIM from the prompts "
        "above (substring, unaltered). If you cannot quote support for a "
        "claim, do not make the claim. Reply ONLY JSON:\n"
        '{"skill_name": "kebab-case-name", '
        '"description": "one line: what the skill does", '
        '"case": "2-3 sentences, only evidenced claims", '
        '"evidence": [{"claim": "short claim", "session_id": "...", '
        '"quote": "verbatim excerpt from a prompt above"}]}'
    )


def _proposal_meta(cfg: Config, cluster, arcs: list[dict], cache: dict) -> dict:
    import hashlib
    from . import llm
    key = hashlib.sha256(json.dumps(
        sorted(a["arc_id"] for a in arcs)).encode()).hexdigest()[:16]
    if key in cache:
        return cache[key]
    prompt = _meta_prompt(cluster, arcs)
    data = llm.complete_json(prompt, model=cfg.model, max_tokens=1024)
    ok, bad = _verify_evidence(data.get("evidence"), arcs)
    if bad:  # one retry with the failures named
        failed = "; ".join(str(e.get("quote", ""))[:60] for e in bad)
        data = llm.complete_json(
            prompt + "\n\nThese quotes were NOT verbatim substrings of the "
            f"prompts and were rejected: {failed}\nFix: quote exactly, and "
            "drop any claim you cannot support.",
            model=cfg.model, max_tokens=1024)
        ok, bad = _verify_evidence(data.get("evidence"), arcs)
    result = {
        "skill_name": str(data.get("skill_name", cluster.name))[:60],
        "description": str(data.get("description", cluster.theme))[:300],
        "case": str(data.get("case", ""))[:1000],
        "evidence": ok,
        "unverified_evidence_dropped": len(bad),
    }
    cache[key] = result
    return result


# --- orchestration ---------------------------------------------------------

def _load_json_cache(path, force: bool) -> dict:
    if path.is_file() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def propose(cfg: Config, force: bool = False, backend_name: str = "embed") -> dict:
    cfg.ensure_dirs()
    sessions = load_sessions(cfg)
    if not sessions:
        raise SystemExit("No cached sessions. Run 'skill-miner scan' first.")

    print(f"Segmenting {len(sessions)} sessions into arcs...")
    arcs = build_arcs(cfg, sessions, force=force)
    arcs_by_id = {a["arc_id"]: a for a in arcs}
    print(f"{len(arcs)} arcs from {len(sessions)} sessions")

    print(f"Clustering arc signatures ({backend_name} backend)...")
    signatures = {a["arc_id"]: a["signature"] for a in arcs}
    clusters = get_backend(cfg, backend_name).cluster(signatures)

    if backend_name != "llm":
        # Deterministic cores from the mechanical backend; the LLM may only
        # merge whole clusters and improve labels (cached by membership).
        from .clustering import llm_refine
        refine_cache = _load_json_cache(cfg.refine_cache_path, force)
        clusters = llm_refine(cfg, clusters, signatures, refine_cache)
        cfg.refine_cache_path.write_text(
            json.dumps(refine_cache, ensure_ascii=False, indent=1), encoding="utf-8")

    judgment_cache = _load_json_cache(cfg.judgments_path, force)
    meta_cache = _load_json_cache(cfg.meta_cache_path, force)

    total_corrections = sum(s["correction_count"] for s in sessions)
    ceiling = corpus_ceiling(len(arcs), total_corrections)

    proposals = []
    for cluster in clusters:
        members = [arcs_by_id[aid] for aid in cluster.member_ids if aid in arcs_by_id]
        is_eligible, why = _eligible(cluster, members, cfg)
        if not is_eligible:
            continue
        scores = score_cluster(cfg, cluster, members, judgment_cache)
        meta = _proposal_meta(cfg, cluster, members, meta_cache)
        proposals.append({
            "cluster_name": cluster.name,
            "theme": cluster.theme,
            "eligibility": why,
            "arc_ids": [a["arc_id"] for a in members],
            "arcs": [{"arc_id": a["arc_id"], "session_id": a["session_id"],
                      "start": a["start"], "end": a["end"],
                      "signature": a["signature"]} for a in members],
            "session_ids": sorted({a["session_id"] for a in members}),
            **meta,
            **scores,
        })
    cfg.judgments_path.write_text(
        json.dumps(judgment_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg.meta_cache_path.write_text(
        json.dumps(meta_cache, ensure_ascii=False, indent=1), encoding="utf-8")

    proposals.sort(key=lambda p: -p["total"])
    for i, p in enumerate(proposals):
        p["id"] = f"P{i + 1:03d}"

    out = {
        "min_cluster_size": cfg.min_cluster_size,
        "pair_rule": f"or 2 sessions with signature similarity >= {_SIMILARITY_FOR_PAIR}",
        "weights": WEIGHTS,
        "formulas": FORMULAS,
        "corpus": {"sessions": len(sessions), "arcs": len(arcs),
                   "total_corrections": total_corrections},
        "ceiling": ceiling,
        "num_clusters": len(clusters),
        "all_clusters": [
            {"name": c.name, "theme": c.theme, "arc_ids": c.member_ids}
            for c in clusters],
        "proposals": proposals,
    }
    if cfg.proposals_path.is_file():
        shutil.copy2(cfg.proposals_path, cfg.proposals_path.with_suffix(".prev.json"))
    cfg.proposals_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg.report_path.write_text(_render_report(out), encoding="utf-8")
    return out


# --- report ----------------------------------------------------------------

def _render_report(data: dict) -> str:
    corpus = data["corpus"]
    ceil = data["ceiling"]
    lines = [
        "# Skill proposals",
        "",
        f"Corpus: {corpus['sessions']} sessions -> {corpus['arcs']} arcs · "
        f"{data['num_clusters']} clusters · {len(data['proposals'])} proposals",
        "",
        f"Eligibility: >= {data['min_cluster_size']} sessions, {data['pair_rule']}.",
        "",
        "Sub-score formulas (weights): "
        + " · ".join(f"{k} = {data['formulas'][k]} ({data['weights'][k]:.0%})"
                     for k in data["weights"]),
        "",
        f"**Max attainable total in this corpus: {ceil['total']}/10** — "
        f"friction is capped at {ceil['per_sub_score']['friction']}/10 because the whole "
        f"history contains only {corpus['total_corrections']} correction turn(s); "
        f"frequency is capped at {ceil['per_sub_score']['frequency']}/10 by corpus size. "
        "Read proposal totals against this ceiling, not against 10.",
        "",
    ]
    if not data["proposals"]:
        lines.append("_No cluster met the eligibility rule._")
    for p in data["proposals"]:
        s = p["sub_scores"]
        d = p["detail"]
        r = p["receipts"]["re_explanation"]
        lines += [
            f"## {p['id']} · `{p['skill_name']}` — {p['total']}/10",
            "",
            f"**{p['description']}**",
            "",
            f"{p['case']}",
            "",
            f"- Eligible via: {p['eligibility']}",
            f"- Arcs: {d['arcs']} across {d['sessions']} sessions",
        ]
        for a in p["arcs"]:
            lines.append(f"  - [{a['session_id'][:8]}#{a['arc_id'].split('#')[1]}] "
                         f"prompts {a['start']}-{a['end']}: {a['signature']}")
        lines += [
            "",
            "| frequency | friction | re-explanation | consistency | stability |",
            "|---|---|---|---|---|",
            f"| {s['frequency']} | {s['friction']} | {s['re_explanation']} "
            f"| {s['consistency']} | {s['stability']} |",
            "",
            f"_friction: {d['avg_corrections_per_arc']} corrections/arc · "
            f"re-explanation: {r['avg_shared_words']} shared setup words/pair · "
            f"judge: {d['judge_rationale']}_",
            "",
            "**Receipts**",
        ]
        for e in p["evidence"]:
            lines.append(f"- claim: {e['claim']} — [{e['session_id'][:8]}] "
                         f"\"{e['quote'][:200]}\"")
        if not p["evidence"]:
            lines.append("- _no verifiable claims; case text is limited to the "
                         "cluster theme_")
        if p.get("unverified_evidence_dropped"):
            lines.append(f"- _{p['unverified_evidence_dropped']} unverifiable "
                         "quote(s) dropped_")
        if r["shared_words"]:
            lines.append(f"- shared setup words behind the re-explanation score: "
                         f"`{'`, `'.join(r['shared_words'][:15])}`")
        for o in r["openings"][:6]:
            sid, ak = o["arc_id"].split("#")
            lines.append(f"  - [{sid[:8]}#{ak}] opening: \"{o['excerpt']}\"")
        lines += ["", f"Build it: `skill-miner build {p['id']}`", ""]
    return "\n".join(lines)
