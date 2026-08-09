"""Stage 3b: usefulness rubric for a cluster of workflow ARCS.

Sub-scores (each 0-10, all surfaced in the report — never just the total):
  frequency          mechanical: min(10, 1 + number of arcs in the cluster)
  friction           mechanical: min(10, 4 x correction turns per arc)
  re_explanation     mechanical: min(10, avg shared setup words / 4), where
                     "shared setup words" = content words appearing in BOTH
                     arcs' opening prompts, averaged over all arc pairs.
                     The actual shared words are returned as receipts.
  consistency        LLM-judged: do arcs follow the same steps?
  stability          LLM-judged: will this workflow stay relevant?

Total = weighted mean. corpus_ceiling() computes the maximum total attainable
given the corpus (e.g. friction is capped by how many correction turns exist
in the whole history), so a 3.7/10 can be read against its real ceiling.
"""

from __future__ import annotations

import hashlib
import json
import re

from .config import Config
from . import llm

WEIGHTS = {
    "frequency": 0.30,
    "friction": 0.25,
    "re_explanation": 0.20,
    "consistency": 0.15,
    "stability": 0.10,
}

FORMULAS = {
    "frequency": "min(10, 1 + arcs_in_cluster)",
    "friction": "min(10, 4 x avg_corrections_per_arc)",
    "re_explanation": "min(10, avg_shared_setup_words / 4)",
    "consistency": "LLM-judged 1-10",
    "stability": "LLM-judged 1-10",
}

_WORD_RE = re.compile(r"[a-z0-9_\-/\.]{3,}")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "you", "your", "from", "have",
    "are", "was", "were", "will", "can", "should", "would", "please", "want",
    "need", "make", "sure", "then", "than", "them", "they", "its", "it's",
    "not", "but", "all", "any", "into", "out", "use", "using", "when", "what",
    "how", "there", "here", "also", "just", "like", "each", "which",
}


def _content_words(text: str) -> set[str]:
    words = set()
    for w in _WORD_RE.findall(text.lower()):
        if w in _STOPWORDS:
            continue
        # crude plural strip so demo/demos, gif/gifs count as the same word
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        words.add(w)
    return words


def frequency_score(n_arcs: int) -> float:
    return float(min(10, 1 + n_arcs))


def friction_score(arcs: list[dict]) -> tuple[float, float]:
    """(score, avg corrections per arc)."""
    avg = sum(a["correction_count"] for a in arcs) / max(1, len(arcs))
    return min(10.0, round(avg * 4, 1)), round(avg, 2)


def _arc_opening(arc: dict) -> str:
    return next((p["text"] for p in arc["prompts"] if not p["is_correction"]),
                arc["prompts"][0]["text"])


def re_explanation_score(arcs: list[dict]) -> tuple[float, dict]:
    """Score = min(10, avg_shared_setup_words / 4).

    avg_shared_setup_words: for every pair of arcs, count content words
    present in both openings; average over pairs. Receipts include the
    words themselves and each arc's opening excerpt.
    """
    openings = [(a["arc_id"], _arc_opening(a)[:1500]) for a in arcs]
    word_sets = [(aid, _content_words(t)) for aid, t in openings]
    if len(word_sets) < 2:
        return 0.0, {"avg_shared_words": 0, "shared_words": [], "openings": []}

    shared_counts = []
    word_freq: dict[str, int] = {}
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            inter = word_sets[i][1] & word_sets[j][1]
            shared_counts.append(len(inter))
            for w in inter:
                word_freq[w] = word_freq.get(w, 0) + 1
    avg_shared = sum(shared_counts) / len(shared_counts) if shared_counts else 0.0
    score = min(10.0, round(avg_shared / 4, 1))
    # (-count, word) not just -count: ties must have a total order, or set
    # hash randomization reorders the receipts on every run.
    top_shared = [w for w, _ in
                  sorted(word_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]
    return score, {
        "avg_shared_words": round(avg_shared, 1),
        "shared_words": top_shared,
        "openings": [{"arc_id": aid, "excerpt": t[:200]} for aid, t in openings],
    }


def _judged_scores(cfg: Config, cluster_name: str, theme: str,
                   signatures: list[str], cache: dict) -> dict:
    key_src = json.dumps([cluster_name, sorted(signatures)], sort_keys=True)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    if key in cache:
        return cache[key]
    data = llm.complete_json(
        "Rate this cluster of coding-workflow arcs on two axes, 1-10 each.\n"
        f"Cluster: {cluster_name}\nTheme: {theme}\n"
        "Member arc signatures:\n" + "\n".join(f"- {s}" for s in signatures) + "\n\n"
        "consistency: do these arcs really follow the same repeatable steps "
        "(10 = near-identical workflow, 1 = grab-bag)?\n"
        "stability: would a written playbook for this stay useful over months "
        "(10 = durable process, 1 = one-off/transient)?\n"
        'Reply ONLY JSON: {"consistency": n, "stability": n, '
        '"rationale": "1-2 short sentences"}',
        model=cfg.model, max_tokens=384)
    rationale = str(data.get("rationale", ""))
    if len(rationale) > 600:  # cut at a sentence/word boundary, never mid-word
        cut = rationale[:600]
        rationale = cut[: max(cut.rfind(". ") + 1, cut.rfind(" "))].rstrip() + " …"
    result = {
        "consistency": float(max(1, min(10, data.get("consistency", 5)))),
        "stability": float(max(1, min(10, data.get("stability", 5)))),
        "rationale": rationale,
    }
    cache[key] = result
    return result


def score_cluster(cfg: Config, cluster, arcs: list[dict], judgment_cache: dict) -> dict:
    freq = frequency_score(len(arcs))
    fric, avg_corr = friction_score(arcs)
    reexp, reexp_detail = re_explanation_score(arcs)
    judged = _judged_scores(cfg, cluster.name, cluster.theme,
                            [a["signature"] for a in arcs], judgment_cache)
    subs = {
        "frequency": freq,
        "friction": fric,
        "re_explanation": reexp,
        "consistency": judged["consistency"],
        "stability": judged["stability"],
    }
    total = sum(subs[k] * w for k, w in WEIGHTS.items())
    return {
        "sub_scores": subs,
        "total": round(max(1.0, min(10.0, total)), 1),
        "detail": {
            "arcs": len(arcs),
            "sessions": len({a["session_id"] for a in arcs}),
            "avg_corrections_per_arc": avg_corr,
            "judge_rationale": judged["rationale"],
        },
        "receipts": {"re_explanation": reexp_detail},
    }


def corpus_ceiling(total_arcs: int, total_corrections: int,
                   min_sessions: int = 2) -> dict:
    """Best total any cluster could reach in THIS corpus.

    frequency: capped by putting every arc in one cluster.
    friction: capped by concentrating every correction turn in the corpus
      into the smallest eligible cluster (min_sessions arcs).
    re_explanation / consistency / stability: theoretical 10.
    """
    maxes = {
        "frequency": frequency_score(total_arcs),
        "friction": min(10.0, round(4 * total_corrections / max(1, min_sessions), 1)),
        "re_explanation": 10.0,
        "consistency": 10.0,
        "stability": 10.0,
    }
    total = round(sum(maxes[k] * w for k, w in WEIGHTS.items()), 1)
    return {"per_sub_score": maxes, "total": total}
