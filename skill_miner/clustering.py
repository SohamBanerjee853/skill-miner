"""Stage 3a: group similar workflow signatures into clusters.

Backend protocol so an embeddings backend can be swapped in later:

    class ClusterBackend(Protocol):
        def cluster(self, signatures: dict[str, str]) -> list[Cluster]: ...

The default LLMClusterBackend batches signatures into one prompt and asks the
model for clusters with member session ids. For histories too large for one
prompt it chunks, then runs a merge pass over the chunk-level cluster names.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Protocol

from .config import Config
from . import llm

CHUNK_SIZE = 150


@dataclasses.dataclass
class Cluster:
    name: str
    theme: str
    member_ids: list[str]


class ClusterBackend(Protocol):
    def cluster(self, signatures: dict[str, str]) -> list[Cluster]: ...


class LLMClusterBackend:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def cluster(self, signatures: dict[str, str]) -> list[Cluster]:
        items = list(signatures.items())
        if not items:
            return []
        chunks = [items[i:i + CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]
        clusters: list[Cluster] = []
        for chunk in chunks:
            clusters.extend(self._cluster_chunk(dict(chunk)))
        if len(chunks) > 1:
            clusters = self._merge(clusters)
        return self._validate(clusters, signatures)

    def _cluster_chunk(self, signatures: dict[str, str]) -> list[Cluster]:
        listing = "\n".join(f"{sid}: {sig}" for sid, sig in signatures.items())
        data = llm.complete_json(
            "Below are workflow signatures of coding sessions, one per line as "
            "'session_id: signature'. Group sessions that follow the SAME "
            "recurring workflow pattern (would benefit from the same reusable "
            "playbook). Singletons are fine as their own cluster. Reply with "
            "ONLY JSON:\n"
            '{"clusters": [{"name": "kebab-case-name", '
            '"theme": "one sentence describing the shared workflow", '
            '"members": ["session_id", ...]}]}\n\n'
            f"Sessions:\n{listing}",
            model=self.cfg.model, max_tokens=4096)
        return [Cluster(c.get("name", f"cluster-{i}"), c.get("theme", ""),
                        list(c.get("members", [])))
                for i, c in enumerate(data.get("clusters", []))]

    def _merge(self, clusters: list[Cluster]) -> list[Cluster]:
        listing = "\n".join(f"{i}: {c.name} — {c.theme} ({len(c.member_ids)} sessions)"
                            for i, c in enumerate(clusters))
        data = llm.complete_json(
            "These clusters came from separate batches and may overlap. Merge "
            "clusters describing the same workflow. Reply with ONLY JSON:\n"
            '{"merged": [{"name": "kebab-case-name", "theme": "one sentence", '
            '"indices": [0, 3]}]}\n\n'
            f"Clusters:\n{listing}",
            model=self.cfg.model, max_tokens=2048)
        merged = []
        used: set[int] = set()
        for m in data.get("merged", []):
            idxs = [i for i in m.get("indices", []) if isinstance(i, int)
                    and 0 <= i < len(clusters) and i not in used]
            if not idxs:
                continue
            used.update(idxs)
            members = [sid for i in idxs for sid in clusters[i].member_ids]
            merged.append(Cluster(m.get("name", clusters[idxs[0]].name),
                                  m.get("theme", clusters[idxs[0]].theme), members))
        merged.extend(c for i, c in enumerate(clusters) if i not in used)
        return merged

    @staticmethod
    def _validate(clusters: list[Cluster], signatures: dict[str, str]) -> list[Cluster]:
        """Drop hallucinated ids; assign each session to at most one cluster;
        orphans become singletons."""
        assigned: set[str] = set()
        out = []
        for c in clusters:
            members = [sid for sid in c.member_ids
                       if sid in signatures and sid not in assigned]
            assigned.update(members)
            if members:
                out.append(Cluster(c.name, c.theme, members))
        for sid in signatures:
            if sid not in assigned:
                out.append(Cluster(f"singleton-{sid[:8]}", signatures[sid], [sid]))
        return out


class EmbeddingClusterBackend:
    """Deterministic clustering over locally computed signature vectors.

    "Embeddings" here are IDF-weighted bags of normalized content words
    (lowercased, plural-stripped) — computed locally, so nothing leaves the
    machine and identical inputs always produce identical clusters. To swap
    in a real embedding service later, override embed() and keep cluster()
    as is.

    Clustering is average-linkage agglomerative over cosine similarity with
    a fixed merge threshold; all iteration orders are sorted and ties break
    on the lowest (i, j) pair, so the result is fully deterministic.
    """

    def __init__(self, cfg: Config, threshold: float = 0.30):
        # 0.30 chosen by sweep against LLM clustering on a real 47-arc corpus:
        # highest recall with zero spurious pairs (precision 1.0); below 0.30
        # surface-form false merges appear.
        self.cfg = cfg
        self.threshold = threshold

    # -- vectorization ------------------------------------------------------

    def embed(self, texts: list[str]) -> list[dict[str, float]]:
        import math
        from .scoring import _content_words
        docs = [sorted(_content_words(t)) for t in texts]
        df: dict[str, int] = {}
        for words in docs:
            for w in words:
                df[w] = df.get(w, 0) + 1
        n = max(1, len(docs))
        vecs = []
        for words in docs:
            v = {w: 1.0 + math.log(n / df[w]) for w in words}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            vecs.append({w: x / norm for w, x in v.items()})
        return vecs

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if len(b) < len(a):
            a, b = b, a
        return sum(x * b.get(w, 0.0) for w, x in a.items())

    # -- clustering ---------------------------------------------------------

    def cluster(self, signatures: dict[str, str]) -> list[Cluster]:
        ids = sorted(signatures)  # stable order regardless of dict order
        if not ids:
            return []
        vecs = self.embed([signatures[i] for i in ids])
        sim = [[self._cosine(vecs[i], vecs[j]) for j in range(len(ids))]
               for i in range(len(ids))]
        groups: list[list[int]] = [[i] for i in range(len(ids))]

        def linkage(g1: list[int], g2: list[int]) -> float:
            return sum(sim[i][j] for i in g1 for j in g2) / (len(g1) * len(g2))

        while len(groups) > 1:
            best, pair = self.threshold, None
            for gi in range(len(groups)):
                for gj in range(gi + 1, len(groups)):
                    s = linkage(groups[gi], groups[gj])
                    if s > best or (s == best and pair is None and s >= self.threshold):
                        best, pair = s, (gi, gj)
            if pair is None:
                break
            gi, gj = pair
            groups[gi] = sorted(groups[gi] + groups[gj])
            del groups[gj]
            groups.sort()  # keep index order canonical after every merge

        clusters = []
        for g in groups:
            members = [ids[i] for i in g]
            name, theme = self._label(g, vecs, [signatures[ids[i]] for i in g])
            clusters.append(Cluster(name, theme, members))
        return clusters

    def _label(self, group: list[int], vecs: list[dict[str, float]],
               sigs: list[str]) -> tuple[str, str]:
        """name = top shared words; theme = medoid member signature."""
        weight: dict[str, float] = {}
        for i in group:
            for w, x in vecs[i].items():
                weight[w] = weight.get(w, 0.0) + x
        top = [w for w, _ in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
        name = "-".join(top) if top else "cluster"
        if len(group) == 1:
            return name, sigs[0]
        medoid = max(range(len(group)), key=lambda k: (
            sum(self._cosine(vecs[group[k]], vecs[j]) for j in group), -k))
        return name, sigs[medoid]


def llm_refine(cfg: Config, clusters: list[Cluster],
               signatures: dict[str, str], cache: dict) -> list[Cluster]:
    """Optional LLM pass over mechanically built clusters: merge whole
    clusters that describe the same workflow, and improve names/themes.

    The LLM may only COARSEN — merge entire clusters — never split one or
    reassign individual arcs, so the deterministic embed cores survive
    intact. Results are cached by the exact cluster memberships, so
    re-running on unchanged input replays the cached refinement instead of
    re-rolling the dice. Degrades gracefully (returns input unchanged) when
    the API is unavailable.
    """
    import hashlib
    from . import llm as _llm

    key_src = json.dumps(sorted(sorted(c.member_ids) for c in clusters))
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    if key in cache:
        data = cache[key]
    else:
        listing = "\n".join(
            f"{i}: {c.name} — {c.theme} | members: "
            + "; ".join(signatures[m] for m in c.member_ids[:6])
            for i, c in enumerate(clusters))
        try:
            data = _llm.complete_json(
                "These workflow clusters were built mechanically. Merge whole "
                "clusters that describe the SAME recurring workflow (do not "
                "split any cluster), and give each resulting group a "
                "kebab-case name and a one-sentence theme. Clusters you leave "
                "out stay as they are. Reply ONLY JSON:\n"
                '{"groups": [{"name": "kebab-name", "theme": "one sentence", '
                '"indices": [0, 3]}]}\n\n'
                f"Clusters:\n{listing}",
                model=cfg.model, max_tokens=4096)
        except _llm.LLMError as e:
            print(f"  (LLM refine skipped: {e})")
            return clusters
        cache[key] = data

    used: set[int] = set()
    out: list[Cluster] = []
    for g in data.get("groups", []):
        idxs = [i for i in g.get("indices", []) if isinstance(i, int)
                and 0 <= i < len(clusters) and i not in used]
        if not idxs:
            continue
        used.update(idxs)
        members = [m for i in idxs for m in clusters[i].member_ids]
        out.append(Cluster(str(g.get("name", clusters[idxs[0]].name))[:60],
                           str(g.get("theme", clusters[idxs[0]].theme))[:300],
                           members))
    out.extend(c for i, c in enumerate(clusters) if i not in used)
    return out


def get_backend(cfg: Config, name: str = "llm") -> ClusterBackend:
    if name == "llm":
        return LLMClusterBackend(cfg)
    if name in ("embed", "embeddings"):
        return EmbeddingClusterBackend(cfg)
    raise ValueError(f"Unknown clustering backend: {name!r} (available: llm, embed)")
