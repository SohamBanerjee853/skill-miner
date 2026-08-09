"""Stage 4 (build): generate ~/.claude/skills/<name>/SKILL.md for accepted
proposals.

The cluster's actual (redacted) transcripts are mined for correction turns —
each correction plus the instruction that preceded it becomes candidate
material for explicit rules/gotchas in the generated skill. SKILL.md follows
the skill-creator conventions: YAML frontmatter with name and a deliberately
"pushy" description (trigger contexts included), imperative body under 500
lines, rationale explained rather than bare MUSTs.
"""

from __future__ import annotations

import json
import re

from .config import Config
from .parser import load_sessions
from . import llm


def _arc_slices(proposal: dict, sessions_by_id: dict) -> list[dict]:
    """Materialize the proposal's arcs as prompt slices from the cache."""
    slices = []
    for a in proposal.get("arcs", []):
        s = sessions_by_id.get(a["session_id"])
        if not s:
            continue
        prompts = s["user_prompts"][a["start"]: a["end"] + 1]
        if prompts:
            slices.append({"session_id": a["session_id"],
                           "arc_id": a["arc_id"], "prompts": prompts,
                           "tools_used": s["tools_used"]})
    if not slices:  # pre-arc proposals.json: fall back to whole sessions
        for sid in proposal["session_ids"]:
            s = sessions_by_id.get(sid)
            if s:
                slices.append({"session_id": sid, "arc_id": f"{sid}#a0",
                               "prompts": s["user_prompts"],
                               "tools_used": s["tools_used"]})
    return slices


def _mine_corrections(slices: list[dict]) -> list[dict]:
    """Correction turns (within the arcs) with the preceding instruction."""
    mined = []
    for sl in slices:
        prompts = sl["prompts"]
        for i, p in enumerate(prompts):
            if not p["is_correction"]:
                continue
            prior = prompts[i - 1]["text"][:300] if i > 0 else ""
            mined.append({
                "session_id": sl["session_id"],
                "prior_instruction": prior,
                "correction": p["text"][:400],
                "reasons": p["correction_reasons"],
            })
    return mined


def _generation_prompt(proposal: dict, slices: list[dict],
                       corrections: list[dict]) -> str:
    openings = []
    for sl in slices[:6]:
        first = next((p["text"] for p in sl["prompts"] if not p["is_correction"]),
                     sl["prompts"][0]["text"])
        openings.append(first[:400])
    tools: dict[str, int] = {}
    for sl in slices:
        for k, v in sl["tools_used"].items():
            tools[k] = tools.get(k, 0) + v
    evidence = "\n".join(
        f"- [{e['session_id'][:8]}] \"{e['quote'][:200]}\" (supports: {e['claim']})"
        for e in proposal.get("evidence", []))
    corr_lines = [
        f"- after being told: \"{c['prior_instruction']}\" the user corrected: "
        f"\"{c['correction']}\" (signals: {', '.join(c['reasons'])})"
        for c in corrections[:20]]
    return f"""Write a complete SKILL.md for a Claude Code personal skill.

Skill name: {proposal['skill_name']}
What it covers: {proposal['description']}
Workflow theme: {proposal['theme']}

Example opening prompts from real arcs of this workflow:
{chr(10).join('- ' + o for o in openings)}

Verified transcript evidence for the workflow's specifics (ground your
instructions in these; do not invent capabilities beyond them):
{evidence if evidence else '(none)'}

Tool usage across these arcs: {json.dumps(dict(sorted(tools.items(), key=lambda kv: -kv[1])[:10]))}

Corrections the user had to make during these sessions (mine these into
explicit rules/gotchas so future runs do NOT repeat the mistakes):
{chr(10).join(corr_lines) if corr_lines else '(none captured)'}

Requirements for the SKILL.md you produce:
- Start with YAML frontmatter containing exactly `name` and `description`.
- The description must state what the skill does AND when to use it, phrased
  a little "pushy" so it triggers reliably (list concrete user phrasings and
  contexts that should activate it).
- Body: imperative instructions for the workflow, under 300 lines. Explain
  WHY behind non-obvious rules instead of bare MUSTs. Generalize from the
  examples; never hardcode one-off file names or project names.
- Include a "## Gotchas" section that turns the corrections above into
  explicit rules (each rule = what to do + why, derived from a real
  correction). Skip corrections that are one-off noise.
- Do not invent capabilities the sessions don't show.

Reply with ONLY the SKILL.md content, no code fences, no commentary."""


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\bname:\s*(\S+).*?\n---\s*\n", re.S)


def build_skills(cfg: Config, proposal_ids: list[str], force: bool = False) -> list[dict]:
    if not cfg.proposals_path.is_file():
        raise SystemExit("No proposals.json. Run 'skill-miner propose' first.")
    data = json.loads(cfg.proposals_path.read_text(encoding="utf-8"))
    by_pid = {p["id"]: p for p in data["proposals"]}
    sessions_by_id = {s["session_id"]: s for s in load_sessions(cfg)}

    results = []
    for pid in proposal_ids:
        proposal = by_pid.get(pid)
        if not proposal:
            print(f"!! Unknown proposal id {pid} (have: {', '.join(by_pid)})")
            continue
        name = re.sub(r"[^a-z0-9\-]", "-", proposal["skill_name"].lower()).strip("-")
        skill_dir = cfg.skills_dir / name
        skill_path = skill_dir / "SKILL.md"
        if skill_path.exists() and not force:
            print(f"!! {skill_path} exists; use --force to overwrite")
            continue

        slices = _arc_slices(proposal, sessions_by_id)
        corrections = _mine_corrections(slices)
        print(f"Generating {name} from {len(slices)} arcs, "
              f"{len(corrections)} mined corrections...")
        content = llm.complete(
            _generation_prompt(proposal, slices, corrections),
            model=cfg.model, max_tokens=8192)
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```[a-z]*\n|\n```$", "", content, flags=re.S)
        m = _FRONTMATTER_RE.match(content)
        if not m:
            content = f"---\nname: {name}\ndescription: {proposal['description']}\n---\n\n" + content

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(content, encoding="utf-8", newline="\n")
        # Provenance sidecar so eval can find the cluster later.
        (skill_dir / ".skill-miner.json").write_text(json.dumps({
            "proposal_id": pid,
            "cluster_name": proposal["cluster_name"],
            "session_ids": proposal["session_ids"],
            "arcs": proposal.get("arcs", []),
            "mined_corrections": corrections,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"   -> {skill_path}")
        results.append({"id": pid, "name": name, "path": str(skill_path)})
    return results
