---
name: skill-miner
description: >
  Mine the user's Claude Code session history for recurring workflows and
  turn them into personal skills. Use this whenever the user asks to analyze
  their session history, find repeated workflows, discover skill candidates,
  generate skills from past sessions, or measure whether a skill actually
  helps — including phrasings like "what skills should I make", "look at my
  past sessions", "mine my history", "is this skill actually useful", or
  "rerun the skill eval".
---

# Skill Miner

Drive the `skill-miner` CLI (installed via `pip install -e .` from this
repo) end to end: transcripts → arcs → clusters → scored proposals →
generated skills → measured A/B verdicts. The pipeline is staged and
cached; run stages in order and never re-implement one by hand — re-runs
are cheap and, with the default backend, byte-reproducible.

## Prerequisites

- `skill-miner` on PATH (`pip install -e <repo>` once).
- `ANTHROPIC_API_KEY` set for `propose`, `build`, `eval`; `scan` is fully
  offline. If the key is missing, run `scan`, then tell the user exactly
  which stages need the key.

## Workflow

1. **Scan** — `skill-miner scan`
   Parses every transcript under `~/.claude/projects/` into a redacted
   cache. If the user mentions sensitive projects, add them to
   `exclude_projects` in `~/.skill-miner/config.toml` BEFORE scanning,
   then `scan --force`.

2. **Propose** — `skill-miner propose`
   Segments sessions into topic arcs, clusters them (deterministic embed
   backend by default; `--backend llm` for more semantic recall at the
   cost of run-to-run variance), scores each cluster, and writes
   `~/.skill-miner/out/report.md` + `proposals.json`. Walk the user
   through the report's sub-scores, the corpus ceiling line (totals must
   be read against it, not against 10), and the evidence receipts. Let
   the user pick proposal IDs — do not auto-build everything.

3. **Build** — `skill-miner build P001 ...`
   Writes `~/.claude/skills/<name>/SKILL.md` per accepted ID, mining the
   cluster's correction turns into Gotchas. Review the generated skill
   with the user — the generator writes drafts, not truth. `--force` to
   overwrite, but ask first.

4. **Eval** — `skill-miner eval <name> --prompts 3..5 --check claude-md`
   A/B headless runs (skill installed vs temporarily removed) over real
   prompts from the skill's own arcs. Warn the user first: runs bill their
   Claude login and take minutes each; the CLI prints a cost/time estimate
   before starting. Read the verdict from the report:
   - `improvement` — keep the skill;
   - `narrow_the_trigger` — the skill wins its trigger case but adds
     overhead elsewhere: tighten its description, then re-eval;
   - `no_improvement` — recommend revising or deleting.
   Check `skill invoked` (not just loaded) and the per-run failure modes
   (`did_nothing` vs `attempted_but_blocked` demand opposite fixes).

## Gotchas

- Never bypass the CLI to read transcripts and send them to an API
  yourself: parse-time redaction plus the llm.py fence is the privacy
  guarantee ("no unredacted transcript content leaves the machine").
- Small corpus (< ~30 sessions) means weak clusters and n=1 trigger
  buckets in evals — present verdicts as directional, not conclusive.
- Eval replays the user's historical prompts, which can contain live
  commands ("run claude update"). The evaluator's narrow exec allowlist
  exists for that reason — do not widen it to unblock a run without
  telling the user what it exposes.
- During `eval`, the skill under test is briefly moved out of
  `~/.claude/skills`; avoid running other work that depends on it in
  parallel.
- An eval verdict is only as good as the harness. If runs fail oddly
  (unparseable output, mass permission denials, cap-hit turns), fix the
  harness and rerun rather than interpreting broken runs as skill failure.
