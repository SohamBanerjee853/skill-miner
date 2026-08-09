# skill-miner

Mine your Claude Code session history for recurring workflows and turn them
into personal skills (`~/.claude/skills/<name>/SKILL.md`).

## Pipeline

```
~/.claude/projects/**/*.jsonl
        │
        ▼
1. scan      parser.py       parse + redact + detect corrections  →  ~/.skill-miner/cache/sessions/
        │
        ▼
2.           signatures.py   segment each session into 1-4 topic ARCS; one 5-10 word
                             signature per arc, tagged with its parent session_id
                             (API, cached by session content hash)
        │
        ▼
3. propose   clustering.py   group similar arc signatures (LLM backend, swappable)
             scoring.py      rubric sub-scores per cluster of arcs
             propose.py      →  out/report.md + out/proposals.json (ranked, P001…)
                             previous proposals kept as proposals.prev.json
        │
        ▼
4. build     generator.py    accepted IDs → ~/.claude/skills/<name>/SKILL.md
                             (mines your actual corrections into Gotchas rules)
        │
        ▼
5. eval      evaluator.py    A/B: headless `claude -p` with vs. without the skill
                             →  out/eval/<skill>/report.md
```

## Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=...   # needed for propose/build/eval, not for scan
```

## Usage

```bash
skill-miner scan                 # parse all session transcripts (offline, idempotent)
skill-miner propose              # signatures + clustering + ranked proposals
skill-miner build P001 P003      # generate SKILL.md for accepted proposals
skill-miner eval doc-to-spec     # A/B test a generated skill
```

Read `~/.skill-miner/out/report.md` after `propose`. Each proposal shows the
rubric **sub-scores** with their exact formulas, not just a total:

| sub-score | formula |
|---|---|
| frequency | `min(10, 1 + arcs_in_cluster)` |
| friction | `min(10, 4 × avg correction turns per arc)` (heuristics: "no,", "that's not", repeated near-identical instructions, …) |
| re-explanation | `min(10, avg_shared_setup_words / 4)` — content words shared between arc-opening prompts, averaged over pairs; the words themselves are shown as receipts |
| consistency | LLM-judged — do the arcs follow the same repeatable steps? |
| stability | LLM-judged — will a playbook stay useful over months? |

Total = weighted mean (weights shown in the report and `proposals.json`).
The report also states the **max attainable total for your corpus** (e.g.
friction is capped by how many correction turns your whole history contains),
so totals are read against a realistic ceiling rather than 10.

**Cluster eligibility**: ≥ `min_cluster_size` distinct sessions (default 3),
or ≥ 2 distinct sessions when the arcs' signatures are mechanically
near-identical (avg pairwise content-word Jaccard ≥ 0.5) — useful for small
corpora.

**Evidence receipts**: every proposal's case text is backed by verbatim
(redacted) transcript quotes with session ids. Quotes are verified
mechanically against the cache; the model is re-prompted once on failures
and unverifiable evidence is dropped rather than displayed.

## Privacy

- A redaction pass (API keys, tokens, JWTs, passwords, emails, private key
  blocks, URL credentials, long hex secrets) runs at **parse time**, so the
  on-disk cache never contains raw secrets.
- `llm.py` is the only module that calls the Anthropic API and it re-redacts
  every outbound string as a second fence. **No unredacted transcript content
  ever leaves the machine.**
- Exclude sensitive projects entirely via `exclude_projects` in config.

## Configuration

`~/.skill-miner/config.toml` (all optional — see `config.example.toml`):

```toml
model = "claude-sonnet-4-6"
min_cluster_size = 3
exclude_projects = ["secret-project", "*client-x*"]   # matches dir name or path
eval_max_turns = 15
eval_timeout_s = 600
```

## Caching / idempotency

- `scan` keeps an index of `(mtime, size)` per transcript — unchanged files
  are skipped. `--force` re-parses everything.
- Signatures are cached by session **content hash**: re-running `propose`
  only pays API calls for new or changed sessions. LLM rubric judgments are
  cached by cluster content too.
- `build` refuses to overwrite an existing skill without `--force`.

## Evaluation notes

`eval` moves the skill dir to `~/.claude/.skill-miner-disabled/` for the
baseline runs and restores it in a `finally:` block. Each prompt runs with
`--permission-mode acceptEdits` in a throwaway temp working directory
seeded with a small toy project (so resume/document-style prompts have real
state to act on), with `--max-turns` capped. Per run the report records:

- whether the skill actually loaded (its name grepped from the run's
  `--debug-file` log) — if it never loaded, the whole comparison is flagged
  as baseline-vs-baseline;
- an optional mechanical success criterion (`--check claude-md`: does
  CLAUDE.md contain plausible resume state *after* the run — the seed
  CLAUDE.md is deliberately too thin to pass on its own);
- turns/tokens, and LLM-judged recurrence of historically mined corrections
  (skipped when the cluster produced none).

An estimated cost/time is printed before the runs start.

**Verdicts** are three-way, computed from a per-prompt breakdown rather than
aggregates alone. Each sampled prompt is classified (LLM-judged against the
skill's own description) as on-trigger or off-trigger, and each failed run
gets a failure mode from its debug log (`did_nothing`,
`attempted_but_blocked`, `attempted_but_failed`, `blocked_before_attempt`):

- `improvement` — wins on at least one global axis;
- `narrow_the_trigger` — dominates its trigger scenarios but only adds
  overhead elsewhere: tighten the description, don't delete the skill;
- `no_improvement` — no wins anywhere; revise or remove.

The report shows "wins on trigger scenarios: X/Y ... elsewhere: X/Y" so an
aggregate tie can't hide a decisive trigger-case win.

## Clustering backends

`clustering.py` defines a `ClusterBackend` protocol
(`cluster(signatures: dict[arc_id, signature]) -> list[Cluster]`). Two
backends ship:

- `--backend embed` (**default**): deterministic agglomerative clustering
  over local IDF-weighted content-word vectors (case/plural-normalized;
  cosine, average linkage, threshold 0.30). Identical inputs always produce
  identical clusters, and nothing leaves the machine. Compared against the
  LLM backend on a real 47-arc corpus it has perfect precision (every pair
  it groups, the LLM also groups) but lower recall — it can't see synonymy
  beyond shared words. Swap in a real embedding API later by overriding
  `EmbeddingClusterBackend.embed()`.
- `--backend llm`: batches signatures into one prompt; groups semantically
  but is **non-deterministic** across runs (observed: two runs on identical
  arcs produced different cluster boundaries).

**Why embed is the default**: cluster *membership* is decided mechanically,
so proposals are reproducible and diffable across runs. The LLM still
contributes where it is safe: a refinement pass may **merge whole clusters
and improve names/themes — never split one or reassign individual arcs** —
and both that refinement and the per-proposal case text are cached keyed by
exact membership, so an unchanged corpus re-proposes byte-identically
(verified). `--backend llm` remains available when you want maximum
semantic recall and accept the run-to-run variance.

## Prior art

Two adjacent projects exist (as of mid-2026); skill-miner deliberately
occupies a different point in the design space:

- [sugarforever's mining-session-skills](https://github.com/sugarforever/01coder-agent-skills)
  — a Claude Code skill that mines the *current/single session* for skill
  candidates, with an LLM judgment layer deciding what is worth keeping.
  Great for "turn what we just did into a skill"; it does not look across a
  whole corpus of history.
- [shutootaki's generating-skills-from-logs](https://github.com/shutootaki/skills)
  ([write-up](https://zenn.dev/takiko/articles/claude-code-skill-from-logs?locale=en))
  — a skill that analyzes session *and CLI* history corpus-wide to spot
  repetitive workflows and generate skills. It runs in-session each time
  (no persistent parse/signature cache) and generation is where it stops —
  no measurement of whether a generated skill helps.

What skill-miner adds that neither has: **corpus-scale mechanical scoring**
(frequency/friction/re-explanation with formulas and verbatim evidence
receipts, not just LLM judgment) and the **measured A/B eval loop** —
headless with/without runs, per-run skill-invocation ground truth, a
mechanical success criterion, and a three-way verdict that can tell
"narrow the trigger" apart from "delete it". The flag→salvage→improvement
case study below only exists because the eval loop closed.

## Lessons (learned the hard way, encoded in the code)

**`skill_loaded` ≠ `skill_invoked`.** The skill's name appears in a run's
debug log merely because the available-skills *listing* is logged — grep
for that and every with-skill run looks like the skill "worked". Actual use
is a `Skill` tool dispatch. The evaluator reports both; only invocation
means the skill influenced the run.

**Replaying history can execute history.** Eval replays real prompts from
your past sessions — and one of ours contained "run claude update". With a
blanket Bash allowance, the eval harness would have *actually self-updated
the CLI mid-eval*. Hence the narrow exec allowlist
(`Bash(git *) Bash(python *) Bash(echo *)`…): treat historical prompts as
untrusted input with side effects, because they are.

**Evidence receipts need mechanical verification.** Early proposals stated
plausible specifics ("gate enums", "on constrained interfaces like mobile")
with no way to tell grounded from confabulated. Requiring a verbatim quote
per claim — and verifying each quote is a real substring of the claimed
session's prompts, re-prompting once, dropping what fails — turned the
case text from vibes into citations. (Some "hallucinations" turned out to
be real: the mobile claim traced to an actual prompt.)

**Prove the harness before trusting its verdicts.** The same skill "failed"
three evals for three unrelated reasons: cmd.exe truncating multi-line
argv prompts (fix: pipe via stdin), permission denials starving skill-led
runs (fix: explicit allowlist), and a turn cap below what the skill's final
phase needed. Only after the harness demonstrably passed a probe did the
A/B numbers mean anything.

### Case study: P004 flag → salvage → improvement

First eval of the generated `session-resume-safe-close` (broad trigger,
"manage the whole session lifecycle"):

| metric | with skill | without |
|---|---|---|
| success (CLAUDE.md holds resume state) | 2/3 | 2/3 |
| avg turns | 16.7 | 8.0 |

Aggregate verdict: *no improvement* — but the per-prompt data showed the
skill winning decisively on its one true trigger prompt ("I accidentally
closed the terminal": baseline answered conversationally in 1 turn and
persisted nothing) while adding pure overhead elsewhere. That pattern is
now a first-class verdict (`narrow_the_trigger`). After narrowing the
description to session-loss/recovery only (+ "persist via Write/Edit, never
shell copies", + conditional env-verify):

| metric | with skill | without |
|---|---|---|
| success | 3/5 | 2/5 |
| on-trigger wins | 1/1 | 0/1 |
| skill invoked | 1 run — the on-trigger one | — |

Verdict: *improvement*, and the invocation pattern confirms the narrowed
description triggers exactly where intended.

## Limitations (honest)

- **n-sensitivity**: a ~20-session corpus yields 2-5-arc clusters; scores
  and verdicts move a lot per added session, and an on-trigger eval bucket
  of n=1 is directional, not conclusive. The rubric's corpus ceiling makes
  this visible but doesn't cure it.
- **LLM cluster nondeterminism**: `--backend llm` gives better semantic
  recall but different boundaries run to run; even the default's LLM
  refinement introduces variance on a *cold* cache (cached thereafter).
- **IDF vectors are an embedding floor**: word-overlap cosine cannot see
  synonymy ("record demo" vs "capture screencast" won't pair). It is the
  deterministic floor, not the ceiling — `EmbeddingClusterBackend.embed()`
  is the seam for a real embedding model.
- Correction detection is heuristic (regex + near-duplicate similarity);
  quiet frustration ("fine, I'll do it myself") goes uncounted.

## Transcript schema (as observed, Claude Code ~2.1.x)

One JSON object per line. `type` ∈ {`user`, `assistant`, `system`, plus
metadata kinds like `mode`, `file-history-snapshot`, `ai-title`, …}.
Typed user prompts have `message.content` as a *string*; tool results arrive
as user-type lines whose content is a list of `tool_result` blocks. Assistant
lines carry `text`/`thinking`/`tool_use` blocks (`tool_use.input.file_path`
for file tools) and `message.usage` token counts. Noise is filtered via
`isMeta`, `isSidechain`, and wrapper tags (`<command-name>`,
`<local-command-stdout>`, `<task-notification>`, …). Session-id-named
*directories* next to the `.jsonl` files are not transcripts.
