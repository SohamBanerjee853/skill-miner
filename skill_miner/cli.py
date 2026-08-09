"""skill-miner CLI: scan | propose | build | eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="skill-miner",
        description="Mine Claude Code session history for reusable skills.")
    parser.add_argument("--config", type=Path, default=None,
                        help="path to config.toml (default: ~/.skill-miner/config.toml)")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="cache/output dir (default: ~/.skill-miner)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="parse session transcripts into the local cache")
    p_scan.add_argument("--force", action="store_true", help="re-parse everything")

    p_prop = sub.add_parser("propose", help="cluster workflows and rank skill proposals")
    p_prop.add_argument("--force", action="store_true",
                        help="ignore signature/judgment caches")
    p_prop.add_argument("--backend", default="embed",
                        help="clustering backend: embed (deterministic, default) or llm")
    p_prop.add_argument("--min-cluster-size", type=int, default=None)

    p_build = sub.add_parser("build", help="generate SKILL.md for accepted proposals")
    p_build.add_argument("ids", nargs="+", metavar="PROPOSAL_ID", help="e.g. P001 P002")
    p_build.add_argument("--force", action="store_true", help="overwrite existing skills")

    p_eval = sub.add_parser("eval", help="A/B test a generated skill via headless claude")
    p_eval.add_argument("skill", help="skill name (dir under ~/.claude/skills)")
    p_eval.add_argument("--prompts", type=int, default=3, help="prompts to sample (3-5)")
    p_eval.add_argument("--check", choices=["claude-md", "git-clean"], default=None,
                        help="mechanical success criterion applied to each run's workdir")

    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)

    from .llm import LLMError
    try:
        return _dispatch(args, cfg)
    except LLMError as e:
        print(f"error: {e}")
        return 2


def _dispatch(args, cfg) -> int:

    if args.command == "scan":
        from .parser import scan
        stats = scan(cfg, force=args.force)
        print(json.dumps(stats, indent=1))
        print(f"Cache: {cfg.sessions_cache_dir}")

    elif args.command == "propose":
        if args.min_cluster_size:
            cfg.min_cluster_size = args.min_cluster_size
        from .propose import propose
        out = propose(cfg, force=args.force, backend_name=args.backend)
        print(f"\n{len(out['proposals'])} proposals -> {cfg.report_path}")
        for p in out["proposals"]:
            print(f"  {p['id']}  {p['total']:>4}/10  {p['skill_name']} "
                  f"({len(p['session_ids'])} sessions)")

    elif args.command == "build":
        from .generator import build_skills
        built = build_skills(cfg, args.ids, force=args.force)
        if built:
            print(f"\nBuilt {len(built)} skill(s). "
                  f"Evaluate with: skill-miner eval <name>")

    elif args.command == "eval":
        from .evaluator import evaluate_skill
        summary = evaluate_skill(cfg, args.skill, n_prompts=args.prompts,
                                 check=args.check)
        print(json.dumps(summary, indent=1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
