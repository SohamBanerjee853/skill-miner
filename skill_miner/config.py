"""Configuration and filesystem layout for skill-miner.

Data lives under ~/.skill-miner/ by default so re-runs are cheap regardless of
where the CLI is invoked from:

    ~/.skill-miner/
        config.toml                  (optional user config)
        cache/sessions/<id>.json     normalized, REDACTED session records
        cache/scan_index.json        jsonl path -> {mtime, size, session_id}
        cache/signatures.json        session content hash -> signature
        cache/judgments.json         cluster hash -> LLM rubric judgment
        out/proposals.json
        out/report.md
        out/eval/<skill>/...
"""

from __future__ import annotations

import dataclasses
import fnmatch
import os
import tomllib
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclasses.dataclass
class Config:
    claude_dir: Path
    data_dir: Path
    model: str = DEFAULT_MODEL
    exclude_projects: list[str] = dataclasses.field(default_factory=list)
    min_cluster_size: int = 3
    max_prompt_chars: int = 2000     # per-prompt truncation in the cache
    signature_prompt_count: int = 8  # prompts sent (redacted) per signature call
    # High enough that multi-phase skills (verify -> work -> persist) can
    # reach their final phase; 15 proved too low in practice.
    eval_max_turns: int = 25
    eval_timeout_s: int = 600

    @property
    def projects_dir(self) -> Path:
        return self.claude_dir / "projects"

    @property
    def skills_dir(self) -> Path:
        return self.claude_dir / "skills"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def sessions_cache_dir(self) -> Path:
        return self.cache_dir / "sessions"

    @property
    def scan_index_path(self) -> Path:
        return self.cache_dir / "scan_index.json"

    @property
    def signatures_path(self) -> Path:
        return self.cache_dir / "signatures.json"

    @property
    def judgments_path(self) -> Path:
        return self.cache_dir / "judgments.json"

    @property
    def refine_cache_path(self) -> Path:
        return self.cache_dir / "refine.json"

    @property
    def meta_cache_path(self) -> Path:
        return self.cache_dir / "proposal_meta.json"

    @property
    def out_dir(self) -> Path:
        return self.data_dir / "out"

    @property
    def proposals_path(self) -> Path:
        return self.out_dir / "proposals.json"

    @property
    def report_path(self) -> Path:
        return self.out_dir / "report.md"

    def ensure_dirs(self) -> None:
        for d in (self.sessions_cache_dir, self.out_dir):
            d.mkdir(parents=True, exist_ok=True)

    def is_excluded(self, project_dir_name: str, project_path: str | None) -> bool:
        """Match exclude patterns against both the encoded dir name and the
        decoded cwd path (case-insensitive, glob or substring)."""
        candidates = [project_dir_name.lower()]
        if project_path:
            candidates.append(project_path.replace("\\", "/").lower())
        for pattern in self.exclude_projects:
            p = pattern.replace("\\", "/").lower()
            for c in candidates:
                if p in c or fnmatch.fnmatch(c, p) or fnmatch.fnmatch(c, f"*{p}*"):
                    return True
        return False


def load_config(config_path: Path | None = None, data_dir: Path | None = None) -> Config:
    home = Path.home()
    data_dir = data_dir or home / ".skill-miner"
    cfg = Config(claude_dir=home / ".claude", data_dir=data_dir)

    path = config_path or data_dir / "config.toml"
    if path.is_file():
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        cfg.model = raw.get("model", cfg.model)
        cfg.exclude_projects = list(raw.get("exclude_projects", []))
        cfg.min_cluster_size = int(raw.get("min_cluster_size", cfg.min_cluster_size))
        cfg.eval_max_turns = int(raw.get("eval_max_turns", cfg.eval_max_turns))
        cfg.eval_timeout_s = int(raw.get("eval_timeout_s", cfg.eval_timeout_s))
        if "claude_dir" in raw:
            cfg.claude_dir = Path(os.path.expanduser(raw["claude_dir"]))
    return cfg
