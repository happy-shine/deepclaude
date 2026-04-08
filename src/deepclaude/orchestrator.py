"""Evolution orchestrator — launches concurrent Claude Code instances."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from deepclaude import registry


CLAUDE_CMD = "claude"


@dataclass
class Config:
    n_parallel: int = 3
    max_rounds: int = 10
    top_k: int = 5
    max_iterations: int = 20
    project_root: str = "."
    data_dir: str = "/Users/shine/trader-data"
    composite_weights: dict = field(default_factory=lambda: {
        "ic_ir": 0.30, "sharpe": 0.20, "monotonicity": 0.15,
        "ic_positive_pct": 0.15, "long_return": 0.10, "decay": 0.10,
    })


def _load_prompt_template() -> str:
    """Load prompt template from package."""
    template_path = Path(__file__).parent / "prompt_template.md"
    return template_path.read_text(encoding="utf-8")


def build_prompt(top_k_factors: list[dict], config: Config) -> str:
    """Fill prompt template with top-K factors and config."""
    template = _load_prompt_template()

    if top_k_factors:
        lines = []
        for f in top_k_factors:
            lines.append(f"### {f.get('name', 'unnamed')} (score: {f.get('composite_score', '?')})")
            lines.append(f"```python\n{f.get('code', '')}\n```")
            metrics = f.get("metrics", {})
            lines.append(f"Metrics: IC_IR={metrics.get('ic_ir', '?')}, "
                         f"Sharpe={metrics.get('sharpe', '?')}, "
                         f"Mono={metrics.get('monotonicity', '?')}")
            if f.get("analysis"):
                lines.append(f"Analysis: {f['analysis']}")
            lines.append("")
        top_k_str = "\n".join(lines)
    else:
        top_k_str = "（首轮探索，无历史因子）"

    prompt = template.replace("{top_k_factors}", top_k_str)
    prompt = prompt.replace("{max_iterations}", str(config.max_iterations))

    return prompt


class Orchestrator:
    def __init__(self, config: Config):
        self.config = config

    def _make_workspace(self, session_id: str) -> str:
        ws = Path(self.config.project_root) / "workspace" / session_id
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "scratch").mkdir(exist_ok=True)
        return str(ws)

    def _launch_claude(self, prompt: str, workspace: str, session_id: str) -> subprocess.Popen:
        """Launch a Claude Code CLI process."""
        final_prompt = prompt.replace("{output_dir}", workspace)

        factor_dir = str(Path(self.config.project_root) / "factors")
        Path(factor_dir).mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update({
            "DEEPCLAUDE_DATA_DIR": self.config.data_dir,
            "DEEPCLAUDE_FACTOR_DIR": factor_dir,
            "DEEPCLAUDE_WORKSPACE": workspace,
            "DEEPCLAUDE_SESSION_ID": session_id,
        })

        cmd = [
            CLAUDE_CMD,
            "--print",
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--verbose",
            "--cwd", workspace,
            "-p", final_prompt,
        ]

        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc

    def _stream_output(self, proc: subprocess.Popen, session_id: str):
        """Read stream-json output in a thread, print status updates."""
        def _reader(pipe, label):
            for line in pipe:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event_type = event.get("type", "")
                    if event_type == "assistant":
                        msg = event.get("message", "")[:100]
                        print(f"[{session_id}] {msg}")
                    elif event_type == "tool_use":
                        name = event.get("name", "")
                        print(f"[{session_id}] tool: {name}")
                    elif event_type == "result":
                        print(f"[{session_id}] Done")
                except json.JSONDecodeError:
                    print(f"[{session_id}] {label}: {line[:120]}")

        t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        t_out.start()
        t_err.start()
        return t_out, t_err

    def run(self):
        """Execute the evolution loop."""
        print(f"=== DeepClaude Orchestrator ===")
        print(f"Config: {self.config.n_parallel} parallel, {self.config.max_rounds} rounds, "
              f"top {self.config.top_k} selection")
        print()

        for round_num in range(self.config.max_rounds):
            print(f"--- Round {round_num + 1}/{self.config.max_rounds} ---")

            top_k = registry.get_top_k(k=self.config.top_k)
            prompt = build_prompt(top_k, self.config)

            processes = []
            threads = []
            for i in range(self.config.n_parallel):
                session_id = f"r{round_num + 1:03d}_i{i + 1:03d}"
                workspace = self._make_workspace(session_id)
                proc = self._launch_claude(prompt, workspace, session_id)
                t_out, t_err = self._stream_output(proc, session_id)
                processes.append((proc, session_id))
                threads.extend([t_out, t_err])
                print(f"[{session_id}] Launched")

            for proc, sid in processes:
                proc.wait()
                print(f"[{sid}] Exited with code {proc.returncode}")

            for t in threads:
                t.join(timeout=5)

            new_top = registry.get_top_k(k=1)
            if new_top:
                best = new_top[0]
                print(f"\nBest so far: {best['name']} (score: {best['composite_score']})")

            print()

        final_top = registry.get_top_k(k=self.config.top_k)
        print(f"\n=== Final Results ===")
        for i, f in enumerate(final_top):
            print(f"  #{i+1} {f['name']} -- score: {f['composite_score']}, "
                  f"IC_IR: {f['metrics'].get('ic_ir', '?')}")
