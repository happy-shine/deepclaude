"""Evolution orchestrator — launches sequential Claude Code instances with resume support."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from deepclaude import registry


CLAUDE_CMD = "claude"
STATE_FILE = "deepclaude_state.json"
MAX_RESUME_RETRIES = 3
RESUME_PROMPT = "你的session因网络问题中断了，请从中断处继续你的研究工作。"


@dataclass
class Config:
    max_rounds: int = 10
    top_k: int = 5
    max_iterations: int = 20
    project_root: str = "."
    data_dir: str = "/Users/shine/trader-data"
    composite_weights: dict = field(default_factory=lambda: {
        "ic_ir": 0.25, "long_sharpe": 0.25, "monotonicity": 0.15,
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
                         f"Long_Sharpe={metrics.get('long_sharpe', '?')}, "
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
        self._claude_session_id: str | None = None

    # -- State persistence --------------------------------------------------

    def _state_path(self) -> Path:
        return Path(self.config.project_root) / STATE_FILE

    def _save_state(self, run_ts: str, completed_rounds: int,
                    session_id: str | None = None,
                    claude_session_id: str | None = None,
                    status: str = "running"):
        state = {
            "run_ts": run_ts,
            "completed_rounds": completed_rounds,
            "max_rounds": self.config.max_rounds,
            "current_session": {
                "session_id": session_id,
                "claude_session_id": claude_session_id,
                "status": status,
            },
        }
        tmp = self._state_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.rename(str(tmp), str(self._state_path()))

    def _load_state(self) -> dict | None:
        path = self._state_path()
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- Workspace ----------------------------------------------------------

    def _make_workspace(self, session_id: str) -> str:
        ws = Path(self.config.project_root) / "workspace" / session_id
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "scratch").mkdir(exist_ok=True)
        return str(ws)

    # -- Claude CLI ---------------------------------------------------------

    def _launch_claude(self, prompt: str, workspace: str, session_id: str,
                       resume_id: str | None = None) -> subprocess.Popen:
        """Launch a Claude Code CLI process. If resume_id is given, resume that session."""
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
            "--model", "opus",
        ]

        if resume_id:
            cmd.extend(["--resume", resume_id, "-p", RESUME_PROMPT])
        else:
            final_prompt = prompt.replace("{output_dir}", workspace)
            cmd.extend(["--add-dir", workspace, "-p", final_prompt])

        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=workspace,
        )
        return proc

    def _stream_output(self, proc: subprocess.Popen, session_id: str):
        """Read stream-json output, capture Claude session ID, print status, save to logs/."""
        logs_dir = Path(self.config.project_root) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(logs_dir / f"{session_id}.jsonl", "a", encoding="utf-8")
        self._claude_session_id = None

        def _reader(pipe, label):
            for line in pipe:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                log_file.write(line + "\n")
                log_file.flush()
                try:
                    event = json.loads(line)
                    event_type = event.get("type", "")

                    # Capture Claude's internal session ID for resume
                    if "session_id" in event and not self._claude_session_id:
                        self._claude_session_id = event["session_id"]

                    if event_type == "assistant":
                        msg = event.get("message", "")
                        if isinstance(msg, dict):
                            msg = msg.get("content", str(msg))
                        print(f"[{session_id}] {msg}")
                    elif event_type == "tool_use":
                        name = event.get("name", event.get("tool", ""))
                        inp = event.get("input", "")
                        if isinstance(inp, dict):
                            inp = inp.get("command", inp.get("content", str(inp)))
                        print(f"[{session_id}] tool: {name} | {inp}")
                    elif event_type == "tool_result":
                        output = event.get("output", "")
                        print(f"[{session_id}] result: {output}")
                    elif event_type == "result":
                        if "session_id" in event:
                            self._claude_session_id = event["session_id"]
                        print(f"[{session_id}] Done")
                except json.JSONDecodeError:
                    print(f"[{session_id}] {label}: {line}")

        t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        t_out.start()
        t_err.start()
        return t_out, t_err, log_file

    # -- Single round -------------------------------------------------------

    def _run_round(self, round_num: int, run_ts: str, prompt: str,
                   resume_id: str | None = None) -> bool:
        """Run a single round. Returns True on success."""
        session_id = f"{run_ts}_r{round_num:03d}"
        workspace = self._make_workspace(session_id)

        self._save_state(run_ts, round_num - 1, session_id, status="running")

        proc = self._launch_claude(prompt, workspace, session_id, resume_id)
        t_out, t_err, log_file = self._stream_output(proc, session_id)
        suffix = " (resumed)" if resume_id else ""
        print(f"[{session_id}] Launched{suffix}")

        proc.wait()
        for t in (t_out, t_err):
            t.join(timeout=5)
        log_file.close()

        exit_code = proc.returncode
        print(f"[{session_id}] Exited with code {exit_code}")

        # Auto-resume on non-zero exit if we captured a Claude session ID
        if exit_code != 0 and self._claude_session_id:
            for retry in range(1, MAX_RESUME_RETRIES + 1):
                print(f"[{session_id}] Auto-resuming (attempt {retry}/{MAX_RESUME_RETRIES})...")
                self._save_state(run_ts, round_num - 1, session_id,
                                 self._claude_session_id, "resuming")

                proc = self._launch_claude("", workspace, session_id,
                                           self._claude_session_id)
                t_out, t_err, log_file = self._stream_output(proc, session_id)
                proc.wait()
                for t in (t_out, t_err):
                    t.join(timeout=5)
                log_file.close()

                exit_code = proc.returncode
                print(f"[{session_id}] Resume exited with code {exit_code}")
                if exit_code == 0:
                    break
            else:
                print(f"[{session_id}] Failed after {MAX_RESUME_RETRIES} resume attempts")
                self._save_state(run_ts, round_num - 1, session_id,
                                 self._claude_session_id, "failed")
                return False

        self._save_state(run_ts, round_num, session_id,
                         self._claude_session_id, "completed")
        return True

    # -- Main loop ----------------------------------------------------------

    def run(self, resume: bool = False):
        """Execute the evolution loop."""
        if resume:
            state = self._load_state()
            if not state:
                print("No previous run to resume.")
                return
            run_ts = state["run_ts"]
            start_round = state["completed_rounds"] + 1
            max_rounds = state["max_rounds"]

            # If last session was interrupted mid-round, resume it first
            current = state.get("current_session", {})
            if (current.get("status") in ("running", "resuming")
                    and current.get("claude_session_id")):
                print(f"Resuming interrupted session: {current['session_id']}")
                top_k = registry.get_top_k(k=self.config.top_k)
                prompt = build_prompt(top_k, self.config)
                success = self._run_round(start_round, run_ts, prompt,
                                          current["claude_session_id"])
                if success:
                    start_round += 1

            print(f"=== DeepClaude Orchestrator (Resumed) ===")
            print(f"Continuing from round {start_round}/{max_rounds}")
        else:
            run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            start_round = 1
            max_rounds = self.config.max_rounds
            print(f"=== DeepClaude Orchestrator ===")

        print(f"Config: {max_rounds} rounds, top {self.config.top_k} selection")
        print()

        for round_num in range(start_round, max_rounds + 1):
            print(f"--- Round {round_num}/{max_rounds} ---")

            top_k = registry.get_top_k(k=self.config.top_k)
            prompt = build_prompt(top_k, self.config)

            success = self._run_round(round_num, run_ts, prompt)
            if not success:
                print(f"Round {round_num} failed. Continuing to next round.")

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

        # Clean up state file on successful completion
        if self._state_path().exists():
            self._state_path().unlink()
