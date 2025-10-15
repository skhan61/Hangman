"""Run BiLSTM training in supervised and self-supervised modes via main.py.

This helper simply executes `main.py` twice with the exact flag sets you
requested—first in standard supervised mode with a large batch size, then with
the self-supervised contrastive options—and summarizes the win rate reported by
the Hangman evaluation callback. Both runs share a single epoch count but keep
independent batch-size controls. Run it inside the `orchestra` Conda
environment so the same dependencies as `main.py` are available.
"""

from __future__ import annotations

import argparse
import errno
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import pty
import select
import subprocess


RE_WIN_RATE = re.compile(
    r"Win rate:\s*([0-9.]+)%.*Avg tries remaining:\s*([0-9.]+)",
    re.IGNORECASE,
)


@dataclass
class RunConfig:
    """Parameters for one `main.py` invocation."""

    name: str
    args: Sequence[str]


@dataclass
class RunResult:
    """Outcome of executing a training command."""

    name: str
    returncode: int
    log: str
    win_rate: float | None
    avg_tries: float | None


def parse_win_rate(log: str) -> tuple[float | None, float | None]:
    """Extract the final win rate / average tries pair from training logs."""

    matches = RE_WIN_RATE.findall(log)
    if not matches:
        return None, None
    win, tries = matches[-1]
    try:
        return float(win), float(tries)
    except ValueError:
        return None, None


def stream_subprocess(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
    """Run a command with a pseudo-TTY so progress bars render correctly."""

    master_fd, slave_fd = pty.openpty()
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdin=None,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    log_chunks: list[str] = []
    with os.fdopen(master_fd, "rb", buffering=0) as master:
        while True:
            try:
                ready, _, _ = select.select([master], [], [], 0.1)
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise

            if master in ready:
                try:
                    data = master.read(1024)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break  # Slave closed; process likely finished
                    raise

                if not data:
                    if process.poll() is not None:
                        break
                    continue

                text = data.decode(errors="replace")
                sys.stdout.write(text)
                sys.stdout.flush()
                log_chunks.append(text)

            if process.poll() is not None and not ready:
                break

    returncode = process.wait()
    return returncode, "".join(log_chunks)


def run_training(
    config: RunConfig,
    python_executable: str,
    project_root: Path,
) -> RunResult:
    """Execute `main.py` with the provided arguments and collect summary stats."""

    cmd = [python_executable, "main.py", *config.args]
    print(f"\n=== Running {config.name} ===")
    print("Command:", " ".join(cmd))

    returncode, log = stream_subprocess(cmd, cwd=project_root)
    win_rate, avg_tries = parse_win_rate(log)

    if win_rate is None:
        print(
            f"[{config.name}] WARNING: Could not parse final win rate from logs. "
            "Check the output above for details."
        )
    else:
        print(
            f"[{config.name}] Final win rate: {win_rate:.2f}% | "
            f"Average tries remaining: {avg_tries:.2f}"
        )

    return RunResult(
        name=config.name,
        returncode=returncode,
        log=log,
        win_rate=win_rate,
        avg_tries=avg_tries,
    )


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    """Create the two training runs (supervised + self-supervised)."""

    supervised = RunConfig(
        name=f"Supervised (batch {args.supervised_batch_size})",
        args=[
            "--max-epochs",
            str(args.epochs),
            "--batch-size",
            str(args.supervised_batch_size),
        ],
    )

    self_supervised = RunConfig(
        name=f"Self-supervised (batch {args.contrastive_batch_size})",
        args=[
            "--max-epochs",
            str(args.epochs),
            "--batch-size",
            str(args.contrastive_batch_size),
            "--use-contrastive",
            "--lambda-contrast",
            str(args.lambda_contrast),
            "--num-embedding-layers",
            str(args.num_embedding_layers),
            "--embedding-regularizer",
            args.embedding_regularizer,
            "--regularizer-weight",
            str(args.regularizer_weight),
        ],
    )

    return [supervised, self_supervised]


def print_summary(results: Iterable[RunResult]) -> None:
    """Pretty-print the comparison once both runs finish."""

    print("\n=== Summary ===")
    print(f"{'Run':<28} | {'Win Rate':>9} | {'Avg Tries':>10} | {'Status':>8}")
    print("-" * 65)

    for res in results:
        win_display = f"{res.win_rate:6.2f}%" if res.win_rate is not None else "   N/A "
        tries_display = (
            f"{res.avg_tries:7.2f}" if res.avg_tries is not None else "    N/A"
        )
        status = "OK" if res.returncode == 0 else f"ERR({res.returncode})"
        print(f"{res.name:<28} | {win_display:<9} | {tries_display:<10} | {status:>8}")

    print("-" * 65)

    parsed = [r for r in results if r.win_rate is not None]
    if len(parsed) == 2:
        base, contrastive = parsed
        delta = contrastive.win_rate - base.win_rate  # type: ignore[arg-type]
        comparison = (
            "improves" if delta > 0 else "matches" if abs(delta) < 1e-6 else "trails"
        )
        print(
            f"Self-supervised run {comparison} the supervised baseline by "
            f"{abs(delta):.2f} percentage points."
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run `main.py` twice (supervised & self-supervised BiLSTM) and compare metrics."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to pass to both runs (--max-epochs).",
    )
    parser.add_argument(
        "--supervised-batch-size",
        type=int,
        default=4096,
        help="Batch size for the supervised run.",
    )
    parser.add_argument(
        "--contrastive-batch-size",
        type=int,
        default=256,
        help="Batch size for the self-supervised run.",
    )
    parser.add_argument(
        "--lambda-contrast",
        type=float,
        default=0.1,
        help="Contrastive loss weight used in the self-supervised run.",
    )
    parser.add_argument(
        "--num-embedding-layers",
        type=int,
        default=3,
        help="Number of BiLSTM layers used for embeddings in the self-supervised run.",
    )
    parser.add_argument(
        "--embedding-regularizer",
        choices=["lp", "center_invariant", "zero_mean"],
        default="lp",
        help="Embedding regularizer applied in the self-supervised run.",
    )
    parser.add_argument(
        "--regularizer-weight",
        type=float,
        default=0.01,
        help="Regularizer weight for the self-supervised run.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root to run the commands from.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter to use (defaults to the current interpreter).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.project_root.exists():
        print(f"Project root not found: {args.project_root}", file=sys.stderr)
        return 1

    configs = build_configs(args)
    results: list[RunResult] = []
    for cfg in configs:
        result = run_training(cfg, str(args.python_executable), args.project_root)
        results.append(result)
        if result.returncode != 0:
            print(
                f"[{cfg.name}] exited with status {result.returncode}. "
                "Stopping early.",
                file=sys.stderr,
            )
            break

    print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
