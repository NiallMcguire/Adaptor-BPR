"""Prepare (but never submit) Slurm jobs for the mixed-modality BPR matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path


AUDIO_BPR_DIR = Path(__file__).resolve().parent
EXPERIMENTS = (
    ("task_only", ("task",)),
    ("audio_only", ("audio",)),
    ("visual_only", ("visual",)),
    ("audio_visual", ("audio", "visual")),
    ("audio_task", ("audio", "task")),
    ("visual_task", ("visual", "task")),
    ("audio_visual_task", ("audio", "visual", "task")),
)


def build_plan(audio_data: Path, visual_data: Path, task_data: Path) -> list[dict]:
    sources = {"audio": str(audio_data.resolve()), "visual": str(visual_data.resolve()),
               "task": str(task_data.resolve())}
    plan = []
    for name, source_names in EXPERIMENTS:
        plan.append({"name": name, "sources": list(source_names),
                     "data_paths": [sources[source] for source in source_names]})
    return plan


def _controller_command(experiment: dict, output_path: Path, python_executable: str,
                        seed: int, epochs: int, batch_size: int, learning_rate: float,
                        use_pretrained_text: bool, use_lora: bool,
                        pooling_strategy: str) -> list[str]:
    command = [python_executable, str(AUDIO_BPR_DIR / "controller.py")]
    paths = experiment["data_paths"]
    if len(paths) == 1:
        command.extend(["--data_path", paths[0]])
    else:
        command.append("--data_paths")
        command.extend(paths)
    command.extend([
        "--pooling_strategy", pooling_strategy,
        "--encoder_type", "dual",
        "--query_type", "eeg",
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate),
        "--epochs", str(epochs),
        "--patience", "10",
        "--seed", str(seed),
        "--training_masking_level", "90",
        "--enable_multi_masking_validation",
        "--validation_masking_levels", "0", "25", "50", "75", "90", "100",
        "--primary_masking_level", "90",
        "--output_dir", str(output_path),
    ])
    if len(paths) > 1:
        command.append("--balance_sources")
    if use_pretrained_text:
        command.append("--use_pretrained_text")
    if not use_lora:
        command.append("--no_lora")
    return command


def _slurm_script(experiment: dict, command: list[str], output_dir: Path, partition: str,
                  time_limit: str, memory: str, cpus: int, gpu_request: str) -> str:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --job-name=bpr-{experiment['name']}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --gres={gpu_request}",
        f"#SBATCH --output={shlex.quote(str(log_dir / (experiment['name'] + '-%j.out')))}",
        f"#SBATCH --error={shlex.quote(str(log_dir / (experiment['name'] + '-%j.err')))}",
        "set -euo pipefail",
        f"cd {shlex.quote(str(AUDIO_BPR_DIR))}",
        "export PYTHONUNBUFFERED=1",
        f"{command_text}",
        "",
    ])


def prepare_jobs(audio_data: Path, visual_data: Path, task_data: Path, output_dir: Path,
                 partition: str = "gpu", time_limit: str = "12:00:00", memory: str = "64G",
                 cpus: int = 8, gpu_request: str = "gpu:1", python_executable: str = sys.executable,
                 seed: int = 42, epochs: int = 200, batch_size: int = 32,
                 learning_rate: float = 1e-5, use_pretrained_text: bool = True,
                 use_lora: bool = False, pooling_strategy: str = "multi", validate_inputs: bool = False,
                 overwrite: bool = False) -> dict:
    inputs = {"audio": Path(audio_data), "visual": Path(visual_data), "task": Path(task_data)}
    if validate_inputs:
        missing = [f"{name}: {path}" for name, path in inputs.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing data files:\n" + "\n".join(missing))

    output_dir = Path(output_dir).resolve()
    jobs_dir = output_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(inputs["audio"], inputs["visual"], inputs["task"])
    plan_path = output_dir / "experiment_matrix.json"
    if not overwrite and plan_path.exists():
        raise FileExistsError(f"{plan_path} exists; pass --overwrite to regenerate prepared jobs")

    for experiment in plan:
        experiment_output = output_dir / "results" / experiment["name"]
        command = _controller_command(experiment, experiment_output, python_executable, seed,
                                      epochs, batch_size, learning_rate, use_pretrained_text, use_lora,
                                      pooling_strategy)
        script_path = jobs_dir / f"{experiment['name']}.sbatch"
        script_path.write_text(_slurm_script(experiment, command, output_dir, partition,
                                             time_limit, memory, cpus, gpu_request), encoding="utf-8")
        script_path.chmod(0o755)
        experiment["job_script"] = str(script_path)
        experiment["output_dir"] = str(experiment_output)
        experiment["balance_sources"] = len(experiment["sources"]) > 1

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": False,
        "split_policy": "deterministic, source-stratified pair-level 80/10/10 (seed shared across conditions)",
        "training_policy": "equal-probability source sampling for mixed-source conditions",
        "text_encoder_policy": "pretrained ColBERT frozen (LoRA disabled)" if use_pretrained_text and not use_lora
        else "pretrained ColBERT with LoRA" if use_pretrained_text else "text encoder trained from scratch",
        "evaluation_policy": "each source's held-out queries ranked against only that source's held-out candidate passages",
        "unaligned_task_policy": "P001/P002 task EEG uses the configured zero-offset assumption and is flagged unverified in the pair metadata",
        "resources": {"partition": partition, "time": time_limit, "memory": memory,
                      "cpus_per_task": cpus, "gres": gpu_request},
        "matrix": plan,
    }
    plan_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    readme = output_dir / "PREPARED_NOT_SUBMITTED.txt"
    readme.write_text(
        "Seven Slurm job files have been generated under jobs/. No jobs were submitted.\n"
        "Review experiment_matrix.json and the data paths/resources before manually submitting any job.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-data", type=Path, required=True)
    parser.add_argument("--visual-data", type=Path, required=True)
    parser.add_argument("--task-data", type=Path,
                        default=AUDIO_BPR_DIR.parent / "dataset" / "authentic_query_task_pairs.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--gpu-request", default="gpu:1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--pooling-strategy", choices=("cls", "mean", "max", "multi"), default="multi")
    parser.add_argument("--no-pretrained-text", action="store_true")
    parser.add_argument("--use-lora", action="store_true",
                        help="Enable LoRA adapters; disabled by default for cluster compatibility")
    parser.add_argument("--validate-inputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = prepare_jobs(
        args.audio_data, args.visual_data, args.task_data, args.output_dir,
        partition=args.partition, time_limit=args.time, memory=args.memory,
        cpus=args.cpus, gpu_request=args.gpu_request, python_executable=args.python,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, use_pretrained_text=not args.no_pretrained_text,
        use_lora=args.use_lora, pooling_strategy=args.pooling_strategy, validate_inputs=args.validate_inputs,
        overwrite=args.overwrite,
    )
    print(f"Prepared {len(report['matrix'])} Slurm job scripts; no jobs submitted.")
    print(f"Plan: {Path(args.output_dir).resolve() / 'experiment_matrix.json'}")


if __name__ == "__main__":
    main()