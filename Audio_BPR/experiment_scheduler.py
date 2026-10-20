"""Prepare (but never submit) Slurm jobs for the mixed-modality BPR matrix."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
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
                  time_limit: str, memory: str, cpus: int, gpu_request: str,
                  array_spec: str = None) -> str:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    directives = [
        "#!/bin/bash",
        f"#SBATCH --job-name=bpr-{experiment['name']}",
    ]
    if array_spec is not None:
        directives.append(f"#SBATCH --array={array_spec}")
    directives.extend([
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
    return "\n".join(directives)


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


def prepare_subject_adapter_job(audio_data: Path, visual_data: Path, output_dir: Path,
                                partition: str = 'gpu', time_limit: str = '12:00:00', memory: str = '64G',
                                cpus: int = 8, gpu_request: str = 'gpu:1', python_executable: str = sys.executable,
                                seed: int = 1, epochs: int = 200, batch_size: int = 32,
                                learning_rate: float = 1e-5, pooling_strategy: str = 'cls',
                                ranks=(4, 8, 16, 32), alpha_ratios=(0.25, 1, 2, 3),
                                placements=('input_only', 'eeg_backbone', 'full_eeg'),
                                warmup_epochs: int = 10, validate_inputs: bool = False,
                                overwrite: bool = False) -> dict:
    """Prepare one sequential full-search job; never call sbatch."""
    from subject_adapter_experiment import parse_args, search_trials

    if validate_inputs:
        for path in (audio_data, visual_data):
            if not Path(path).is_file():
                raise FileNotFoundError(f'Missing data file: {path}')
    output_dir = Path(output_dir).resolve()
    plan_path = output_dir / 'experiment_matrix.json'
    if not overwrite and plan_path.exists():
        raise FileExistsError(f'{plan_path} exists; pass --overwrite to regenerate prepared jobs')
    experiment_output = output_dir / 'results' / 'subject_adapter_search'
    runner_args = [
        '--audio_path', str(Path(audio_data).resolve()), '--visual_path', str(Path(visual_data).resolve()),
        '--output_dir', str(experiment_output), '--seed', str(seed),
        '--epochs', str(epochs), '--batch_size', str(batch_size), '--lr', str(learning_rate),
        '--pooling_strategy', pooling_strategy, '--warmup_epochs', str(warmup_epochs), '--balance_sources',
        '--ranks', *map(str, ranks), '--alpha_ratios', *map(str, alpha_ratios),
        '--placements', *placements,
    ]
    args = parse_args(runner_args)
    trials = search_trials(args.ranks, args.alpha_ratios, args.placements)
    command = [python_executable, str(AUDIO_BPR_DIR / 'subject_adapter_experiment.py'), *runner_args]
    jobs_dir = output_dir / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    script_path = jobs_dir / 'subject_adapter_search.sbatch'
    script_path.write_text(_slurm_script({'name': 'subject-adapter-search'}, command, output_dir,
                                        partition, time_limit, memory, cpus, gpu_request), encoding='utf-8')
    script_path.chmod(0o755)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'submitted': False,
        'experiment_type': 'seen_subject_adapter_placement_search',
        'seed': seed, 'placements': args.placements, 'total_fits': len(trials), 'matrix': trials,
        'job_script': str(script_path), 'command': command, 'output_dir': str(experiment_output),
        'split_policy': 'seen subjects only; fixed per-dataset/per-subject split seed 42',
        'text_encoder_policy': 'frozen BERT; no text LoRA',
        'evaluation_policy': 'one combined index; baseline plus validation-selected configuration per placement',
        'resources': {'partition': partition, 'time': time_limit, 'memory': memory,
                      'cpus_per_task': cpus, 'gres': gpu_request},
    }
    plan_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'PREPARED_NOT_SUBMITTED.txt').write_text(
        f'One Slurm script prepares {len(trials)} sequential training fits using one seed. No jobs were submitted.\n'
        'Review data paths, walltime, memory, and checkpoint disk requirements before manual submission.\n',
        encoding='utf-8')
    return report


def prepare_subject_adapter_array(audio_data: Path, visual_data: Path, output_dir: Path,
                                  partition: str = 'gpu', time_limit: str = '2-00:01:00',
                                  memory: str = '128G', cpus: int = 8, gpu_request: str = 'gpu:1',
                                  python_executable: str = sys.executable, seed: int = 1,
                                  epochs: int = 200, batch_size: int = 32,
                                  learning_rate: float = 1e-5, pooling_strategy: str = 'cls',
                                  ranks=(4, 8, 16, 32), alpha_ratios=(0.25, 1, 2, 3),
                                  placements=('input_only', 'eeg_backbone', 'full_eeg'),
                                  warmup_epochs: int = 10, array_concurrency: int = 2,
                                  validate_inputs: bool = False, overwrite: bool = False) -> dict:
    """Prepare independent fits and a dependent finalizer; submission is separate."""
    from subject_adapter_experiment import parse_args, search_trials

    if validate_inputs:
        missing = [str(path) for path in (audio_data, visual_data) if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError('Missing data files:\n' + '\n'.join(missing))
    if array_concurrency < 1:
        raise ValueError('array_concurrency must be positive')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / 'experiment_matrix.json'
    if not overwrite and (plan_path.exists() or any(output_dir.iterdir())):
        raise FileExistsError(f'{output_dir} is not empty; choose a new output directory or pass overwrite=True')

    experiment_output = output_dir / 'results'
    common_args = [
        '--audio_path', str(Path(audio_data).resolve()), '--visual_path', str(Path(visual_data).resolve()),
        '--output_dir', str(experiment_output), '--seed', str(seed),
        '--epochs', str(epochs), '--batch_size', str(batch_size), '--lr', str(learning_rate),
        '--pooling_strategy', pooling_strategy, '--warmup_epochs', str(warmup_epochs), '--balance_sources',
        '--ranks', *map(str, ranks), '--alpha_ratios', *map(str, alpha_ratios),
        '--placements', *placements,
    ]
    parse_args([*common_args, '--array-task'])
    trials = search_trials(ranks, alpha_ratios, placements)
    array_command = [python_executable, str(AUDIO_BPR_DIR / 'subject_adapter_experiment.py'),
                     *common_args, '--array-task']
    finalizer_command = [python_executable, str(AUDIO_BPR_DIR / 'subject_adapter_experiment.py'),
                         *common_args, '--finalize']
    jobs_dir = output_dir / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    array_script = jobs_dir / 'subject_adapter_array.sbatch'
    array_script.write_text(_slurm_script(
        {'name': 'subject-adapter-array'}, array_command, output_dir, partition, time_limit,
        memory, cpus, gpu_request, array_spec=f"0-{len(trials) - 1}%{array_concurrency}"), encoding='utf-8')
    array_script.chmod(0o755)
    finalizer_script = jobs_dir / 'subject_adapter_finalize.sbatch'
    finalizer_script.write_text(_slurm_script(
        {'name': 'subject-adapter-finalize'}, finalizer_command, output_dir, partition,
        time_limit, memory, cpus, gpu_request), encoding='utf-8')
    finalizer_script.chmod(0o755)
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'submitted': False,
        'experiment_type': 'seen_subject_adapter_independent_array', 'seed': seed,
        'array_concurrency': array_concurrency, 'placements': list(placements),
        'total_fits': len(trials), 'matrix': trials, 'array_script': str(array_script),
        'finalizer_script': str(finalizer_script), 'array_command': array_command,
        'finalizer_command': finalizer_command, 'output_dir': str(experiment_output),
        'split_policy': 'seen subjects only; fixed per-dataset/per-subject split seed 42',
        'selection_policy': 'best validation checkpoint per fit; finalizer selects validation-MRR winners',
        'text_encoder_policy': 'frozen BERT; no text LoRA',
        'resources': {'partition': partition, 'time': time_limit, 'memory': memory,
                      'cpus_per_task': cpus, 'gres': gpu_request},
    }
    write_json = json.dumps(report, indent=2) + '\n'
    plan_path.write_text(write_json, encoding='utf-8')
    (output_dir / 'SUBMISSION_STATUS.txt').write_text(
        f"Prepared {len(trials)} independent array fits (throttle {array_concurrency}) and a finalizer. "
        "No jobs submitted yet.\n", encoding='utf-8')
    return report


def submit_subject_adapter_array(output_dir: Path) -> dict:
    """Submit prepared array and after-success finalizer; record Slurm job IDs."""
    output_dir = Path(output_dir).resolve()
    plan_path = output_dir / 'experiment_matrix.json'
    report = json.loads(plan_path.read_text(encoding='utf-8'))
    if report.get('submitted'):
        raise ValueError('This array submission manifest is already marked submitted')
    array_script = Path(report['array_script'])
    finalizer_script = Path(report['finalizer_script'])
    array_output = subprocess.run(['sbatch', '--parsable', str(array_script)],
                                  check=True, capture_output=True, text=True).stdout.strip()
    array_job_id = array_output.split(';', 1)[0]
    try:
        finalizer_output = subprocess.run(
            ['sbatch', '--parsable', f'--dependency=afterok:{array_job_id}', str(finalizer_script)],
            check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        subprocess.run(['scancel', array_job_id], check=False, capture_output=True, text=True)
        raise
    finalizer_job_id = finalizer_output.split(';', 1)[0]
    report.update({'submitted': True, 'array_job_id': array_job_id,
                   'finalizer_job_id': finalizer_job_id,
                   'dependency': f'afterok:{array_job_id}',
                   'submitted_utc': datetime.now(timezone.utc).isoformat(),
                   'array_state': 'SUBMITTED', 'finalizer_state': 'DEPENDENCY_PENDING'})
    plan_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'SUBMISSION_STATUS.txt').write_text(
        f"Submitted array {array_job_id} (0-{report['total_fits'] - 1}, throttle "
        f"{report['array_concurrency']}) and finalizer {finalizer_job_id} "
        f"(dependency {report['dependency']}).\nMonitor: squeue -j {array_job_id},{finalizer_job_id}\n"
        f"Results: {report['output_dir']}\n", encoding='utf-8')
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--pooling-strategy", choices=("cls", "mean", "max", "multi"), default=None)
    parser.add_argument('--subject-adapter-search', action='store_true',
                        help='Prepare the combined-data three-placement, one-seed subject-adapter search')
    parser.add_argument('--subject-adapter-array', action='store_true',
                        help='Prepare and submit independent config jobs plus a dependent finalizer')
    parser.add_argument('--ranks', nargs='+', type=int, default=[4, 8, 16, 32])
    parser.add_argument('--alpha-ratios', nargs='+', type=float, default=[0.25, 1, 2, 3])
    parser.add_argument('--placements', nargs='+', choices=('input_only', 'eeg_backbone', 'full_eeg'),
                        default=['input_only', 'eeg_backbone', 'full_eeg'])
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--array-concurrency', type=int, default=2)
    parser.add_argument("--no-pretrained-text", action="store_true")
    parser.add_argument("--use-lora", action="store_true",
                        help="Legacy text-encoder LoRA only; use --subject-adapter-search for subject EEG adapters")
    parser.add_argument("--validate-inputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.subject_adapter_array:
        if args.use_lora or args.no_pretrained_text:
            parser.error('Subject-adapter array requires frozen pretrained text without text LoRA')
        if args.validate_inputs:
            missing = [str(path) for path in (args.audio_data, args.visual_data) if not path.is_file()]
            if missing:
                parser.error('Missing data files: ' + ', '.join(missing))
        report = prepare_subject_adapter_array(
            args.audio_data, args.visual_data, args.output_dir, partition=args.partition,
            time_limit=args.time, memory=args.memory, cpus=args.cpus, gpu_request=args.gpu_request,
            python_executable=args.python, seed=args.seed if args.seed is not None else 1,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            pooling_strategy=args.pooling_strategy or 'cls', ranks=args.ranks,
            alpha_ratios=args.alpha_ratios, placements=args.placements,
            warmup_epochs=args.warmup_epochs, array_concurrency=args.array_concurrency,
            validate_inputs=args.validate_inputs, overwrite=args.overwrite)
        report = submit_subject_adapter_array(args.output_dir)
        print(f"Submitted array {report['array_job_id']} and dependent finalizer {report['finalizer_job_id']}.")
        return
    if args.subject_adapter_search:
        if args.use_lora or args.no_pretrained_text:
            parser.error('Subject-adapter search requires frozen pretrained text without text LoRA')
        report = prepare_subject_adapter_job(
            args.audio_data, args.visual_data, args.output_dir, partition=args.partition,
            time_limit=args.time, memory=args.memory, cpus=args.cpus, gpu_request=args.gpu_request,
            python_executable=args.python, seed=args.seed if args.seed is not None else 1,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            pooling_strategy=args.pooling_strategy or 'cls', ranks=args.ranks, alpha_ratios=args.alpha_ratios,
            placements=args.placements, warmup_epochs=args.warmup_epochs,
            validate_inputs=args.validate_inputs, overwrite=args.overwrite,
        )
        print(f"Prepared one subject-adapter search script for {report['total_fits']} fits; no jobs submitted.")
        print(f"Plan: {Path(args.output_dir).resolve() / 'experiment_matrix.json'}")
        return
    report = prepare_jobs(
        args.audio_data, args.visual_data, args.task_data, args.output_dir,
        partition=args.partition, time_limit=args.time, memory=args.memory,
        cpus=args.cpus, gpu_request=args.gpu_request, python_executable=args.python,
        seed=args.seed if args.seed is not None else 42, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, use_pretrained_text=not args.no_pretrained_text,
        use_lora=args.use_lora, pooling_strategy=args.pooling_strategy or 'multi', validate_inputs=args.validate_inputs,
        overwrite=args.overwrite,
    )
    print(f"Prepared {len(report['matrix'])} Slurm job scripts; no jobs submitted.")
    print(f"Plan: {Path(args.output_dir).resolve() / 'experiment_matrix.json'}")


if __name__ == "__main__":
    main()