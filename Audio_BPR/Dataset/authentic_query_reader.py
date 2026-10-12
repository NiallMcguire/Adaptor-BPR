"""Build EEG-question to gold-answer pairs from the extracted task dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
DEFAULT_OUTPUT = DEFAULT_DATASET_DIR / "authentic_query_task_pairs.npy"


def _key(participant: str, session: str, part: object, trial: object, question_id: str) -> tuple[str, ...]:
    return participant, session, str(int(float(part))), str(int(float(trial))), question_id


def _normalize_word(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _is_true(value: object) -> bool:
    return value is True or str(value).strip().casefold() in {"true", "1", "yes"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _matched_word_index(row: dict, question_tokens: list[str]) -> int | None:
    index = row.get("word_index")
    if index is None or str(index).strip() == "" or _is_true(row.get("word_index_ambiguous", False)):
        return None
    try:
        numeric_index = float(index)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_index) or not numeric_index.is_integer():
        return None
    index = int(numeric_index)
    if not 0 <= index < len(question_tokens):
        return None
    if _normalize_word(row.get("word", "")) != _normalize_word(question_tokens[index]):
        return None
    return index


def _make_pair(record: dict[str, str], eeg_epochs: list[np.ndarray], source_path: Path,
               alignment_status: str, offset_seconds: float | None) -> dict:
    response = record["ground_truth_response"].strip()
    pair = {
        "query_text": record["question_text"].strip(),
        "query_eeg": np.asarray(eeg_epochs, dtype=np.float32),
        "query_words": record["question_text"].split(),
        "doc_text": response,
        "doc_eeg": np.zeros((0, 0, 0), dtype=np.float32),
        "doc_words": response.split(),
        "full_sentence_text": response,
        "full_sentence_words": response.split(),
        "query_start_idx": 0,
        "query_end_idx": 0,
        "participant_id": record["participant_id"],
        "session_id": record["session_id"],
        "part": int(record["part"]),
        "trial": int(record["trial"]),
        "sentence_id": int(record["trial"]),
        "question_id": record["question_id"],
        "condition": record["condition"],
        "answer_status": record["answer_status"],
        "answer_provenance": record["answer_provenance"],
        "dataset_source": "authentic_query",
        "supports_runtime_masking": False,
        "alignment_status": alignment_status,
        "unaligned_clock_offset_seconds": offset_seconds,
        "source_identity_group": "P001_P002_byte_identical" if record["participant_id"] in {"P001", "P002"} else "",
        "source_eeg_file": str(source_path),
    }
    return pair


def build_task_pairs(dataset_dir: Path = DEFAULT_DATASET_DIR, output_path: Path = DEFAULT_OUTPUT,
                     unaligned_clock_offset_seconds: float = 0.0, min_fixations: int = 1) -> dict:
    """Build pairs from aligned epochs and explicitly approximate P001/P002 timing.

    For P001/P002, PsychoPy elapsed seconds are mapped to EEG-relative seconds as
    ``EEG time = PsychoPy time + unaligned_clock_offset_seconds``. The default
    zero offset is an unverified assumption and is preserved in each pair.
    """
    dataset_dir = Path(dataset_dir).resolve()
    output_path = Path(output_path).resolve()
    trial_path = dataset_dir / "trial_response_ground_truth.csv"
    if not trial_path.is_file():
        raise FileNotFoundError(f"Missing task trial table: {trial_path}")

    records = {}
    for row in _read_rows(trial_path):
        key = (row["participant_id"], row["session_id"], row["part"], row["trial"], row["question_id"])
        records[key] = row

    participant_dirs = sorted(path for path in dataset_dir.glob("P*") if path.is_dir())
    pairs = []
    counters = defaultdict(int)
    participants_seen = set()
    eeg_read_cache = {}

    for participant_dir in participant_dirs:
        participant = participant_dir.name
        aligned_files = sorted(participant_dir.glob("ses-*/*_reading-epo.fif"))
        for epochs_path in aligned_files:
            session = epochs_path.parent.name
            epochs = mne.read_epochs(epochs_path, preload=True, verbose="ERROR")
            metadata = epochs.metadata
            if metadata is None:
                counters["aligned_sessions_missing_metadata"] += 1
                continue
            data = epochs.get_data(copy=False)
            grouped = defaultdict(list)
            for event_index, row in enumerate(metadata.to_dict("records")):
                key = _key(participant, session, row.get("part", 0), row.get("trial", 0), str(row.get("question_id", "")))
                grouped[key].append((row, data[event_index]))

            for key, events in grouped.items():
                record = records.get(key)
                if record is None:
                    counters["aligned_trial_missing_answer_row"] += 1
                    continue
                tokens = record["question_text"].split()
                matched = []
                for row, eeg in sorted(events, key=lambda item: float(item[0].get("onset", 0.0))):
                    if _matched_word_index(row, tokens) is not None:
                        matched.append(eeg)
                counters["aligned_trials_seen"] += 1
                counters["aligned_fixations_matched"] += len(matched)
                if len(matched) < min_fixations:
                    counters["aligned_trials_without_verified_question_fixations"] += 1
                    continue
                pairs.append(_make_pair(record, matched, epochs_path, "aligned_word_index", None))
                participants_seen.add(participant)
            del epochs

        unaligned_dir = participant_dir / "unaligned_task_export"
        if not unaligned_dir.is_dir():
            continue
        raw_path = unaligned_dir / f"{participant}_unaligned_raw.fif"
        fixation_path = unaligned_dir / f"{participant}_reading_fixations.csv"
        if not raw_path.is_file() or not fixation_path.is_file():
            counters["unaligned_participants_missing_raw_or_fixations"] += 1
            continue

        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
        fixation_rows = _read_rows(fixation_path)
        grouped_fixations = defaultdict(list)
        for row in fixation_rows:
            key = _key(participant, "unaligned_task_export", row.get("part", 0),
                       row.get("trial", 0), str(row.get("question_id", "")))
            grouped_fixations[key].append(row)

        sfreq = float(raw.info["sfreq"])
        epoch_tmin = -0.2
        epoch_samples = int(round(1.0 * sfreq)) + 1
        for key, events in grouped_fixations.items():
            record = records.get(key)
            if record is None:
                counters["unaligned_trial_missing_answer_row"] += 1
                continue
            tokens = record["question_text"].split()
            selected = []
            for row in sorted(events, key=lambda item: float(item.get("onset", 0.0))):
                if _matched_word_index(row, tokens) is None:
                    continue
                psycho_onset = float(row["onset"])
                eeg_start_seconds = psycho_onset + unaligned_clock_offset_seconds + epoch_tmin
                start = int(round(eeg_start_seconds * sfreq))
                stop = start + epoch_samples
                if start < 0 or stop > raw.n_times:
                    counters["unaligned_fixations_outside_eeg"] += 1
                    continue
                selected.append(raw.get_data(start=start, stop=stop).T)

            counters["unaligned_trials_seen"] += 1
            counters["unaligned_fixations_matched"] += len(selected)
            if len(selected) < min_fixations:
                counters["unaligned_trials_without_usable_question_fixations"] += 1
                continue
            pairs.append(_make_pair(record, selected, raw_path,
                                    "unverified_psychopy_to_eeg_offset", unaligned_clock_offset_seconds))
            participants_seen.add(participant)
        del raw

    if not pairs:
        raise ValueError("No usable query-answer EEG pairs were found")

    metadata = {
        "dataset_name": "authentic_query",
        "version": "authentic_query_word_fixation_pairs_v1",
        "supports_runtime_masking": False,
        "query_definition": "question_text with word-index-verified reading-fixation EEG",
        "document_definition": "ground_truth_response from trial_response_ground_truth.csv",
        "split_policy": "pair-level deterministic 80/10/10 applied by the training dataloader",
        "unaligned_participant_policy": "P001/P002 included with explicit unverified PsychoPy-to-EEG offset assumption",
        "unaligned_clock_offset_seconds": unaligned_clock_offset_seconds,
        "unaligned_duplicate_identity_warning": "P001 and P002 source EEG/CSV inputs are byte-identical; IDs are retained but are not established as independent participants.",
        "min_fixations": min_fixations,
        "participants_with_pairs": sorted(participants_seen),
        "pair_count": len(pairs),
        "counters": dict(counters),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, {"ict_pairs": pairs, "metadata": metadata}, allow_pickle=True)
    output_path.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unaligned-clock-offset-seconds", type=float, default=0.0,
                        help="Assumed EEG-relative offset for PsychoPy times in P001/P002; default 0 is unverified.")
    parser.add_argument("--min-fixations", type=int, default=1)
    args = parser.parse_args()
    if args.min_fixations < 1:
        parser.error("--min-fixations must be at least one")
    metadata = build_task_pairs(args.dataset_dir, args.output, args.unaligned_clock_offset_seconds,
                                args.min_fixations)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()