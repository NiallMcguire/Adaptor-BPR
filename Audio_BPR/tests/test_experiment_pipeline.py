import csv
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

AUDIO_BPR_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUDIO_BPR_DIR))

from Dataset.authentic_query_reader import build_task_pairs
from controller import create_dataloaders, create_source_test_loaders
from mv_dataloader import compute_global_eeg_dimensions, load_combined_datasets
from mv_models import create_model
from mv_models import SubjectLowRankLinear
from mv_training import (compute_contrastive_loss, test_model, subject_key,
                         evaluate_unified_index, compute_ranking_metrics)
from experiment_scheduler import (EXPERIMENTS, prepare_jobs, prepare_subject_adapter_job,
                                  prepare_subject_adapter_array, submit_subject_adapter_array)
from subject_adapter_experiment import parse_args, run_experiment, select_adapter, search_trials


class TinyTokenizer:
    mask_token_id = 3
    sep_token_id = 2

    def __len__(self):
        return 256

    def encode_plus(self, text, add_special_tokens=True, max_length=32, truncation=True,
                    padding=None, return_tensors="pt"):
        import hashlib

        words = str(text).split()
        ids = [1] if add_special_tokens else []
        ids.extend(4 + int(hashlib.sha1(word.encode()).hexdigest()[:6], 16) % 240 for word in words)
        if add_special_tokens:
            ids.append(self.sep_token_id)
        if truncation:
            ids = ids[:max_length]
            if add_special_tokens and ids[-1] != self.sep_token_id:
                ids[-1] = self.sep_token_id
        mask = [1] * len(ids)
        if padding == "max_length":
            pad = max(0, max_length - len(ids))
            ids.extend([0] * pad)
            mask.extend([0] * pad)
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }


def write_synthetic_pairs(path: Path, source: str, count: int, supports_masking: bool) -> Path:
    pairs = []
    for index in range(count):
        if supports_masking:
            question = f"queryword{index} context{index}"
            document = f"before{index} queryword{index} context{index} after{index}"
            words = document.split()
            query_start, query_end = 1, 3
        else:
            question = f"What fact {index}?"
            document = f"Reference answer number {index} includes enough words."
            words = document.split()
            query_start, query_end = 0, 0
        query_eeg = np.random.default_rng(index).normal(size=(2, 8, 2)).astype(np.float32)
        pairs.append({
            "query_text": question,
            "query_eeg": query_eeg,
            "query_words": question.split(),
            "doc_text": document,
            "doc_words": words,
            "full_sentence_text": document,
            "full_sentence_words": words,
            "query_start_idx": query_start,
            "query_end_idx": query_end,
            "participant_id": f"{source}_p{index % 5}",
            "sentence_id": index,
            "question_id": f"{source}_q{index}",
            "supports_runtime_masking": supports_masking,
            "dataset_source": source,
        })
    np.save(path, {"ict_pairs": pairs, "metadata": {
        "dataset_name": source,
        "supports_runtime_masking": supports_masking,
    }})
    return path


def make_trial_row(participant, session, question_id):
    return {
        "participant_id": participant,
        "session_id": session,
        "part": "1",
        "trial": "1",
        "question_id": question_id,
        "question_text": "What is capital?",
        "typed_text": "capital",
        "condition": "test",
        "chatbot_condition": "baseline",
        "response_seen": "edited response",
        "ground_truth_answer": "Paris",
        "ground_truth_response": "Paris is the capital of France.",
        "answer_status": "source_verified",
        "answer_provenance": "synthetic fixture",
        "answer_notes": "",
    }


class ExperimentPipelineSmokeTests(unittest.TestCase):
    def test_three_adapter_placements_preserve_shared_model_and_train_all_updates(self):
        inputs = torch.randn(2, 4, 8, 2)
        inputs[:, -1] = 0
        masks = torch.ones(2, 4)
        ids = torch.tensor([0, 1])
        counts = []
        for placement, expected_layers in (('input_only', 1), ('eeg_backbone', 8), ('full_eeg', 14)):
            baseline = create_model(hidden_dim=8, eeg_arch='transformer', device='cpu', use_lora=False,
                                    pooling_strategy='cls', global_eeg_dims=(4, 8, 2), use_pretrained_text=False)
            adapted = copy.deepcopy(baseline)
            adapted.enable_subject_adapters(3, 2, 4, placement)
            modules = [module for module in adapted.modules() if isinstance(module, SubjectLowRankLinear)]
            self.assertEqual(len(modules), expected_layers)
            self.assertFalse(any(isinstance(module, SubjectLowRankLinear) for module in adapted.text_encoder.modules()))
            counts.append(sum(module.adapter_a.numel() + module.adapter_b.numel() for module in modules))
            updates = [module.adapter_b.detach().clone() for module in modules]
            with torch.no_grad():
                for module in modules:
                    module.adapter_b.zero_()
                baseline.eval()
                adapted.eval()
                torch.testing.assert_close(adapted.encode_eeg(inputs, masks, ids),
                                           baseline.encode_eeg(inputs, masks), atol=1e-5, rtol=1e-4)
                for module, update in zip(modules, updates):
                    module.adapter_b.copy_(update)
                encoded = adapted.encode_eeg(inputs, masks, ids)
                changed = adapted.encode_eeg(inputs, masks, ids.flip(0))
                self.assertFalse(torch.allclose(encoded, changed))
            adapted.train()
            adapted.encode_eeg(inputs, masks, ids).square().sum().backward()
            for module in modules:
                self.assertGreater(module.adapter_a.grad[:2].abs().sum().item(), 0)
                self.assertGreater(module.adapter_b.grad[:2].abs().sum().item(), 0)
                self.assertEqual(module.adapter_a.grad[2].abs().sum().item(), 0)
                self.assertGreater(module.shared.weight.grad.abs().sum().item(), 0)
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])

    def test_search_defaults_cover_three_placements_with_exactly_one_seed(self):
        argv = ['--audio_path', 'alice.npy', '--visual_path', 'nieuwland.npy', '--output_dir', 'unused']
        args = parse_args(argv)
        self.assertEqual(args.seeds, [1])
        self.assertEqual(args.placements, ['input_only', 'eeg_backbone', 'full_eeg'])
        trials = search_trials(args.ranks, args.alpha_ratios, args.placements)
        self.assertEqual(len(trials), 49)
        self.assertEqual(len({trial['name'] for trial in trials}), 49)
        for placement in args.placements:
            candidates = [trial for trial in trials if trial.get('placement') == placement]
            self.assertEqual(len(candidates), 16)
            self.assertEqual({(trial['rank'], trial['alpha_over_rank']) for trial in candidates},
                             {(rank, ratio) for rank in args.ranks for ratio in args.alpha_ratios})
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                parse_args(argv + ['--seeds', '1', '2'])
            with self.assertRaises(SystemExit):
                parse_args(argv + ['--pooling_strategy', 'mean'])

    def test_subject_search_scheduler_prepares_all_configs_without_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch('subprocess.run') as submit:
                report = prepare_subject_adapter_job(root / 'alice.npy', root / 'nieuwland.npy', root / 'prepared')
            submit.assert_not_called()
            self.assertFalse(report['submitted'])
            self.assertEqual(report['total_fits'], 49)
            self.assertEqual(report['seed'], 1)
            script = Path(report['job_script']).read_text()
            self.assertIn('subject_adapter_experiment.py', script)
            self.assertIn('--seed 1', script)
            self.assertIn('--placements input_only eeg_backbone full_eeg', script)
            self.assertNotIn('controller.py', script)
            self.assertIn('No jobs were submitted', (root / 'prepared' / 'PREPARED_NOT_SUBMITTED.txt').read_text())

    def test_array_scheduler_submits_throttled_configs_and_dependent_finalizer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared_dir = root / 'prepared'
            report = prepare_subject_adapter_array(
                root / 'alice.npy', root / 'nieuwland.npy', prepared_dir,
                ranks=(2,), alpha_ratios=(1,), array_concurrency=2,
            )
            self.assertEqual(report['total_fits'], 4)
            array_script = Path(report['array_script']).read_text()
            finalizer_script = Path(report['finalizer_script']).read_text()
            self.assertIn('#SBATCH --array=0-3%2', array_script)
            self.assertIn('--array-task', array_script)
            self.assertIn('--finalize', finalizer_script)
            self.assertEqual(report['submitted'], False)
            with patch('experiment_scheduler.subprocess.run', side_effect=[
                subprocess.CompletedProcess([], 0, '91001\n', ''),
                subprocess.CompletedProcess([], 0, '91002\n', ''),
            ]) as submit:
                submitted = submit_subject_adapter_array(prepared_dir)
            self.assertEqual(submitted['array_job_id'], '91001')
            self.assertEqual(submitted['finalizer_job_id'], '91002')
            self.assertEqual(submitted['dependency'], 'afterok:91001')
            self.assertIn('--dependency=afterok:91001', submit.call_args_list[1].args[0])
            self.assertTrue(json.loads((prepared_dir / 'experiment_matrix.json').read_text())['submitted'])

    def test_subject_adapter_routes_and_trains_only_selected_subjects(self):
        layer = SubjectLowRankLinear(torch.nn.Linear(3, 4), 3, rank=2, alpha=4)
        inputs = torch.randn(2, 5, 3)
        ids = torch.tensor([0, 2])
        output = layer(inputs, ids)
        expected = layer.shared(inputs) + 2 * torch.stack([
            inputs[index] @ layer.adapter_a[subject] @ layer.adapter_b[subject]
            for index, subject in enumerate(ids)
        ])
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertGreater(layer.shared.weight.grad.abs().sum().item(), 0)
        self.assertGreater(layer.adapter_a.grad[0].abs().sum().item(), 0)
        self.assertGreater(layer.adapter_b.grad[2].abs().sum().item(), 0)
        self.assertEqual(layer.adapter_a.grad[1].abs().sum().item(), 0)
        with self.assertRaisesRegex(ValueError, 'Unknown subject'):
            layer(inputs, torch.tensor([0, 3]))
        with self.assertRaisesRegex(ValueError, 'required'):
            layer(inputs, None)

    def test_hit_cutoffs_larger_than_index_remain_correct(self):
        metrics = compute_ranking_metrics([4, 2], 2)
        self.assertEqual(metrics['rr'], 0.5)
        self.assertEqual(metrics['hit_at_1'], 0)
        self.assertEqual(metrics['hit_at_5'], 1)
        self.assertEqual(metrics['hit_at_20'], 1)

    def test_unified_index_with_seen_subjects_and_every_pooling_strategy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [write_synthetic_pairs(root / f'{source}_ict_pairs.npy', source, 30, True)
                     for source in ('alice', 'nieuwland')]
            pairs, metadata = load_combined_datasets(paths)
            for pair in pairs:
                pair['participant_id'] = pair['participant_id'].split('_')[-1]
            train, val, test, dimensions = create_dataloaders(
                None, TinyTokenizer(), 4, 24, 4, 0.8, False, 90,
                combined_dataset={'ict_pairs': pairs, 'metadata': metadata}, seen_subjects_only=True,
            )
            mapping = {key: index for index, key in enumerate(sorted(
                {subject_key(pair) for pair in train.dataset.processed_pairs}))}
            self.assertEqual(len(mapping), 10)
            train_indices = {pair['original_idx'] for pair in train.dataset.processed_pairs}
            val_indices = {pair['original_idx'] for pair in val.dataset.processed_pairs}
            test_indices = {pair['original_idx'] for pair in test.dataset.processed_pairs}
            self.assertFalse(train_indices & val_indices)
            self.assertFalse(train_indices & test_indices)
            self.assertFalse(val_indices & test_indices)
            for loader in (val, test):
                self.assertTrue({subject_key(pair) for pair in loader.dataset.processed_pairs} <= set(mapping))
            test.dataset.set_masking_probability(1.0)
            unique_passages = {test.dataset[index]['metadata']['document_text'].strip()
                               for index in range(len(test.dataset))}
            test.dataset.set_masking_probability(0.9)
            for pooling in ('cls', 'mean', 'max', 'multi'):
                model = create_model(hidden_dim=8, eeg_arch='transformer', device='cpu', use_lora=False,
                                     pooling_strategy=pooling, global_eeg_dims=dimensions,
                                     use_pretrained_text=False)
                before = torch.get_rng_state().clone()
                baseline = evaluate_unified_index(model, test, 'cpu', mapping, 100)
                torch.testing.assert_close(torch.get_rng_state(), before)
                self.assertTrue(model.training)
                self.assertEqual(test.dataset.get_current_masking_probability(), 0.9)
                self.assertEqual(baseline['num_queries'], len(test.dataset))
                self.assertEqual(baseline['num_documents'], len(unique_passages))
                self.assertEqual(set(baseline['by_source']), {'alice', 'nieuwland'})
                self.assertEqual(set(baseline['overall']), {'mrr', 'hit_at_1', 'hit_at_5', 'hit_at_10', 'hit_at_20'})
                model.enable_subject_adapters(len(mapping), 2, 4)
                adapted = evaluate_unified_index(model, test, 'cpu', mapping, 100)
                self.assertEqual(adapted['index_fingerprint'], baseline['index_fingerprint'])
                with self.assertRaisesRegex(ValueError, 'Unseen subjects'):
                    evaluate_unified_index(model, test, 'cpu', {}, 100)
                self.assertEqual(test.dataset.get_current_masking_probability(), 0.9)

    def test_seen_subject_split_rejects_insufficient_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_synthetic_pairs(Path(temp_dir) / 'alice_ict_pairs.npy', 'alice', 5, True)
            with self.assertRaisesRegex(ValueError, 'at least three'):
                create_dataloaders(path, TinyTokenizer(), 4, 24, 4, 0.8, False, 90,
                                   seen_subjects_only=True)

    def test_adapter_search_uses_mean_validation_mrr(self):
        trials = search_trials([2], [0.25, 1], ['input_only'])
        runs = [{'trial': trials[1]['name'], 'validation_mrr': value} for value in (0.2, 0.8)]
        runs += [{'trial': trials[2]['name'], 'validation_mrr': value} for value in (0.6, 0.6)]
        selected, scores = select_adapter(trials, runs)
        self.assertEqual(selected['alpha'], 2)
        self.assertEqual(scores[trials[1]['name']], 0.5)

    def test_complete_subject_search_training_testing_and_logging(self):
        from transformers import BertConfig, BertModel

        def tiny_bert(*args, **kwargs):
            return BertModel(BertConfig(vocab_size=256, hidden_size=8, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=16))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = write_synthetic_pairs(root / 'alice_ict_pairs.npy', 'alice', 30, True)
            visual = write_synthetic_pairs(root / 'nieuwland_ict_pairs.npy', 'nieuwland', 30, True)
            output = root / 'results'
            args = parse_args([
                '--audio_path', str(audio), '--visual_path', str(visual), '--output_dir', str(output),
                '--hidden_dim', '8', '--batch_size', '4', '--epochs', '1', '--warmup_epochs', '0',
                '--ranks', '2', '--alpha_ratios', '0.25', '1', '--seed', '1',
                '--test_masking_levels', '0', '100', '--max_eeg_len', '4', '--max_text_len', '24',
                '--device', 'cpu', '--balance_sources',
            ])
            with patch('subject_adapter_experiment.AutoTokenizer.from_pretrained', return_value=TinyTokenizer()), \
                    patch('mv_models.AutoModel.from_pretrained', side_effect=tiny_bert):
                comparison = run_experiment(args)
            self.assertEqual(set(comparison['baseline']), {'0', '100', 'mean_across_masking_levels'})
            for placement in args.placements:
                self.assertIn('hit_at_20', comparison[placement]['100'])
                self.assertEqual(comparison[placement]['100']['mrr']['std'], 0)
            self.assertEqual(comparison['num_seeds'], 1)
            self.assertEqual(set(comparison['paired_sulora_minus_baseline']), set(args.placements))
            self.assertEqual(len(json.loads((output / 'search_results.json').read_text())), 7)
            self.assertEqual(len(list(output.glob('*/seed_*/best_model.pt'))), 7)
            self.assertEqual(len(list(output.glob('*/seed_*/test_metrics.json'))), 4)
            selection = json.loads((output / 'selected_adapter.json').read_text())
            self.assertEqual(set(selection['selected_by_placement']), set(args.placements))
            self.assertIn(selection['selected']['placement'], args.placements)
            self.assertEqual(json.loads((output / 'experiment_matrix.json').read_text())['total_fits'], 7)
            events = [json.loads(line) for line in (output / 'experiment_log.jsonl').read_text().splitlines()]
            selection_index = next(index for index, event in enumerate(events) if event['event'] == 'adapter_selected')
            self.assertTrue(all(index > selection_index for index, event in enumerate(events)
                                if event['event'] == 'test_completed'))
            self.assertEqual(events[-1]['event'], 'experiment_completed')
            checkpoint = torch.load(output / 'baseline' / 'seed_1' / 'best_model.pt', weights_only=False)
            self.assertEqual(checkpoint['parameters']['subject_adapters'], 0)
            self.assertEqual(len(checkpoint['subject_mapping']), 10)
            with self.assertRaisesRegex(ValueError, 'empty output'):
                run_experiment(args)

    def test_independent_array_trials_finalize_only_after_all_configs_exist(self):
        from transformers import BertConfig, BertModel

        def tiny_bert(*args, **kwargs):
            return BertModel(BertConfig(vocab_size=256, hidden_size=8, num_hidden_layers=1,
                                        num_attention_heads=2, intermediate_size=16))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = write_synthetic_pairs(root / 'alice_ict_pairs.npy', 'alice', 30, True)
            visual = write_synthetic_pairs(root / 'nieuwland_ict_pairs.npy', 'nieuwland', 30, True)
            output = root / 'array_results'
            args = parse_args([
                '--audio_path', str(audio), '--visual_path', str(visual), '--output_dir', str(output),
                '--hidden_dim', '8', '--batch_size', '4', '--epochs', '1', '--warmup_epochs', '0',
                '--ranks', '2', '--alpha_ratios', '1', '--seed', '1',
                '--max_eeg_len', '4', '--max_text_len', '24', '--device', 'cpu',
            ])
            with patch('subject_adapter_experiment.AutoTokenizer.from_pretrained', return_value=TinyTokenizer()), \
                    patch('mv_models.AutoModel.from_pretrained', side_effect=tiny_bert):
                for trial_index in range(3):
                    task_args = copy.copy(args)
                    task_args.trial_index = trial_index
                    run_experiment(task_args)
                with self.assertRaisesRegex(FileNotFoundError, 'did not complete'):
                    finalize_args = copy.copy(args)
                    finalize_args.finalize = True
                    run_experiment(finalize_args)
                last_task = copy.copy(args)
                last_task.trial_index = 3
                run_experiment(last_task)
                with patch('subject_adapter_experiment.AutoTokenizer.from_pretrained', return_value=TinyTokenizer()), \
                        patch('mv_models.AutoModel.from_pretrained', side_effect=tiny_bert):
                    summary = run_experiment(finalize_args)
            self.assertEqual(summary['num_seeds'], 1)
            self.assertEqual(set(summary['paired_sulora_minus_baseline']),
                             {'input_only', 'eeg_backbone', 'full_eeg'})
            self.assertTrue((output / 'comparison.json').is_file())
            self.assertEqual(len(list((output / 'trials').glob('*/test_metrics.json'))), 4)

    def test_single_source_loader_deserializes_source_once_for_three_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = write_synthetic_pairs(root / "alice_ict_pairs.npy", "alice", 12, True)
            dimensions = compute_global_eeg_dimensions(audio_path, max_eeg_len=4)
            with patch("controller.load_combined_datasets", wraps=load_combined_datasets) as loader:
                create_dataloaders(
                    data_path=audio_path, tokenizer=TinyTokenizer(), batch_size=4, max_text_len=24,
                    max_eeg_len=4, train_ratio=0.8, holdout_subjects=False,
                    training_masking_level=90, global_eeg_dims=dimensions,
                    pair_split_seed=42, balance_sources=False,
                )
            self.assertEqual(loader.call_count, 1)

    def test_task_reader_keeps_aligned_and_requested_unaligned_people(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            root.mkdir()
            question_id = "q_smoke"
            trial_rows = [
                make_trial_row("P003", "ses-smoke", question_id),
                make_trial_row("P001", "unaligned_task_export", question_id),
                make_trial_row("P002", "unaligned_task_export", question_id),
            ]
            with (root / "trial_response_ground_truth.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0]))
                writer.writeheader()
                writer.writerows(trial_rows)

            aligned_dir = root / "P003" / "ses-smoke"
            aligned_dir.mkdir(parents=True)
            info = mne.create_info(["C1", "C2"], sfreq=10.0, ch_types="eeg")
            events = np.array([[20, 0, 1], [40, 0, 1]])
            metadata = pd.DataFrame([
                {"word": "what", "onset": 2.0, "part": 1, "trial": 1, "question_id": question_id,
                 "word_index": 0, "word_index_ambiguous": False},
                {"word": "capital", "onset": 4.0, "part": 1, "trial": 1, "question_id": question_id,
                 "word_index": 2, "word_index_ambiguous": False},
            ])
            epochs = mne.EpochsArray(np.ones((2, 2, 11), dtype=np.float64), info, events,
                                     tmin=-0.2, event_id={"fix": 1}, metadata=metadata, verbose="ERROR")
            epochs.save(aligned_dir / "P003_reading-epo.fif", overwrite=True, verbose="ERROR")

            for participant in ("P001", "P002"):
                unaligned_dir = root / participant / "unaligned_task_export"
                unaligned_dir.mkdir(parents=True)
                raw = mne.io.RawArray(np.ones((2, 200), dtype=np.float64), info, verbose="ERROR")
                raw.save(unaligned_dir / f"{participant}_unaligned_raw.fif", overwrite=True, verbose="ERROR")
                fixation_path = unaligned_dir / f"{participant}_reading_fixations.csv"
                with fixation_path.open("w", newline="", encoding="utf-8") as stream:
                    fields = ["word", "onset", "part", "trial", "question_id", "word_index", "word_index_ambiguous"]
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows([
                        {"word": "what", "onset": 2.0, "part": 1, "trial": 1, "question_id": question_id,
                         "word_index": 0, "word_index_ambiguous": False},
                        {"word": "capital", "onset": 4.0, "part": 1, "trial": 1, "question_id": question_id,
                         "word_index": 2, "word_index_ambiguous": False},
                    ])

            output = root / "task_pairs.npy"
            manifest = build_task_pairs(root, output, unaligned_clock_offset_seconds=0.0)
            pairs = np.load(output, allow_pickle=True).item()["ict_pairs"]
            self.assertEqual(len(pairs), 3)
            self.assertEqual({pair["participant_id"] for pair in pairs}, {"P001", "P002", "P003"})
            self.assertTrue(all(not pair["supports_runtime_masking"] for pair in pairs))
            self.assertEqual({pair["alignment_status"] for pair in pairs},
                             {"aligned_word_index", "unverified_psychopy_to_eeg_offset"})
            self.assertTrue(all(pair["source_identity_group"] == "P001_P002_byte_identical"
                                for pair in pairs if pair["participant_id"] in {"P001", "P002"}))
            self.assertEqual(manifest["pair_count"], 3)
            self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text())["pair_count"], 3)

    def test_all_seven_mixtures_split_balance_train_and_test_by_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "audio": write_synthetic_pairs(root / "alice_ict_pairs.npy", "alice", 30, True),
                "visual": write_synthetic_pairs(root / "nieuwland_ict_pairs.npy", "nieuwland", 30, True),
                "task": write_synthetic_pairs(root / "authentic_query_task_pairs.npy", "authentic_query", 15, False),
            }
            source_labels = {"audio": "alice", "visual": "nieuwland", "task": "authentic_query"}
            tokenizer = TinyTokenizer()
            test_ids_by_source = {source: [] for source in source_labels.values()}

            for name, source_names in EXPERIMENTS:
                selected_paths = [paths[source] for source in source_names]
                combined = None
                data_path = None
                if len(selected_paths) == 1:
                    data_path = selected_paths[0]
                    dimensions = compute_global_eeg_dimensions(data_path, max_eeg_len=4)
                else:
                    pairs, metadata = load_combined_datasets(selected_paths)
                    combined = {"ict_pairs": pairs, "metadata": metadata}
                    from mv_dataloader import compute_combined_eeg_dimensions
                    dimensions = compute_combined_eeg_dimensions(pairs, max_eeg_len=4)

                train_loader, val_loader, test_loader, _ = create_dataloaders(
                    data_path=data_path, tokenizer=tokenizer, batch_size=6, max_text_len=24,
                    max_eeg_len=4, train_ratio=0.8, holdout_subjects=False,
                    training_masking_level=90, global_eeg_dims=dimensions,
                    combined_dataset=combined, pair_split_seed=42, balance_sources=len(source_names) > 1,
                )
                self.assertGreater(len(train_loader.dataset), 0, name)
                self.assertGreater(len(val_loader.dataset), 0, name)
                self.assertGreater(len(test_loader.dataset), 0, name)

                source_test_loaders = create_source_test_loaders(test_loader)
                for source, source_loader in source_test_loaders.items():
                    test_ids = {
                        (row["participant_id"], row["sentence_id"], row["question_id"])
                        for row in test_loader.dataset.processed_pairs
                        if row["dataset_source"] == source
                    }
                    test_ids_by_source[source].append(test_ids)
                    self.assertEqual({
                        source_loader.dataset.dataset.processed_pairs[index]["dataset_source"]
                        for index in source_loader.dataset.indices
                    }, {source})

                if len(source_names) > 1:
                    self.assertIsInstance(train_loader.sampler, WeightedRandomSampler, name)
                    expected_sources = [source_labels[source] for source in source_names]
                    totals = {source: 0.0 for source in expected_sources}
                    for weight, pair in zip(train_loader.sampler.weights.tolist(),
                                            train_loader.dataset.processed_pairs):
                        source = pair["dataset_source"]
                        totals[source] += weight
                    for total in totals.values():
                        self.assertAlmostEqual(total, 1.0, places=6, msg=name)

                task_loader = create_source_test_loaders(test_loader).get("authentic_query")
                if task_loader is not None:
                    sample = task_loader.dataset[0]
                    self.assertFalse(sample["metadata"]["was_masked"], name)
                    self.assertIn("Reference answer", sample["metadata"]["document_text"], name)

                model = create_model(
                    hidden_dim=8, eeg_arch="simple", device="cpu", use_lora=False,
                    pooling_strategy="mean", encoder_type="dual", global_eeg_dims=dimensions,
                    query_type="eeg", use_pretrained_text=False,
                )
                model.set_tokenizer_vocab_size(len(tokenizer))
                batch = next(iter(train_loader))
                output = model(batch["eeg_queries"], batch["text_queries"], batch["docs"], batch["eeg_mv_masks"])
                loss, _ = compute_contrastive_loss(output["query_vectors"], output["doc_vectors"], "mean")
                self.assertTrue(torch.isfinite(loss).item(), name)
                loss.backward()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                optimizer.step()

                for source, source_loader in source_test_loaders.items():
                    with patch("mv_training.wandb.log"):
                        metrics = test_model(
                            model, source_loader, torch.device("cpu"),
                            test_masking_levels=[0, 100], primary_masking_level=100,
                            wandb_namespace=f"smoke/{source}",
                        )
                    self.assertEqual(metrics["test_ranking/summary/num_test_samples"],
                                     len(source_loader.dataset), name)
                    self.assertEqual(metrics["test_ranking/summary/has_multiple_datasets"], False, name)

            for source, split_sets in test_ids_by_source.items():
                self.assertTrue(split_sets, source)
                self.assertTrue(all(test_set == split_sets[0] for test_set in split_sets), source)

    def test_scheduler_prepares_without_submission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = prepare_jobs(
                root / "alice_ict_pairs.npy", root / "nieuwland_ict_pairs.npy",
                root / "authentic_query_task_pairs.npy", root / "prepared",
                python_executable="python", epochs=1, batch_size=2,
            )
            prepared = root / "prepared"
            self.assertFalse(report["submitted"])
            self.assertEqual(len(report["matrix"]), 7)
            self.assertEqual(len(list((prepared / "jobs").glob("*.sbatch"))), 7)
            for script in (prepared / "jobs").glob("*.sbatch"):
                self.assertIn("#SBATCH --partition=gpu", script.read_text())
                self.assertIn("controller.py", script.read_text())
                self.assertIn("--no_lora", script.read_text())
            self.assertEqual(report["text_encoder_policy"], "pretrained ColBERT frozen (LoRA disabled)")
            self.assertIn("No jobs were submitted", (prepared / "PREPARED_NOT_SUBMITTED.txt").read_text())


if __name__ == "__main__":
    unittest.main()
