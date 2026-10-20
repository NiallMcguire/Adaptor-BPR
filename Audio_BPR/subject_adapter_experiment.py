#!/usr/bin/env python3
"""Matched cross-sensory BPR experiments with seen-subject EEG adapters."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import AutoTokenizer

from controller import create_dataloaders, handle_multiple_datasets, set_seeds
from mv_models import create_model, SubjectLowRankLinear, SUBJECT_ADAPTER_PLACEMENTS
from mv_training import batch_subject_ids, compute_contrastive_loss, evaluate_unified_index, subject_key


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def log_event(output_dir, event, **details):
    record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'event': event, **details}
    with (output_dir / 'experiment_log.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')


def parameter_counts(model):
    return {
        'total': sum(parameter.numel() for parameter in model.parameters()),
        'trainable': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        'subject_adapters': sum(parameter.numel() for name, parameter in model.named_parameters()
                                if 'adapter_a' in name or 'adapter_b' in name),
        'adapted_layers': [name for name, module in model.named_modules()
                  if isinstance(module, SubjectLowRankLinear)],
    }


def make_model(args, dimensions, subject_mapping, trial, device):
    model = create_model(
        colbert_model_name=args.text_model, hidden_dim=args.hidden_dim,
        eeg_arch=args.eeg_arch, device=device, use_lora=False,
        pooling_strategy=args.pooling_strategy, encoder_type='dual',
        global_eeg_dims=dimensions, query_type='eeg', use_pretrained_text=True,
    )
    if model.text_encoder.config.hidden_size != args.hidden_dim:
        raise ValueError('hidden_dim must match the frozen text encoder dimension')
    model.text_projection = torch.nn.Identity()
    if trial['variant'] == 'sulora':
        model.enable_subject_adapters(len(subject_mapping), trial['rank'], trial['alpha'], trial['placement'])
        model.to(device)
    return model


def train_trial(model, train_loader, val_loader, subject_mapping, args, trial,
                seed, device, output_dir, checkpoint):
    started = perf_counter()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    def learning_rate_factor(epoch):
        if args.warmup_epochs and epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        return max(0.0, (args.epochs - epoch) / max(1, args.epochs - args.warmup_epochs))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    best_mrr, best_epoch, stale_epochs = -1.0, 0, 0
    best_validation = None
    counts = parameter_counts(model)
    set_seeds(seed)
    if getattr(train_loader.sampler, 'generator', None) is not None:
        train_loader.sampler.generator.manual_seed(seed)
    log_event(output_dir, 'trial_started', trial=trial, seed=seed, parameters=counts)
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.text_encoder.eval()
        total_loss, sample_count = 0.0, 0
        for batch in train_loader:
            ids = batch_subject_ids(batch, subject_mapping, device)
            outputs = model(
                batch['eeg_queries'].to(device),
                {key: value.to(device) for key, value in batch['text_queries'].items()},
                {key: value.to(device) for key, value in batch['docs'].items()},
                batch['eeg_mv_masks'].to(device), subject_ids=ids,
            )
            loss, _ = compute_contrastive_loss(outputs['query_vectors'], outputs['doc_vectors'],
                                               model.pooling_strategy, temperature=args.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = len(batch['metadata'])
            total_loss += loss.item() * count
            sample_count += count
        validation = evaluate_unified_index(
            model, val_loader, device, subject_mapping, args.primary_masking_level, args.hit_k)
        mrr = validation['overall']['mrr']
        improved = mrr > best_mrr
        if improved:
            best_mrr, best_epoch, stale_epochs = mrr, epoch, 0
            best_validation = validation
            torch.save({
                'model_state_dict': {name: value.detach().cpu().clone()
                                     for name, value in model.state_dict().items()},
                'trial': trial, 'seed': seed, 'best_epoch': best_epoch,
                'validation': validation, 'subject_mapping': subject_mapping,
                'global_eeg_dims': dimensions_from_loader(train_loader),
                'config': vars(args), 'parameters': counts,
            }, checkpoint)
        else:
            stale_epochs += 1
        log_event(output_dir, 'epoch_completed', trial=trial['name'], seed=seed, epoch=epoch,
                  train_loss=total_loss / sample_count, learning_rate=optimizer.param_groups[0]['lr'],
                  validation=validation['overall'], index_fingerprint=validation['index_fingerprint'],
                  best_epoch=best_epoch, improved=improved)
        print(f"{trial['name']} seed={seed} epoch={epoch}: validation MRR={mrr:.4f}, best={best_mrr:.4f}")
        scheduler.step()
        if stale_epochs >= args.patience:
            break
    result = {'trial': trial['name'], 'seed': seed, 'best_epoch': best_epoch,
              'validation_mrr': best_mrr, 'checkpoint': str(checkpoint), 'parameters': counts,
              'validation_index_fingerprint': best_validation['index_fingerprint'],
              'fit_seconds': perf_counter() - started}
    log_event(output_dir, 'trial_completed', **result)
    return result


def dimensions_from_loader(loader):
    dataset = loader.dataset
    return (dataset.global_max_words, dataset.global_max_time, dataset.global_max_channels)


def search_trials(ranks, alpha_ratios, placements=SUBJECT_ADAPTER_PLACEMENTS):
    trials = [{'name': 'baseline', 'variant': 'baseline', 'rank': None, 'alpha': None}]
    for placement in placements:
        if placement not in SUBJECT_ADAPTER_PLACEMENTS:
            raise ValueError(f'Unknown adapter placement: {placement}')
        for rank in ranks:
            for ratio in alpha_ratios:
                name = f"sulora_{placement}_r{rank}_ratio{ratio}".replace('.', 'p')
                trials.append({'name': name, 'variant': 'sulora', 'rank': rank,
                               'alpha': rank * ratio, 'alpha_over_rank': ratio,
                               'placement': placement})
    return trials


def select_adapter(trials, runs):
    scores = {trial['name']: float(np.mean([run['validation_mrr'] for run in runs
                                          if run['trial'] == trial['name']]))
              for trial in trials if trial['variant'] == 'sulora'}
    selected = max(scores, key=scores.get)
    return next(trial for trial in trials if trial['name'] == selected), scores


def aggregate_results(results, masking_levels):
    summary = {}
    groups = {'baseline': [result for result in results if result['variant'] == 'baseline']}
    for result in results:
        if result['variant'] == 'sulora':
            groups.setdefault(result['trial']['placement'], []).append(result)
    for placement, matching in groups.items():
        summary[placement] = {}
        for level in masking_levels:
            rows = [result['metrics'][str(level)]['overall'] for result in matching]
            summary[placement][str(level)] = {
                name: {'mean': float(np.mean([row[name] for row in rows])),
                       'std': float(np.std([row[name] for row in rows]))}
                for name in rows[0]
            }
        averages = [{name: float(np.mean([result['metrics'][str(level)]['overall'][name]
                                         for level in masking_levels]))
                     for name in rows[0]} for result in matching]
        summary[placement]['mean_across_masking_levels'] = {
            name: {'mean': float(np.mean([row[name] for row in averages])),
                   'std': float(np.std([row[name] for row in averages]))}
            for name in averages[0]
        }
    paired = {}
    for placement, matching in groups.items():
        if placement == 'baseline':
            continue
        paired[placement] = {}
        for level in masking_levels:
            differences = []
            for result in matching:
                baseline = next(value for value in groups['baseline'] if value['seed'] == result['seed'])[
                    'metrics'][str(level)]
                adapter = result['metrics'][str(level)]
                if baseline['index_fingerprint'] != adapter['index_fingerprint']:
                    raise RuntimeError('Baseline and SuLoRA candidate indices differ')
                differences.append({name: adapter['overall'][name] - value
                                    for name, value in baseline['overall'].items()})
            paired[placement][str(level)] = {
                name: {'mean': float(np.mean([row[name] for row in differences])),
                       'std': float(np.std([row[name] for row in differences]))}
                for name in differences[0]}
    summary['paired_sulora_minus_baseline'] = paired
    summary['num_seeds'] = len(groups['baseline'])
    summary['standard_deviation_definition'] = 'ddof=0; one seed yields zero std, not an uncertainty estimate'
    return summary


def load_experiment_data(args):
    """Rebuild the deterministic shared splits independently in each Slurm task."""
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    set_seeds(args.split_seed)
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    combined, dimensions = handle_multiple_datasets([args.audio_path, args.visual_path], args.max_eeg_len)
    sources = {pair['dataset_source'] for pair in combined['ict_pairs']}
    if sources != {'alice', 'nieuwland'}:
        raise ValueError(f'Expected Alice and Nieuwland ICT files, found {sorted(sources)}')
    train_loader, val_loader, test_loader, dimensions = create_dataloaders(
        None, tokenizer, args.batch_size, args.max_text_len, args.max_eeg_len,
        args.train_ratio, False, args.training_masking_level,
        global_eeg_dims=dimensions, combined_dataset=combined,
        pair_split_seed=args.split_seed, balance_sources=args.balance_sources, seen_subjects_only=True,
    )
    mapping = {key: index for index, key in enumerate(sorted(
        {subject_key(pair) for pair in train_loader.dataset.processed_pairs}))}
    split_manifest = {}
    for name, loader in zip(('train', 'validation', 'test'), (train_loader, val_loader, test_loader)):
        pairs = loader.dataset.processed_pairs
        if not pairs or {subject_key(pair) for pair in pairs} - set(mapping):
            raise ValueError(f'{name} must be nonempty and contain only training subjects')
        if {pair['dataset_source'] for pair in pairs} != sources:
            raise ValueError(f'{name} must contain both sensory datasets')
        split_manifest[name] = [{'subject': subject_key(pair), 'original_idx': pair['original_idx']}
                                for pair in pairs]
    return device, (train_loader, val_loader, test_loader), dimensions, mapping, split_manifest


def run_single_trial(args):
    """Train exactly one independently schedulable matrix configuration."""
    trials = search_trials(args.ranks, args.alpha_ratios, args.placements)
    if args.trial_index is None or not 0 <= args.trial_index < len(trials):
        raise ValueError(f'trial_index must be in [0, {len(trials) - 1}]')
    trial = trials[args.trial_index]
    output_dir = Path(args.output_dir)
    trial_dir = output_dir / 'trials' / trial['name']
    trial_dir.mkdir(parents=True, exist_ok=True)
    result_path = trial_dir / 'validation_result.json'
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('trial') != trial['name'] or not Path(result.get('checkpoint', '')).is_file():
            raise ValueError(f'Existing completed trial is inconsistent: {result_path}')
        print(f"{trial['name']} already completed; reusing its validation result")
        return result
    write_json(trial_dir / 'experiment_config.json', vars(args))
    log_event(trial_dir, 'trial_process_started', trial=trial, array_index=args.trial_index)
    try:
        device, loaders, dimensions, mapping, _ = load_experiment_data(args)
        train_loader, val_loader, _ = loaders
        set_seeds(args.seeds[0])
        model = make_model(args, dimensions, mapping, trial, device)
        result = train_trial(model, train_loader, val_loader, mapping, args, trial,
                             args.seeds[0], device, trial_dir, trial_dir / 'best_model.pt')
        result.update({'variant': trial['variant'], 'trial_config': trial,
                       'array_index': args.trial_index})
        write_json(result_path, result)
        log_event(trial_dir, 'trial_process_completed', **result)
        return result
    except Exception as error:
        log_event(trial_dir, 'trial_process_failed', error_type=type(error).__name__, message=str(error))
        raise


def finalize_array_experiment(args):
    """Select from completed array trials and test baseline plus placement winners."""
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f'Array output directory does not exist: {output_dir}')
    trials = search_trials(args.ranks, args.alpha_ratios, args.placements)
    runs = []
    for trial in trials:
        result_path = output_dir / 'trials' / trial['name'] / 'validation_result.json'
        if not result_path.is_file():
            raise FileNotFoundError(f"Trial {trial['name']} did not complete: {result_path}")
        result = json.loads(result_path.read_text(encoding='utf-8'))
        checkpoint_path = Path(result['checkpoint'])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Trial checkpoint is missing: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if saved['validation']['index_fingerprint'] != result['validation_index_fingerprint']:
            raise ValueError(f"Validation index fingerprint mismatch for {trial['name']}")
        runs.append(result)

    selected_overall, scores = select_adapter(trials, runs)
    selected_by_placement = {
        placement: select_adapter([trial for trial in trials
                                   if trial.get('placement') == placement], runs)[0]
        for placement in args.placements
    }
    selection = {
        'selected': selected_overall,
        'mean_validation_mrr_by_configuration': scores,
        'selected_by_placement': selected_by_placement,
        'selection_masking_level': args.primary_masking_level,
        'selection_rule': 'highest combined validation MRR; first grid entry breaks ties',
    }
    write_json(output_dir / 'selected_adapter.json', selection)
    log_event(output_dir, 'adapter_selected_from_array', **selection)

    device, loaders, dimensions, mapping, split_manifest = load_experiment_data(args)
    _, _, test_loader = loaders
    write_json(output_dir / 'split_manifest.json', split_manifest)
    write_json(output_dir / 'subject_mapping.json', mapping)
    selected_trials = [trials[0], *selected_by_placement.values()]
    results = []
    for trial in selected_trials:
        run = next(result for result in runs if result['trial'] == trial['name'])
        model = make_model(args, dimensions, mapping, trial, device)
        saved = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
        test_started = perf_counter()
        metrics = {str(level): evaluate_unified_index(
            model, test_loader, device, mapping, level, args.hit_k)
            for level in args.test_masking_levels}
        result = {
            'variant': trial['variant'], 'trial': trial, 'seed': args.seeds[0],
            'metrics': metrics, 'parameters': run['parameters'],
            'fit_seconds': run['fit_seconds'], 'test_seconds': perf_counter() - test_started,
        }
        write_json(output_dir / 'trials' / trial['name'] / 'test_metrics.json', result)
        results.append(result)
        log_event(output_dir, 'array_test_completed', trial=trial['name'],
                  metrics={level: value['overall'] for level, value in metrics.items()},
                  test_seconds=result['test_seconds'])
        del model, saved
    summary = aggregate_results(results, args.test_masking_levels)
    summary['validation_selected_placement'] = selected_overall['placement']
    summary['selected_configurations'] = selected_by_placement
    summary['resources'] = {
        ('baseline' if result['variant'] == 'baseline' else result['trial']['placement']):
        {'parameters': result['parameters'], 'fit_seconds': result['fit_seconds'],
         'test_seconds': result['test_seconds']} for result in results}
    write_json(output_dir / 'comparison.json', summary)
    log_event(output_dir, 'array_experiment_completed', selected=selected_overall, comparison=summary)
    return summary


def run_experiment(args):
    if args.trial_index is not None:
        return run_single_trial(args)
    if args.finalize:
        return finalize_array_experiment(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError('Use a new or empty output directory to preserve previous experiment logs')
    write_json(output_dir / 'experiment_config.json', vars(args))
    log_event(output_dir, 'experiment_started', config=vars(args))
    try:
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        set_seeds(args.split_seed)
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)
        combined, dimensions = handle_multiple_datasets([args.audio_path, args.visual_path], args.max_eeg_len)
        sources = {pair['dataset_source'] for pair in combined['ict_pairs']}
        if sources != {'alice', 'nieuwland'}:
            raise ValueError(f'Expected Alice and Nieuwland ICT files, found {sorted(sources)}')
        loaders = create_dataloaders(
            None, tokenizer, args.batch_size, args.max_text_len, args.max_eeg_len,
            args.train_ratio, False, args.training_masking_level,
            global_eeg_dims=dimensions, combined_dataset=combined,
            pair_split_seed=args.split_seed, balance_sources=args.balance_sources, seen_subjects_only=True,
        )
        train_loader, val_loader, test_loader, dimensions = loaders
        mapping = {key: index for index, key in enumerate(sorted(
            {subject_key(pair) for pair in train_loader.dataset.processed_pairs}))}
        split_manifest = {}
        for name, loader in zip(('train', 'validation', 'test'), loaders[:3]):
            pairs = loader.dataset.processed_pairs
            if not pairs or {subject_key(pair) for pair in pairs} - set(mapping):
                raise ValueError(f'{name} must be nonempty and contain only training subjects')
            if {pair['dataset_source'] for pair in pairs} != sources:
                raise ValueError(f'{name} must contain both sensory datasets')
            split_manifest[name] = [{'subject': subject_key(pair), 'original_idx': pair['original_idx']}
                                    for pair in pairs]
        write_json(output_dir / 'split_manifest.json', split_manifest)
        write_json(output_dir / 'subject_mapping.json', mapping)
        log_event(output_dir, 'data_ready', subjects=len(mapping), global_eeg_dims=dimensions,
                  splits={name: len(pairs) for name, pairs in split_manifest.items()},
                  device=str(device), torch_version=torch.__version__)
        trials = search_trials(args.ranks, args.alpha_ratios, args.placements)
        write_json(output_dir / 'experiment_matrix.json', {'trials': trials, 'seeds': args.seeds,
                                  'total_fits': len(trials) * len(args.seeds)})
        log_event(output_dir, 'search_planned', placements=args.placements,
              configurations=len(trials) - 1, total_fits=len(trials) * len(args.seeds))
        runs = []
        for trial in trials:
            for seed in args.seeds:
                run_dir = output_dir / trial['name'] / f'seed_{seed}'
                run_dir.mkdir(parents=True)
                set_seeds(seed)
                model = make_model(args, dimensions, mapping, trial, device)
                result = train_trial(model, train_loader, val_loader, mapping, args, trial, seed,
                                     device, output_dir, run_dir / 'best_model.pt')
                runs.append(result)
                write_json(run_dir / 'validation_result.json', result)
                write_json(output_dir / 'search_results.json', runs)
                del model
        selected, scores = select_adapter(trials, runs)
        selected_by_placement = {
            placement: select_adapter([trial for trial in trials if trial.get('placement') == placement], runs)[0]
            for placement in args.placements}
        selection = {'selected': selected, 'mean_validation_mrr_by_configuration': scores,
                     'selected_by_placement': selected_by_placement,
                     'selection_masking_level': args.primary_masking_level,
                     'selection_rule': 'highest combined validation MRR averaged across seeds; first grid entry breaks ties'}
        write_json(output_dir / 'selected_adapter.json', selection)
        log_event(output_dir, 'adapter_selected', **selection)
        results = []
        for trial in (trials[0], *selected_by_placement.values()):
            for seed in args.seeds:
                run = next(run for run in runs if run['trial'] == trial['name'] and run['seed'] == seed)
                set_seeds(seed)
                model = make_model(args, dimensions, mapping, trial, device)
                saved = torch.load(run['checkpoint'], map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model_state_dict'])
                test_started = perf_counter()
                metrics = {str(level): evaluate_unified_index(model, test_loader, device, mapping, level, args.hit_k)
                           for level in args.test_masking_levels}
                result = {'variant': trial['variant'], 'trial': trial, 'seed': seed, 'metrics': metrics,
                          'parameters': run['parameters'], 'fit_seconds': run['fit_seconds'],
                          'test_seconds': perf_counter() - test_started}
                write_json(Path(run['checkpoint']).parent / 'test_metrics.json', result)
                results.append(result)
                log_event(output_dir, 'test_completed', trial=trial['name'], seed=seed,
                          metrics={level: value['overall'] for level, value in metrics.items()},
                          test_seconds=result['test_seconds'])
                del model, saved
        summary = aggregate_results(results, args.test_masking_levels)
        summary['validation_selected_placement'] = selected['placement']
        summary['selected_configurations'] = selected_by_placement
        summary['resources'] = {
            ('baseline' if result['variant'] == 'baseline' else result['trial']['placement']):
            {'parameters': result['parameters'], 'fit_seconds': result['fit_seconds'],
             'test_seconds': result['test_seconds']} for result in results}
        write_json(output_dir / 'comparison.json', summary)
        log_event(output_dir, 'experiment_completed', selected=selected, comparison=summary)
        print(json.dumps(summary, indent=2))
        return summary
    except Exception as error:
        log_event(output_dir, 'experiment_failed', error_type=type(error).__name__, message=str(error))
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio_path', required=True)
    parser.add_argument('--visual_path', required=True)
    parser.add_argument('--output_dir', required=True)
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument('--trial-index', type=int,
                                help='Run only the indexed configuration from the declared search matrix')
    execution_mode.add_argument('--array-task', action='store_true',
                                help='Read trial index from SLURM_ARRAY_TASK_ID and train that configuration')
    execution_mode.add_argument('--finalize', action='store_true',
                                help='Select completed array trials by validation MRR and run final test evaluation')
    parser.add_argument('--ranks', nargs='+', type=int, default=[4, 8, 16, 32])
    parser.add_argument('--alpha_ratios', nargs='+', type=float, default=[0.25, 1.0, 2.0, 3.0])
    parser.add_argument('--seed', '--seeds', dest='seeds', nargs='+', type=int, default=[1],
                        help='One matched training seed (default: 1)')
    parser.add_argument('--placements', nargs='+', choices=SUBJECT_ADAPTER_PLACEMENTS,
                        default=list(SUBJECT_ADAPTER_PLACEMENTS))
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--pooling_strategy', choices=['cls', 'mean', 'max', 'multi'], default='cls')
    parser.add_argument('--eeg_arch', choices=['simple', 'complex', 'transformer'], default='transformer')
    parser.add_argument('--text_model', default='bert-base-uncased')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--training_masking_level', type=int, default=90)
    parser.add_argument('--primary_masking_level', type=int, default=90)
    parser.add_argument('--test_masking_levels', nargs='+', type=int, default=[0, 25, 50, 75, 90, 100])
    parser.add_argument('--hit_k', nargs='+', type=int, default=[1, 5, 10, 20])
    parser.add_argument('--max_text_len', type=int, default=256)
    parser.add_argument('--max_eeg_len', type=int, default=50)
    parser.add_argument('--balance_sources', action='store_true')
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)
    for name in ('epochs', 'patience', 'batch_size', 'hidden_dim', 'max_text_len', 'max_eeg_len'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.batch_size < 2 or args.warmup_epochs < 0 or not 0 < args.train_ratio < 1:
        parser.error('Require batch_size >= 2, warmup_epochs >= 0, and 0 < train_ratio < 1')
    if args.max_eeg_len > 50 and args.eeg_arch == 'transformer':
        parser.error('The existing transformer positional embedding supports max_eeg_len <= 50')
    for name in ('lr', 'temperature'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error('weight_decay must be finite and nonnegative')
    for name in ('ranks', 'alpha_ratios', 'hit_k'):
        values = getattr(args, name)
        if len(values) != len(set(values)) or any(not math.isfinite(value) or value <= 0 for value in values):
            parser.error(f'{name} must contain distinct positive finite values')
    if len(args.seeds) != 1:
        parser.error('This experiment uses exactly one training seed')
    if len(args.placements) != len(set(args.placements)):
        parser.error('placements must be distinct')
    if any(placement != 'input_only' for placement in args.placements) and args.eeg_arch != 'transformer':
        parser.error('Backbone/full EEG placements require --eeg_arch transformer')
    if 'full_eeg' in args.placements and args.pooling_strategy != 'cls':
        parser.error('Full EEG placement requires --pooling_strategy cls')
    levels = args.test_masking_levels + [args.training_masking_level, args.primary_masking_level]
    if any(not 0 <= level <= 100 for level in levels):
        parser.error('Masking levels must lie between 0 and 100')
    if len(args.test_masking_levels) != len(set(args.test_masking_levels)):
        parser.error('test_masking_levels must be distinct')
    return args


if __name__ == '__main__':
    cli_args = parse_args()
    if cli_args.array_task:
        if 'SLURM_ARRAY_TASK_ID' not in os.environ:
            raise RuntimeError('--array-task requires SLURM_ARRAY_TASK_ID')
        cli_args.trial_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    run_experiment(cli_args)