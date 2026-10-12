#!/usr/bin/env python3
"""
BM25 Text Retrieval Baseline for Brain Passage Retrieval - Per Dataset
Evaluates BM25 separately on Alice and Nieuwland datasets
Provides dataset-specific baseline comparison for neural EEG retrieval
"""

import numpy as np
import random
import argparse
from pathlib import Path
from typing import List, Dict
import wandb
from tqdm import tqdm

# BM25 import
from rank_bm25 import BM25Okapi

# Import your existing dataloader and utilities
from mv_dataloader import (
    DynamicMaskingDataloader,
    compute_global_eeg_dimensions
)
from transformers import AutoTokenizer


def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Set random seed to {seed}")


class BM25Retriever:
    """BM25 text retriever with preprocessing"""

    def __init__(self):
        self.bm25 = None
        self.documents = None
        print("BM25 retriever initialized")

    def fit(self, documents: List[str]):
        """Fit BM25 on document corpus"""
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents

    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for a query against all documents"""
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit() first.")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return scores.tolist()


def load_test_data(data_path: str, tokenizer, max_text_len: int = 256,
                   max_eeg_len: int = 50, dataset_type: str = 'auto',
                   holdout_subjects: bool = False, fold: int = None,
                   combined_dataset: dict = None, global_eeg_dims: tuple = None) -> DynamicMaskingDataloader:
    """Load test dataset"""

    print(f"Loading test data...")

    if combined_dataset is None:
        test_dataset = DynamicMaskingDataloader(
            data_path=data_path,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            max_eeg_len=max_eeg_len,
            train_ratio=0.8,
            debug=False,
            global_eeg_dims=global_eeg_dims,
            num_vectors=32,
            dataset_type=dataset_type,
            holdout_subjects=holdout_subjects,
            initial_masking_probability=0.9,
            split='test',
            fold=fold
        )
    else:
        test_dataset = DynamicMaskingDataloader(
            data_path=None,
            combined_dataset=combined_dataset,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            max_eeg_len=max_eeg_len,
            split='test',
            train_ratio=0.8,
            debug=False,
            global_eeg_dims=global_eeg_dims,
            num_vectors=32,
            dataset_type='original',
            holdout_subjects=holdout_subjects,
            initial_masking_probability=0.9,
            fold=fold
        )

    print(f"Loaded test dataset: {len(test_dataset)} samples from {len(test_dataset.unique_subjects)} subjects")
    return test_dataset


def split_test_data_by_dataset(test_dataset: DynamicMaskingDataloader, masking_level: int) -> Dict[str, Dict]:
    """
    Split test data by dataset source (Alice vs Nieuwland)
    Returns dict with dataset-specific queries, documents, and mappings
    """

    original_prob = test_dataset.get_current_masking_probability()
    test_dataset.set_masking_probability(masking_level / 100.0)

    try:
        dataset_data = {}  # dataset_name -> {queries, docs, mapping}

        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]

            query_text = sample['metadata']['query_text']
            doc_text = sample['metadata']['document_text']
            dataset_source = sample['metadata'].get('dataset_source', 'unknown')

            # Extract dataset name from source
            if 'nieuwland' in dataset_source.lower() or 'dataset_1' in dataset_source.lower():
                dataset_name = 'nieuwland'
            elif 'alice' in dataset_source.lower() or 'dataset_2' in dataset_source.lower():
                dataset_name = 'alice'
            else:
                dataset_name = 'unknown'

            # Initialize dataset entry if needed
            if dataset_name not in dataset_data:
                dataset_data[dataset_name] = {
                    'queries': [],
                    'unique_docs': {},  # text -> doc_idx
                    'query_to_doc_mapping': {}  # query_idx -> unique_doc_idx
                }

            data = dataset_data[dataset_name]
            query_idx = len(data['queries'])
            data['queries'].append(query_text)

            # Add document to unique set
            if doc_text.strip():
                if doc_text not in data['unique_docs']:
                    unique_doc_idx = len(data['unique_docs'])
                    data['unique_docs'][doc_text] = unique_doc_idx
                else:
                    unique_doc_idx = data['unique_docs'][doc_text]

                data['query_to_doc_mapping'][query_idx] = unique_doc_idx

        # Convert unique_docs dict to list
        for dataset_name in dataset_data:
            doc_list = [''] * len(dataset_data[dataset_name]['unique_docs'])
            for text, idx in dataset_data[dataset_name]['unique_docs'].items():
                doc_list[idx] = text
            dataset_data[dataset_name]['doc_list'] = doc_list
            del dataset_data[dataset_name]['unique_docs']

        return dataset_data

    finally:
        test_dataset.set_masking_probability(original_prob)


def generate_consistent_subsets(doc_list: List[str], query_to_doc_mapping: Dict[int, int],
                                subset_size: int = 100, seed: int = 42) -> Dict[int, List[int]]:
    """Generate consistent document subsets for fair comparison"""

    random.seed(seed)
    query_subsets = {}

    for query_idx, correct_doc_idx in query_to_doc_mapping.items():
        doc_subset_indices = [correct_doc_idx]
        negative_candidates = [i for i in range(len(doc_list)) if i != correct_doc_idx]

        if negative_candidates:
            random_negatives = random.sample(negative_candidates,
                                             min(subset_size - 1, len(negative_candidates)))
            doc_subset_indices.extend(random_negatives)

        query_subsets[query_idx] = doc_subset_indices

    return query_subsets


def compute_ranking_metrics(ranked_doc_indices: List[int], correct_doc_idx: int,
                            k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """Compute ranking metrics"""
    metrics = {}

    try:
        correct_rank = ranked_doc_indices.index(correct_doc_idx) + 1
    except ValueError:
        correct_rank = len(ranked_doc_indices) + 1

    metrics['rr'] = 1.0 / correct_rank if correct_rank <= len(ranked_doc_indices) else 0.0
    metrics['rank_of_correct'] = correct_rank

    for k in k_values:
        if k <= len(ranked_doc_indices):
            hit_at_k = 1.0 if correct_rank <= k else 0.0
            metrics[f'hit_at_{k}'] = hit_at_k
            metrics[f'precision_at_{k}'] = hit_at_k / k
            metrics[f'recall_at_{k}'] = hit_at_k
        else:
            metrics[f'hit_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0

    return metrics


def evaluate_bm25_ranking(bm25_retriever: BM25Retriever, queries: List[str],
                          doc_list: List[str], query_to_doc_mapping: Dict[int, int],
                          query_subsets: Dict[int, List[int]], dataset_name: str = "") -> List[Dict[str, float]]:
    """Evaluate BM25 ranking performance"""

    all_metrics = []
    desc = f"BM25 ranking ({dataset_name})" if dataset_name else "BM25 ranking"

    for query_idx, query_text in enumerate(tqdm(queries, desc=desc)):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        # Get BM25 scores
        all_scores = bm25_retriever.get_scores(query_text)
        subset_scores = [all_scores[idx] for idx in doc_subset_indices]

        # Rank documents
        doc_score_pairs = list(zip(doc_subset_indices, subset_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        ranked_indices = [doc_idx for doc_idx, score in doc_score_pairs]

        # Compute metrics
        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

    return all_metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    """Aggregate ranking metrics across queries"""

    if not all_metrics:
        return {}

    aggregated = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        aggregated[f'{prefix}/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            aggregated[f'{prefix}/{metric_name}_std'] = np.std(values)

    return aggregated


def handle_multiple_datasets(data_paths, dataset_types, max_eeg_len):
    """Handle loading and combining multiple datasets"""
    from mv_dataloader import load_combined_datasets, compute_combined_eeg_dimensions

    print(f"Loading {len(data_paths)} datasets for combination...")
    all_ict_pairs, combined_metadata = load_combined_datasets(data_paths, dataset_types)
    global_eeg_dims = compute_combined_eeg_dimensions(all_ict_pairs, max_eeg_len)

    combined_dataset = {'ict_pairs': all_ict_pairs, 'metadata': combined_metadata}
    return combined_dataset, 'original', global_eeg_dims


def initialize_wandb(config: Dict):
    """Initialize wandb logging"""

    dataset_name = config.get('dataset_name', 'unknown')
    holdout_subjects = config.get('holdout_subjects', False)
    fold = config.get('fold', None)

    if holdout_subjects and fold is not None:
        split_suffix = f"_holdout_fold{fold}"
    elif holdout_subjects:
        split_suffix = "_holdout"
    else:
        split_suffix = "_random"

    run_name = f"text_baseline_bm25_per_dataset{split_suffix}_{dataset_name}"

    tags = ['text-retrieval-baseline', 'bm25', 'brain-retrieval-comparison',
            'holdout-subjects' if holdout_subjects else 'random-split',
            f'dataset-{dataset_name}', 'multi-masking-evaluation', 'per-dataset']

    wandb.init(
        project="ECIR2026",
        name=run_name,
        config={
            'experiment_type': 'text_retrieval_baseline_bm25_per_dataset',
            'retrieval_method': 'bm25',
            'dataset_name': dataset_name,
            'holdout_subjects': holdout_subjects,
            'fold': fold,
            'split_method': 'holdout_subjects' if holdout_subjects else 'random_samples',
            'test_masking_levels': config.get('test_masking_levels', []),
            'test_samples': config.get('test_samples', 0),
            'test_subjects': config.get('test_subjects', 0),
            'baseline_type': 'text_retrieval_per_dataset',
            'seed': config.get('seed', 42),
        },
        tags=tags
    )


def main():
    parser = argparse.ArgumentParser(description='BM25 Text Retrieval Baseline - Per Dataset')

    # Data arguments
    parser.add_argument('--data_path', help='Path to ICT pairs .npy file')
    parser.add_argument('--data_paths', nargs='+', help='Paths to multiple ICT pairs .npy files')
    parser.add_argument('--dataset_type', default='auto', choices=['auto', 'original', 'nieuwland'])
    parser.add_argument('--dataset_types', nargs='*', default=None)

    # Test arguments
    parser.add_argument('--test_masking_levels', nargs='+', type=int,
                        default=[0, 25, 50, 75, 90, 100])
    parser.add_argument('--subset_size', type=int, default=100)

    # Experiment arguments
    parser.add_argument('--max_text_len', type=int, default=256)
    parser.add_argument('--max_eeg_len', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--holdout_subjects', action='store_true')
    parser.add_argument('--fold', type=int, choices=[1, 2, 3, 4, 5], default=None)

    args = parser.parse_args()

    if not args.data_path and not args.data_paths:
        raise ValueError("Must specify either --data_path or --data_paths")

    if args.fold is not None and not args.holdout_subjects:
        raise ValueError("--fold parameter requires --holdout_subjects")

    set_seeds(args.seed)

    # Handle datasets
    is_multi_dataset = args.data_paths and len(args.data_paths) > 1

    if is_multi_dataset:
        combined_dataset, dataset_type_to_use, global_eeg_dims = handle_multiple_datasets(
            args.data_paths, args.dataset_types, args.max_eeg_len)
        data_path_to_use = None
        dataset_name = "combined"
    else:
        data_path_to_use = args.data_path or args.data_paths[0]
        dataset_type_to_use = args.dataset_type
        global_eeg_dims = None
        combined_dataset = None

        filename = Path(data_path_to_use).name.lower()
        if 'nieuwland' in filename:
            dataset_name = "nieuwland"
        elif 'alice' in filename:
            dataset_name = "alice"
        else:
            dataset_name = "single"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load test dataset
    if combined_dataset is None:
        if global_eeg_dims is None:
            global_eeg_dims = compute_global_eeg_dimensions(data_path_to_use, args.max_eeg_len, dataset_type_to_use)

        test_dataset = load_test_data(
            data_path_to_use, tokenizer, args.max_text_len, args.max_eeg_len,
            dataset_type_to_use, args.holdout_subjects, args.fold,
            None, global_eeg_dims
        )
    else:
        test_dataset = load_test_data(
            None, tokenizer, args.max_text_len, args.max_eeg_len,
            'original', args.holdout_subjects, args.fold,
            combined_dataset, global_eeg_dims
        )

    # Configuration
    config = {
        'dataset_name': dataset_name,
        'holdout_subjects': args.holdout_subjects,
        'fold': args.fold,
        'test_masking_levels': args.test_masking_levels,
        'test_samples': len(test_dataset),
        'test_subjects': len(test_dataset.unique_subjects),
        'seed': args.seed,
        'subset_size': args.subset_size
    }

    initialize_wandb(config)

    print("\n" + "=" * 80)
    print("BM25 TEXT RETRIEVAL BASELINE - PER DATASET EVALUATION")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test subjects: {len(test_dataset.unique_subjects)}")
    print(f"Masking levels: {args.test_masking_levels}")
    print(f"Subset size: {args.subset_size}")

    # Evaluate at each masking level
    all_results = {}

    for masking_level in args.test_masking_levels:
        print(f"\n{'=' * 20} MASKING LEVEL: {masking_level}% {'=' * 20}")

        # Split test data by dataset
        dataset_data = split_test_data_by_dataset(test_dataset, masking_level)

        print(f"Found {len(dataset_data)} dataset(s):")
        for ds_name in dataset_data:
            n_queries = len(dataset_data[ds_name]['queries'])
            n_docs = len(dataset_data[ds_name]['doc_list'])
            print(f"  {ds_name}: {n_queries} queries, {n_docs} unique documents")

        # Evaluate each dataset separately
        for dataset_name_split, data in dataset_data.items():
            print(f"\n--- Evaluating {dataset_name_split} at {masking_level}% masking ---")

            queries = data['queries']
            doc_list = data['doc_list']
            query_to_doc_mapping = data['query_to_doc_mapping']

            if len(doc_list) == 0 or len(queries) == 0:
                print(f"  No valid data for {dataset_name_split} - skipping")
                continue

            # Generate consistent subsets
            query_subsets = generate_consistent_subsets(
                doc_list, query_to_doc_mapping, args.subset_size, args.seed
            )

            # Fit BM25 on this dataset's documents
            bm25_retriever = BM25Retriever()
            print(f"  Fitting BM25 on {len(doc_list)} documents...")
            bm25_retriever.fit(doc_list)

            # Evaluate
            bm25_metrics = evaluate_bm25_ranking(
                bm25_retriever, queries, doc_list, query_to_doc_mapping,
                query_subsets, dataset_name_split
            )

            # Aggregate results
            prefix = f'text_baseline/bm25/{dataset_name_split}/masking_{masking_level}'
            bm25_aggregated = aggregate_metrics(bm25_metrics, prefix)

            # Add metadata
            metadata = {
                f'{prefix}/num_queries': len(queries),
                f'{prefix}/num_unique_docs': len(doc_list),
                f'{prefix}/subset_size': args.subset_size,
                f'{prefix}/masking_level': masking_level
            }

            level_results = {**bm25_aggregated, **metadata}
            all_results.update(level_results)

            # Print summary
            if bm25_metrics:
                mrr = np.mean([m['rr'] for m in bm25_metrics])
                hit1 = np.mean([m['hit_at_1'] for m in bm25_metrics])
                hit5 = np.mean([m['hit_at_5'] for m in bm25_metrics])
                hit10 = np.mean([m['hit_at_10'] for m in bm25_metrics])

                print(f"  {dataset_name_split} Results:")
                print(f"    MRR: {mrr:.4f}, Hit@1: {hit1:.4f}, Hit@5: {hit5:.4f}, Hit@10: {hit10:.4f}")

    # Log all results to wandb
    if all_results:
        wandb.log(all_results)

        # Print comprehensive summary
        print(f"\n{'=' * 80}")
        print("BM25 BASELINE SUMMARY - PER DATASET")
        print(f"{'=' * 80}")

        # Determine which datasets we have
        datasets_found = set()
        for key in all_results.keys():
            if 'alice' in key:
                datasets_found.add('alice')
            if 'nieuwland' in key:
                datasets_found.add('nieuwland')

        for ds in sorted(datasets_found):
            print(f"\n{ds.upper()} Dataset:")
            print(f"{'Masking':>8} {'MRR':>8} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8} {'Docs':>8}")
            print("-" * 60)

            for masking_level in args.test_masking_levels:
                mrr_key = f'text_baseline/bm25/{ds}/masking_{masking_level}/rr'
                docs_key = f'text_baseline/bm25/{ds}/masking_{masking_level}/num_unique_docs'

                if mrr_key in all_results:
                    mrr = all_results[mrr_key]
                    hit1 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_1']
                    hit5 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_5']
                    hit10 = all_results[f'text_baseline/bm25/{ds}/masking_{masking_level}/hit_at_10']
                    n_docs = all_results.get(docs_key, 0)

                    print(f"{masking_level:>6}% {mrr:>8.4f} {hit1:>8.4f} {hit5:>8.4f} {hit10:>8.4f} {n_docs:>8.0f}")

    wandb.finish()
    print(f"\n{'=' * 80}")
    print("BM25 PER-DATASET BASELINE EVALUATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()