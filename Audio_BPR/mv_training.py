#!/usr/bin/env python3
"""
Simplified Training for Brain Passage Retrieval with DYNAMIC MULTI-MASKING VALIDATION
Uses single dataloader that changes masking probability on-the-fly
"""
import random
import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from mv_models import compute_similarity
from mv_dataloader import create_positive_negative_pairs


# ==========================================
# CROSS-ENCODER SPECIFIC FUNCTIONS
# ==========================================

def compute_bce_loss(scores, labels):
    """Compute BCE loss for cross-encoder"""
    return F.binary_cross_entropy_with_logits(scores.squeeze(), labels.float())


def cross_encoder_train_step(model, batch, optimizer, device, step_num, debug=False):
    """Training step for cross-encoder"""

    # Move to device
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}  # ADD THIS LINE
    docs = {k: v.to(device) for k, v in batch['docs'].items()}

    # Create positive pairs
    positive_labels, negative_pairs, negative_labels = create_positive_negative_pairs(batch)
    positive_labels = positive_labels.to(device)
    negative_labels = negative_labels.to(device)

    if debug:
        query_type = model.query_type
        print(f"[DEBUG] Cross-encoder training step {step_num} (query_type: {query_type})")
        print(f"  Positive pairs: {len(positive_labels)}")
        print(f"  Negative pairs: {len(negative_labels)}")

    # Forward pass for positive pairs
    positive_scores = model(eeg_queries, text_queries, docs)  # ADD text_queries
    positive_loss = compute_bce_loss(positive_scores, positive_labels)

    # Forward pass for negative pairs
    if len(negative_pairs) > 0:
        neg_eeg = torch.stack([eeg_queries[pair['eeg_idx']] for pair in negative_pairs])
        neg_text_queries = {  # ADD THIS BLOCK
            'input_ids': torch.stack([text_queries['input_ids'][pair['eeg_idx']] for pair in negative_pairs]),
            'attention_mask': torch.stack([text_queries['attention_mask'][pair['eeg_idx']] for pair in negative_pairs])
        }
        neg_docs = {
            'input_ids': torch.stack([docs['input_ids'][pair['doc_idx']] for pair in negative_pairs]),
            'attention_mask': torch.stack([docs['attention_mask'][pair['doc_idx']] for pair in negative_pairs])
        }

        negative_scores = model(neg_eeg, neg_text_queries, neg_docs)  # ADD neg_text_queries
        negative_loss = compute_bce_loss(negative_scores, negative_labels)

        total_loss = positive_loss + negative_loss

        # Compute negative accuracy too
        neg_acc = ((torch.sigmoid(negative_scores.squeeze()) > 0.5) == negative_labels.to(device)).float().mean()
    else:
        total_loss = positive_loss
        neg_acc = torch.tensor(0.0)

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Log metrics
    pos_acc = ((torch.sigmoid(positive_scores.squeeze()) > 0.5) == positive_labels).float().mean()

    wandb.log({
        'train/loss': total_loss.item(),
        'train/positive_loss': positive_loss.item(),
        'train/positive_accuracy': pos_acc.item(),
        'train/negative_accuracy': neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc,
        'train/grad_norm': grad_norm.item(),
        'train/step': step_num
    })

    if debug:
        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Positive accuracy: {pos_acc.item():.4f}")
        print(f"  Negative accuracy: {neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc:.4f}")

    return total_loss.item(), {'accuracy': pos_acc.item(), 'negative_accuracy': neg_acc.item() if torch.is_tensor(
        neg_acc) else neg_acc}, grad_norm.item()


def cross_encoder_validation_step(model, batch, device):
    """Validation step for cross-encoder"""

    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}  # ADD THIS LINE
    docs = {k: v.to(device) for k, v in batch['docs'].items()}

    # Create positive pairs for validation
    positive_labels, negative_pairs, negative_labels = create_positive_negative_pairs(batch)
    positive_labels = positive_labels.to(device)
    negative_labels = negative_labels.to(device)

    with torch.no_grad():
        # Forward pass for positive pairs
        positive_scores = model(eeg_queries, text_queries, docs)  # ADD text_queries
        positive_loss = compute_bce_loss(positive_scores, positive_labels)

        # Compute accuracy
        pos_acc = ((torch.sigmoid(positive_scores.squeeze()) > 0.5) == positive_labels).float().mean()

        # For negative pairs if any
        total_loss = positive_loss
        neg_acc = torch.tensor(0.0)
        if len(negative_pairs) > 0:
            neg_eeg = torch.stack([eeg_queries[pair['eeg_idx']] for pair in negative_pairs])
            neg_text_queries = {  # ADD THIS BLOCK
                'input_ids': torch.stack([text_queries['input_ids'][pair['eeg_idx']] for pair in negative_pairs]),
                'attention_mask': torch.stack(
                    [text_queries['attention_mask'][pair['eeg_idx']] for pair in negative_pairs])
            }
            neg_docs = {
                'input_ids': torch.stack([docs['input_ids'][pair['doc_idx']] for pair in negative_pairs]),
                'attention_mask': torch.stack([docs['attention_mask'][pair['doc_idx']] for pair in negative_pairs])
            }
            negative_scores = model(neg_eeg, neg_text_queries, neg_docs)  # ADD neg_text_queries
            negative_loss = compute_bce_loss(negative_scores, negative_labels)
            total_loss = positive_loss + negative_loss
            neg_acc = ((torch.sigmoid(negative_scores.squeeze()) > 0.5) == negative_labels.to(device)).float().mean()

    return total_loss.item(), {'accuracy': pos_acc.item(),
                               'negative_accuracy': neg_acc.item() if torch.is_tensor(neg_acc) else neg_acc}


# ==========================================
# DUAL-ENCODER SPECIFIC FUNCTIONS
# ==========================================

def compute_contrastive_loss(query_vectors, doc_vectors, pooling_strategy, temperature=0.07):
    """Compute contrastive loss for retrieval - works with any query type"""

    # Handle both list and tensor returns based on pooling strategy
    if pooling_strategy == 'multi':
        batch_size = len(query_vectors)  # CHANGED: use query_vectors instead of eeg_vectors
        device = query_vectors[0].device
    else:  # cls
        batch_size = query_vectors.size(0)  # CHANGED: use query_vectors instead of eeg_vectors
        device = query_vectors.device

    # Compute similarities between queries and documents
    query_to_doc_sims = []  # CHANGED: renamed from eeg_to_doc_sims
    for i in range(batch_size):
        if pooling_strategy == 'multi':
            query_i = query_vectors[i]  # CHANGED: use query_vectors
            doc_i = doc_vectors[i]
        else:  # cls
            query_i = query_vectors[i:i + 1]  # CHANGED: use query_vectors
            doc_i = doc_vectors[i:i + 1]

        sim = compute_similarity([query_i], [doc_i], pooling_strategy, temperature=1.0)
        query_to_doc_sims.append(sim[0])  # CHANGED: renamed

    similarities = torch.stack(query_to_doc_sims)

    # Create contrastive loss using in-batch negatives
    logits = torch.zeros(batch_size, batch_size, device=device)

    for i in range(batch_size):
        for j in range(batch_size):
            if pooling_strategy == 'multi':
                query_i = query_vectors[i]  # CHANGED: use query_vectors
                doc_j = doc_vectors[j]
            else:  # cls
                query_i = query_vectors[i:i + 1]  # CHANGED: use query_vectors
                doc_j = doc_vectors[j:j + 1]

            sim = compute_similarity([query_i], [doc_j], pooling_strategy, temperature=1.0)
            logits[i, j] = sim[0] / temperature

    # Labels: positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=device)

    # Cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss, similarities


def compute_alignment_metrics(query_vectors, doc_vectors, pooling_strategy, query_type='eeg'):
    """Compute alignment metrics between representations"""

    if pooling_strategy == 'multi':
        batch_size = len(query_vectors)
    else:  # cls
        batch_size = query_vectors.size(0)

    # Query-to-doc similarities (always computed)
    query_doc_sims = []
    for i in range(batch_size):
        if pooling_strategy == 'multi':
            query_i = query_vectors[i]
            doc_i = doc_vectors[i]
        else:  # cls
            query_i = query_vectors[i:i + 1]
            doc_i = doc_vectors[i:i + 1]

        sim = compute_similarity([query_i], [doc_i], pooling_strategy, temperature=1.0)
        query_doc_sims.append(sim[0].item())

    metrics = {
        'query_doc_similarity': np.mean(query_doc_sims),
        'query_doc_similarity_std': np.std(query_doc_sims),
    }

    # Add query type specific names for compatibility
    if query_type == 'eeg':
        metrics['eeg_query_similarity'] = metrics['query_doc_similarity']
        metrics['eeg_query_similarity_std'] = metrics['query_doc_similarity_std']
        metrics['eeg_doc_similarity'] = metrics['query_doc_similarity']
        metrics['eeg_doc_similarity_std'] = metrics['query_doc_similarity_std']
    else:  # text
        metrics['text_query_similarity'] = metrics['query_doc_similarity']
        metrics['text_query_similarity_std'] = metrics['query_doc_similarity_std']
        metrics['text_doc_similarity'] = metrics['query_doc_similarity']
        metrics['text_doc_similarity_std'] = metrics['query_doc_similarity_std']

    return metrics


def dual_encoder_train_step(model, batch, optimizer, device, step_num, debug=False):
    """Training step for dual encoder"""

    # Move batch to device
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}
    eeg_mv_masks = batch['eeg_mv_masks'].to(device)

    if debug:
        print(f"[DEBUG] Dual encoder training step {step_num} (query_type: {model.query_type})")
        print(f"  EEG queries: {eeg_queries.shape}")
        print(f"  Text query IDs: {text_queries['input_ids'].shape}")
        print(f"  Doc IDs: {docs['input_ids'].shape}")
        print(f"  Pooling strategy: {model.pooling_strategy}")

    # Forward pass
    outputs = model(eeg_queries, text_queries, docs, eeg_mv_masks)

    if debug:
        print(f"  Output types:")
        for key, value in outputs.items():
            if isinstance(value, list):
                print(f"    {key}: List with {len(value)} elements")
                if len(value) > 0:
                    print(f"      First element: {value[0].shape}")
            else:
                print(f"    {key}: {value.shape if value is not None else 'None'}")

    # Compute contrastive loss
    loss, query_sims = compute_contrastive_loss(
        outputs['query_vectors'],
        outputs['doc_vectors'],
        model.pooling_strategy
    )

    # Compute alignment metrics
    metrics = compute_alignment_metrics(
        outputs['query_vectors'],
        outputs['doc_vectors'],
        model.pooling_strategy,
        model.query_type
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Log to wandb - conditional on query type
    log_dict = {
        'train/loss': loss.item(),
        'train/query_doc_similarity': metrics['query_doc_similarity'],
        'train/grad_norm': grad_norm.item(),
        'train/step': step_num
    }

    # Add query-type specific metrics
    if model.query_type == 'eeg':
        log_dict.update({
            'train/eeg_doc_similarity': metrics['eeg_doc_similarity'],
            'train/eeg_query_similarity': metrics['eeg_query_similarity'],
        })
    else:  # text
        log_dict.update({
            'train/text_doc_similarity': metrics['text_doc_similarity'],
            'train/text_query_similarity': metrics['text_query_similarity'],
        })

    wandb.log(log_dict)

    if debug:
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Query-Doc similarity: {metrics['query_doc_similarity']:.4f}")
        print(f"  Grad norm: {grad_norm.item():.4f}")
        meta = batch['metadata'][0]
        print(f"  Sample query: '{meta['query_text'][:50]}...'")
        print(f"  Sample doc: '{meta['document_text'][:50]}...'")
        print(f"  Participant: {meta['participant_id']}")
        print(f"  Was masked: {meta.get('was_masked', 'N/A')}")
        print(f"  Masking probability: {meta.get('current_masking_probability', 'N/A')}")

    # Return with query-type appropriate metric name
    main_metric_key = 'eeg_query_similarity' if model.query_type == 'eeg' else 'text_query_similarity'
    return loss.item(), metrics, grad_norm.item()


def dual_encoder_validation_step(model, batch, device):
    """Validation step for dual encoder"""

    # Move batch to device
    eeg_queries = batch['eeg_queries'].to(device)
    text_queries = {k: v.to(device) for k, v in batch['text_queries'].items()}
    docs = {k: v.to(device) for k, v in batch['docs'].items()}
    eeg_mv_masks = batch['eeg_mv_masks'].to(device)

    # Forward pass (no gradients)
    with torch.no_grad():
        outputs = model(eeg_queries, text_queries, docs, eeg_mv_masks)

        # Compute loss and metrics
        loss, query_sims = compute_contrastive_loss(
            outputs['query_vectors'],
            outputs['doc_vectors'],
            model.pooling_strategy
        )

        metrics = compute_alignment_metrics(
            outputs['query_vectors'],  # CHANGED
            outputs['doc_vectors'],
            model.pooling_strategy,
            model.query_type  # ADD query_type parameter
        )

    return loss.item(), metrics


# ==========================================
# DISPATCH FUNCTIONS
# ==========================================

def train_step(model, batch, optimizer, device, step_num, debug=False):
    """Single training step - dispatches to dual or cross encoder"""

    if hasattr(model, 'cross_attention'):  # Cross-encoder
        return cross_encoder_train_step(model, batch, optimizer, device, step_num, debug)
    else:  # Dual encoder
        return dual_encoder_train_step(model, batch, optimizer, device, step_num, debug)


def validation_step(model, batch, device):
    """Single validation step - dispatches based on model type"""

    if hasattr(model, 'cross_attention'):  # Cross-encoder
        return cross_encoder_validation_step(model, batch, device)
    else:  # Dual encoder
        return dual_encoder_validation_step(model, batch, device)


# ==========================================
# RANKING VALIDATION (DUAL ENCODER ONLY)
# ==========================================

def build_document_database(val_dataloader):
    """Build unique document database for ranking evaluation"""

    print("Building document database for ranking evaluation...")

    unique_docs = {}  # text -> doc_info
    query_to_doc_mapping = {}  # query_idx -> unique_doc_idx
    query_idx = 0

    for batch in val_dataloader:
        for sample_idx, metadata in enumerate(batch['metadata']):
            doc_text = metadata['document_text'].strip()

            if doc_text:
                # Add to unique docs if not already present
                if doc_text not in unique_docs:
                    unique_doc_idx = len(unique_docs)
                    unique_docs[doc_text] = {
                        'idx': unique_doc_idx,
                        'text': doc_text,
                        'input_ids': batch['docs']['input_ids'][sample_idx].clone(),
                        'attention_mask': batch['docs']['attention_mask'][sample_idx].clone()
                    }
                else:
                    unique_doc_idx = unique_docs[doc_text]['idx']

                query_to_doc_mapping[query_idx] = unique_doc_idx
            query_idx += 1

    # Convert to list for easier batch processing
    doc_list = [None] * len(unique_docs)
    for text, doc_info in unique_docs.items():
        doc_list[doc_info['idx']] = doc_info

    print(f"Found {len(doc_list)} unique documents for {len(query_to_doc_mapping)} queries")
    return doc_list, query_to_doc_mapping


def generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size=100, seed=42):
    """Generate consistent document subsets for fair comparison between encoders"""

    print(f"Generating consistent document subsets (subset_size={subset_size}, seed={seed})...")

    # Set seed for reproducible subsets
    random.seed(seed)

    query_subsets = {}  # query_idx -> list of doc_indices

    for query_idx, correct_doc_idx in query_to_doc_mapping.items():
        # Always include correct document
        doc_subset_indices = [correct_doc_idx]

        # Sample random negatives
        negative_candidates = [i for i in range(len(doc_list)) if i != correct_doc_idx]
        if negative_candidates:
            random_negatives = random.sample(negative_candidates,
                                             min(subset_size - 1, len(negative_candidates)))
            doc_subset_indices.extend(random_negatives)

        query_subsets[query_idx] = doc_subset_indices

    print(f"Generated {len(query_subsets)} consistent subsets")
    return query_subsets


def batch_encode_documents(model, doc_list, device, batch_size=32):
    """Encode all unique documents in batches"""

    print(f"Batch encoding {len(doc_list)} unique documents (batch_size={batch_size})...")

    model.eval()
    all_doc_vectors = []

    with torch.no_grad():
        for i in range(0, len(doc_list), batch_size):
            batch_docs = doc_list[i:i + batch_size]

            # Stack document tensors for batch processing
            doc_input_ids = torch.stack([doc['input_ids'] for doc in batch_docs]).to(device)
            doc_attention_mask = torch.stack([doc['attention_mask'] for doc in batch_docs]).to(device)

            # Batch encode documents
            batch_doc_vectors = model.encode_text(doc_input_ids, doc_attention_mask)

            # Handle different pooling strategies
            if isinstance(batch_doc_vectors, list):
                # Multi-vector pooling
                all_doc_vectors.extend(batch_doc_vectors)
            else:
                # CLS pooling - split batch back to individual documents
                for j in range(len(batch_docs)):
                    all_doc_vectors.append(batch_doc_vectors[j:j + 1])

            # Progress update
            print(f"  Encoded {min(i + batch_size, len(doc_list))}/{len(doc_list)} documents")

    print(f"Document encoding complete. Got {len(all_doc_vectors)} document representations")
    return all_doc_vectors


def batch_similarity_computation(eeg_vectors, doc_vectors_list, pooling_strategy):
    """Compute similarities between one EEG query and all documents"""

    similarities = []

    if pooling_strategy == 'multi':
        # Multi-vector similarity computation
        for doc_vectors in doc_vectors_list:
            sim = compute_similarity([eeg_vectors], [doc_vectors], pooling_strategy, temperature=1.0)
            similarities.append(sim[0].item())

    elif pooling_strategy == 'cls':
        # CLS similarity - can be batched efficiently
        if len(doc_vectors_list) > 0:
            # Stack all document vectors for batch cosine similarity
            doc_stack = torch.stack([doc_vec[0] for doc_vec in doc_vectors_list])

            # Create batch input for similarity computation
            eeg_batch = eeg_vectors.repeat(len(doc_vectors_list), 1, 1)

            # Batch cosine similarity
            batch_similarities = compute_similarity(eeg_batch, doc_stack, pooling_strategy, temperature=1.0)
            similarities = batch_similarities.tolist()

    return similarities


def rank_documents_for_query(model, eeg_query, eeg_mv_mask, doc_vectors_list, pooling_strategy):
    """Rank pre-encoded documents for a single EEG query"""

    model.eval()

    with torch.no_grad():
        # Encode EEG query
        eeg_vectors = model.encode_eeg(eeg_query, eeg_mv_mask)

        # Handle different return types
        if isinstance(eeg_vectors, list):
            eeg_vectors = eeg_vectors[0]  # Get first (and only) element for single query
        else:
            eeg_vectors = eeg_vectors[0:1]  # Keep as batch of 1

        # Compute similarities efficiently
        doc_scores = batch_similarity_computation(eeg_vectors, doc_vectors_list, pooling_strategy)

    # Sort documents by score (descending)
    doc_indices_and_scores = list(enumerate(doc_scores))
    doc_indices_and_scores.sort(key=lambda x: x[1], reverse=True)

    ranked_doc_indices = [idx for idx, score in doc_indices_and_scores]
    ranked_scores = [score for idx, score in doc_indices_and_scores]

    return ranked_doc_indices, ranked_scores


def collect_eeg_queries(val_dataloader, device):
    """Collect all EEG queries and text queries from validation set"""

    print("Collecting EEG and text queries from validation set...")

    queries = []  # Changed name to be more general
    query_idx = 0

    for batch in val_dataloader:
        batch_size = batch['eeg_queries'].size(0)
        for sample_idx in range(batch_size):
            eeg_query = batch['eeg_queries'][sample_idx:sample_idx + 1].to(device)
            eeg_mv_mask = batch['eeg_mv_masks'][sample_idx:sample_idx + 1].to(device)
            text_query = {  # ADD THIS BLOCK
                'input_ids': batch['text_queries']['input_ids'][sample_idx:sample_idx + 1].to(device),
                'attention_mask': batch['text_queries']['attention_mask'][sample_idx:sample_idx + 1].to(device)
            }
            queries.append((eeg_query, eeg_mv_mask, text_query))  # ADD text_query
            query_idx += 1

    print(f"Collected {len(queries)} query sets (EEG + text)")
    return queries


def subject_key(metadata):
    """Namespace a participant identity by its source dataset."""
    return json.dumps([str(metadata['dataset_source']), str(metadata['participant_id'])], separators=(',', ':'))


def batch_subject_ids(batch, subject_mapping, device):
    keys = [subject_key(metadata) for metadata in batch['metadata']]
    unknown = set(keys) - set(subject_mapping)
    if unknown:
        raise ValueError(f"Unseen subjects are not supported: {sorted(unknown)}")
    return torch.tensor([subject_mapping[key] for key in keys], dtype=torch.long, device=device)


def evaluate_unified_index(model, dataloader, device, subject_mapping,
                           masking_level=90, k_values=(1, 5, 10, 20)):
    """Rank every query against the same complete mixed-source passage index."""
    if not 0 <= masking_level <= 100 or not k_values or any(cutoff < 1 for cutoff in k_values):
        raise ValueError("Masking must be 0..100 and Hit@k cutoffs must be positive")
    previous_mode = model.training
    previous_masking = dataloader.dataset.get_current_masking_probability()
    random_state = random.getstate()
    torch_state = torch.get_rng_state()
    model.eval()
    dataloader.dataset.set_masking_probability(masking_level / 100.0)
    try:
        documents, document_ids, queries = [], {}, []
        with torch.no_grad():
            for batch in dataloader:
                ids = batch_subject_ids(batch, subject_mapping, device)
                vectors = model.encode_eeg(batch['eeg_queries'].to(device),
                                           batch['eeg_mv_masks'].to(device), subject_ids=ids)
                for index, metadata in enumerate(batch['metadata']):
                    text = metadata['document_text'].strip()
                    if not text:
                        raise ValueError("Every evaluation query must have a nonempty positive passage")
                    if text not in document_ids:
                        document_ids[text] = len(documents)
                        documents.append({
                            'text': text,
                            'input_ids': batch['docs']['input_ids'][index].clone(),
                            'attention_mask': batch['docs']['attention_mask'][index].clone(),
                        })
                    queries.append({
                        'vector': vectors[index].detach().cpu(),
                        'document_id': document_ids[text],
                        'subject': subject_key(metadata),
                        'source': metadata['dataset_source'],
                        'original_idx': metadata['original_idx'],
                    })
            if not queries:
                raise ValueError("The combined evaluation split is empty")
            document_vectors = batch_encode_documents(model, documents, device, dataloader.batch_size)
            if model.pooling_strategy != 'multi':
                document_matrix = F.normalize(
                    torch.stack([vector.reshape(-1) for vector in document_vectors]), dim=-1)
            rows, grouped = [], {}
            for query in queries:
                vector = query['vector'].to(device)
                if model.pooling_strategy == 'multi':
                    scores = [compute_similarity([vector], [doc_vector], 'multi', temperature=1.0)[0].item()
                              for doc_vector in document_vectors]
                else:
                    scores = (document_matrix @ F.normalize(vector.reshape(-1), dim=-1)).tolist()
                ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
                rank = ranked.index(query['document_id']) + 1
                metrics = {'mrr': 1.0 / rank}
                metrics.update({f'hit_at_{cutoff}': float(rank <= cutoff) for cutoff in k_values})
                row = {key: value for key, value in query.items() if key != 'vector'}
                row.update({'rank': rank, **metrics})
                rows.append(row)
                grouped.setdefault(query['source'], []).append(metrics)
            metric_names = ['mrr'] + [f'hit_at_{cutoff}' for cutoff in k_values]
            fingerprint_data = {
                'passages': [document['text'] for document in documents],
                'qrels': [{key: query[key] for key in ('subject', 'source', 'original_idx', 'document_id')}
                          for query in queries],
            }
            fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
            return {
                'masking_level': masking_level,
                'candidate_scope': 'combined_all_sources',
                'num_documents': len(documents),
                'num_queries': len(queries),
                'index_fingerprint': fingerprint,
                'overall': {name: float(np.mean([row[name] for row in rows])) for name in metric_names},
                'by_source': {source: {name: float(np.mean([row[name] for row in values]))
                                       for name in metric_names} for source, values in grouped.items()},
                'query_results': rows,
            }
    finally:
        dataloader.dataset.set_masking_probability(previous_masking)
        model.train(previous_mode)
        random.setstate(random_state)
        torch.set_rng_state(torch_state)


def compute_ranking_metrics(ranked_doc_indices, correct_doc_idx, k_values=[1, 5, 10, 20]):
    """
    Compute comprehensive ranking metrics for a single query

    Args:
        ranked_doc_indices: List of document indices ranked by relevance (best first)
        correct_doc_idx: Index of the correct/relevant document
        k_values: List of K values for Hit@K, Precision@K, Recall@K

    Returns:
        dict: Comprehensive ranking metrics
    """
    metrics = {}

    # Find rank of correct document (1-indexed)
    try:
        correct_rank = ranked_doc_indices.index(correct_doc_idx) + 1
    except ValueError:
        correct_rank = len(ranked_doc_indices) + 1  # Not found

    # Mean Reciprocal Rank
    metrics['rr'] = 1.0 / correct_rank if correct_rank <= len(ranked_doc_indices) else 0.0
    metrics['rank_of_correct'] = correct_rank

    # Hit@K, Precision@K, and Recall@K for all K values
    for k in k_values:
        if ranked_doc_indices:
            # Hit@K: is correct doc in top-k?
            hit_at_k = 1.0 if correct_rank <= k else 0.0
            metrics[f'hit_at_{k}'] = hit_at_k

            # Precision@K: for single relevant doc, precision = hit@k / k
            precision_at_k = hit_at_k / k
            metrics[f'precision_at_{k}'] = precision_at_k

            # Recall@K: for single relevant doc, recall = hit@k (since there's only 1 relevant doc)
            recall_at_k = hit_at_k
            metrics[f'recall_at_{k}'] = recall_at_k
        else:
            metrics[f'hit_at_{k}'] = 0.0
            metrics[f'precision_at_{k}'] = 0.0
            metrics[f'recall_at_{k}'] = 0.0

    return metrics


# ==========================================
# NEW: DYNAMIC MULTI-MASKING VALIDATION FUNCTIONS
# ==========================================

def validate_single_masking_level(model, val_dataloader, device, masking_probability, masking_level_name,
                                  is_primary=False):
    """Validate model on a single masking level using DYNAMIC masking"""

    # DYNAMIC: Change masking probability on the fly
    original_probability = val_dataloader.dataset.get_current_masking_probability()
    val_dataloader.dataset.set_masking_probability(masking_probability)

    try:
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        # Determine if cross-encoder for metric tracking
        is_cross_encoder = hasattr(model, 'cross_attention')

        # Accumulate validation metrics
        if is_cross_encoder:
            val_accuracies = []
            val_neg_accuracies = []
        else:
            val_similarities = []
            val_query_doc_sims = []
            val_eeg_doc_sims = []

        for batch_idx, batch in enumerate(val_dataloader):
            val_loss, val_metrics = validation_step(model, batch, device)

            total_val_loss += val_loss
            num_val_batches += 1

            if is_cross_encoder:
                val_accuracies.append(val_metrics['accuracy'])
                val_neg_accuracies.append(val_metrics['negative_accuracy'])
            else:
                # Use appropriate similarity key based on query type
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    similarity_key = 'text_query_similarity'
                else:
                    similarity_key = 'eeg_query_similarity'

                val_similarities.append(val_metrics[similarity_key])
                val_query_doc_sims.append(val_metrics['query_doc_similarity'])

                # Use appropriate doc similarity key
                doc_similarity_key = 'text_doc_similarity' if model.query_type == 'text' else 'eeg_doc_similarity'
                val_eeg_doc_sims.append(val_metrics[doc_similarity_key])

        # Compute validation statistics
        avg_val_loss = total_val_loss / num_val_batches

        if is_cross_encoder:
            avg_val_accuracy = np.mean(val_accuracies)
            avg_val_neg_accuracy = np.mean(val_neg_accuracies)
            main_metric = avg_val_accuracy

            metrics_dict = {
                'loss': avg_val_loss,
                'accuracy': avg_val_accuracy,
                'negative_accuracy': avg_val_neg_accuracy
            }
        else:
            avg_val_similarity = np.mean(val_similarities)
            avg_query_doc_sim = np.mean(val_query_doc_sims)
            avg_eeg_doc_sim = np.mean(val_eeg_doc_sims)
            main_metric = avg_val_similarity

            metrics_dict = {
                'loss': avg_val_loss,
                'query_doc_similarity': avg_query_doc_sim,
                'eeg_doc_similarity': avg_eeg_doc_sim
            }

            # Add query-type specific metrics
            if hasattr(model, 'query_type') and model.query_type == 'text':
                metrics_dict.update({
                    'text_query_similarity': avg_val_similarity,
                    'text_query_similarity_std': np.std(val_similarities),
                    'text_doc_similarity': avg_eeg_doc_sim,
                })
            else:
                metrics_dict.update({
                    'eeg_query_similarity': avg_val_similarity,
                    'eeg_query_similarity_std': np.std(val_similarities),
                    'eeg_doc_similarity': avg_eeg_doc_sim,
                })

        return metrics_dict, main_metric

    finally:
        # DYNAMIC: Always restore original masking probability
        val_dataloader.dataset.set_masking_probability(original_probability)


def perform_ranking_validation(model, val_dataloader, device, epoch_num, subset_size=None):
    """Enhanced ranking validation with comprehensive metrics and organized logging"""

    # Detect dataset sources for potential dataset-specific evaluation
    dataset_sources_info = detect_dataset_sources(val_dataloader)

    # Main combined validation
    print(f"\n=== VALIDATION RANKING (Epoch {epoch_num}) ===")

    doc_list, query_to_doc_mapping = build_document_database(val_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for ranking evaluation")
        return {}

    # Handle subset size
    if subset_size is None:
        subset_size = len(doc_list)
        print(f"No subset_size specified, using all {subset_size} documents")
    elif len(doc_list) < subset_size:
        print(f"Warning: Only {len(doc_list)} documents available, using all")
        subset_size = len(doc_list)

    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(val_dataloader, device)

    print(f"Dual-encoder ranking: {len(queries)} queries against {subset_size} documents each")
    print(f"Query type: {model.query_type}")

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_dual_encoder_subset(
            model, eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} queries...")

    if not all_metrics:
        print("Warning: No metrics computed")
        return {}

    # Aggregate main ranking metrics
    main_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        main_metrics[f'val_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':  # Don't compute std for rank
            main_metrics[f'val_ranking/main/{metric_name}_std'] = np.std(values)

    # Add metadata
    main_metrics.update({
        'val_ranking/main/num_unique_documents': len(doc_list),
        'val_ranking/main/num_queries_evaluated': len(all_metrics),
        'val_ranking/main/epoch_num': epoch_num,
        'val_ranking/main/subset_size': subset_size,
        'val_ranking/main/query_type': model.query_type
    })

    # Print main summary
    print(f"Main Ranking Results:")
    print(f"  Query Type: {model.query_type}")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {main_metrics['val_ranking/main/rr']:.4f}")
    print(f"  Hit@1: {main_metrics['val_ranking/main/hit_at_1']:.4f}")
    print(f"  Hit@5: {main_metrics['val_ranking/main/hit_at_5']:.4f}")
    print(f"  Hit@10: {main_metrics['val_ranking/main/hit_at_10']:.4f}")
    print(f"  Hit@20: {main_metrics['val_ranking/main/hit_at_20']:.4f}")

    # Dataset-specific ranking validation if multiple datasets
    dataset_metrics = {}
    if dataset_sources_info['has_multiple']:
        print(f"\n=== DATASET-SPECIFIC VALIDATION RANKING ===")
        dataset_metrics = perform_dataset_specific_ranking(
            model, val_dataloader, device, epoch_num, subset_size, prefix='val_ranking/dataset'
        )

    # Log all metrics to wandb
    all_wandb_metrics = {**main_metrics, **dataset_metrics}
    wandb.log(all_wandb_metrics)

    print("=" * 60)
    return all_wandb_metrics



def split_validation_data_by_dataset(val_dataloader):
    """Split validation data by dataset source for dataset-specific evaluation"""

    dataset_batches = {}  # dataset_name -> list of samples

    print("Splitting validation data by dataset source...")

    for batch in val_dataloader:
        batch_size = len(batch['metadata'])

        for sample_idx in range(batch_size):
            metadata = batch['metadata'][sample_idx]
            dataset_source = metadata.get('dataset_source', 'unknown')

            # More robust dataset name extraction
            if 'nieuwland' in dataset_source.lower():
                dataset_name = 'nieuwland'
            elif 'alice' in dataset_source.lower():
                dataset_name = 'alice'
            elif 'authentic_query' in dataset_source.lower() or 'task' in dataset_source.lower():
                dataset_name = 'authentic_query'
            elif 'dataset_1' in dataset_source.lower():
                # Based on your script: dataset_1 = nieuwland
                dataset_name = 'nieuwland'
            elif 'dataset_2' in dataset_source.lower():
                # Based on your script: dataset_2 = alice
                dataset_name = 'alice'
            else:
                # Debug info to see what we're actually getting
                print(f"DEBUG: Unknown dataset_source: '{dataset_source}'")
                dataset_name = 'unknown'

            if dataset_name not in dataset_batches:
                dataset_batches[dataset_name] = []

            # Extract sample from batch
            sample = {
                'eeg_query': batch['eeg_queries'][sample_idx:sample_idx + 1],
                'text_query': {
                    'input_ids': batch['text_queries']['input_ids'][sample_idx:sample_idx + 1],
                    'attention_mask': batch['text_queries']['attention_mask'][sample_idx:sample_idx + 1]
                },
                'doc': {
                    'input_ids': batch['docs']['input_ids'][sample_idx:sample_idx + 1],
                    'attention_mask': batch['docs']['attention_mask'][sample_idx:sample_idx + 1]
                },
                'eeg_mv_mask': batch['eeg_mv_masks'][sample_idx:sample_idx + 1],
                'metadata': [metadata]
            }

            dataset_batches[dataset_name].append(sample)

    # Print summary
    for dataset_name, samples in dataset_batches.items():
        print(f"  {dataset_name}: {len(samples)} samples")

    return dataset_batches


def validate_dataset_specific(model, dataset_batches, device, primary_masking_level=90):
    """Run validation on dataset-specific subsets at primary masking level"""

    primary_masking_prob = primary_masking_level / 100.0
    is_cross_encoder = hasattr(model, 'cross_attention')

    dataset_results = {}

    print(f"\n=== DATASET-SPECIFIC VALIDATION (Masking: {primary_masking_level}%) ===")

    for dataset_name, samples in dataset_batches.items():
        if len(samples) == 0:
            continue

        print(f"  Validating on {dataset_name} dataset ({len(samples)} samples)...")

        model.eval()
        total_loss = 0
        num_samples = 0

        if is_cross_encoder:
            accuracies = []
            neg_accuracies = []
        else:
            similarities = []
            query_doc_sims = []
            eeg_doc_sims = []

        # Process samples in small batches for memory efficiency
        batch_size = 16
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]

            # Reconstruct batch format
            mini_batch = {
                'eeg_queries': torch.cat([s['eeg_query'] for s in batch_samples], dim=0),
                'text_queries': {
                    'input_ids': torch.cat([s['text_query']['input_ids'] for s in batch_samples], dim=0),
                    'attention_mask': torch.cat([s['text_query']['attention_mask'] for s in batch_samples], dim=0)
                },
                'docs': {
                    'input_ids': torch.cat([s['doc']['input_ids'] for s in batch_samples], dim=0),
                    'attention_mask': torch.cat([s['doc']['attention_mask'] for s in batch_samples], dim=0)
                },
                'eeg_mv_masks': torch.cat([s['eeg_mv_mask'] for s in batch_samples], dim=0),
                'metadata': [s['metadata'][0] for s in batch_samples]
            }

            # Run validation step
            loss, metrics = validation_step(model, mini_batch, device)

            total_loss += loss * len(batch_samples)
            num_samples += len(batch_samples)

            if is_cross_encoder:
                accuracies.append(metrics['accuracy'])
                neg_accuracies.append(metrics['negative_accuracy'])
            else:
                # Use appropriate similarity key based on query type
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    similarity_key = 'text_query_similarity'
                    doc_similarity_key = 'text_doc_similarity'
                else:
                    similarity_key = 'eeg_query_similarity'
                    doc_similarity_key = 'eeg_doc_similarity'

                similarities.append(metrics[similarity_key])
                query_doc_sims.append(metrics['query_doc_similarity'])
                eeg_doc_sims.append(metrics[doc_similarity_key])

        # Compute averages
        avg_loss = total_loss / num_samples

        if is_cross_encoder:
            avg_accuracy = np.mean(accuracies)
            avg_neg_accuracy = np.mean(neg_accuracies)
            main_metric = avg_accuracy

            result_metrics = {
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'negative_accuracy': avg_neg_accuracy,
                'num_samples': num_samples
            }

            print(f"    {dataset_name}: Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.4f}")

        else:
            avg_similarity = np.mean(similarities)
            avg_query_doc_sim = np.mean(query_doc_sims)
            avg_eeg_doc_sim = np.mean(eeg_doc_sims)
            main_metric = avg_similarity

            result_metrics = {
                'loss': avg_loss,
                'query_doc_similarity': avg_query_doc_sim,
                'num_samples': num_samples
            }

            # Add query-type specific metrics
            if hasattr(model, 'query_type') and model.query_type == 'text':
                result_metrics.update({
                    'text_query_similarity': avg_similarity,
                    'text_doc_similarity': avg_eeg_doc_sim
                })
                print(f"    {dataset_name}: Loss {avg_loss:.4f}, Text-Query Sim {avg_similarity:.4f}")
            else:
                result_metrics.update({
                    'eeg_query_similarity': avg_similarity,
                    'eeg_doc_similarity': avg_eeg_doc_sim
                })
                print(f"    {dataset_name}: Loss {avg_loss:.4f}, EEG-Query Sim {avg_similarity:.4f}")

        dataset_results[dataset_name] = {
            'metrics': result_metrics,
            'main_metric': main_metric
        }

    print("=" * 60)
    return dataset_results

def rank_dual_encoder_subset(model, eeg_query, eeg_mv_mask, doc_list, doc_indices, device, text_query=None):
    """EFFICIENT: Batch encode all documents at once for dual-encoder"""
    model.eval()

    with torch.no_grad():
        # Encode query once (based on model's query type)
        if model.query_type == 'eeg':
            eeg_output = model.encode_eeg(eeg_query, eeg_mv_mask)
            query_vectors = eeg_output
        else:  # text
            if text_query is None:
                raise ValueError("text_query is required when model.query_type='text'")
            query_vectors = model.encode_text(text_query['input_ids'], text_query['attention_mask'])

        # Handle different return types
        if isinstance(query_vectors, list):
            query_vectors = query_vectors[0]  # Get first element for single query
        else:
            query_vectors = query_vectors[0:1]  # Keep as batch of 1

        # BATCH encode ALL documents in the subset at once
        doc_input_ids = torch.stack([doc_list[doc_idx]['input_ids'] for doc_idx in doc_indices]).to(device)
        doc_attention_masks = torch.stack([doc_list[doc_idx]['attention_mask'] for doc_idx in doc_indices]).to(device)

        # Single batch encoding for ALL documents
        batch_doc_vectors = model.encode_text(doc_input_ids, doc_attention_masks)

        # Handle different pooling strategies
        if isinstance(batch_doc_vectors, list):
            # Multi-vector pooling
            doc_vectors_list = batch_doc_vectors
        else:
            # CLS pooling - split batch back to individual documents
            doc_vectors_list = [batch_doc_vectors[i:i + 1] for i in range(batch_doc_vectors.size(0))]

        # Compute similarities in batch
        scores = []
        for i, doc_vectors in enumerate(doc_vectors_list):
            sim = compute_similarity([query_vectors], [doc_vectors], model.pooling_strategy, temperature=1.0)
            scores.append((doc_indices[i], sim[0].item()))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked_indices = [doc_idx for doc_idx, score in scores]
    ranked_scores = [score for doc_idx, score in scores]

    return ranked_indices, ranked_scores


# REPLACE the existing perform_cross_encoder_ranking_validation function with this:
def perform_cross_encoder_ranking_validation(model, val_dataloader, device, epoch_num, subset_size=100):
    """Enhanced cross-encoder ranking validation with comprehensive metrics"""

    # Detect dataset sources
    dataset_sources_info = detect_dataset_sources(val_dataloader)

    # Main validation
    print(f"\n=== CROSS-ENCODER VALIDATION RANKING (Epoch {epoch_num}) ===")

    doc_list, query_to_doc_mapping = build_document_database(val_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for ranking evaluation")
        return {}

    if len(doc_list) < subset_size:
        print(f"Warning: Only {len(doc_list)} documents available, using all")
        subset_size = len(doc_list)

    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(val_dataloader, device)

    print(f"Cross-encoder ranking: {len(queries)} queries against {subset_size} documents each")

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_cross_encoder_subset(
            model, eeg_query, text_query, doc_list, doc_subset_indices, device
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} queries...")

    if not all_metrics:
        print("Warning: No metrics computed")
        return {}

    # Aggregate main ranking metrics
    main_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        main_metrics[f'val_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            main_metrics[f'val_ranking/main/{metric_name}_std'] = np.std(values)

    # Add metadata
    main_metrics.update({
        'val_ranking/main/num_unique_documents': len(doc_list),
        'val_ranking/main/num_queries_evaluated': len(all_metrics),
        'val_ranking/main/epoch_num': epoch_num,
        'val_ranking/main/subset_size': subset_size,
        'val_ranking/main/encoder_type': 'cross'
    })

    # Print main summary
    print(f"Cross-Encoder Main Ranking Results:")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {main_metrics['val_ranking/main/rr']:.4f}")
    print(f"  Hit@1: {main_metrics['val_ranking/main/hit_at_1']:.4f}")
    print(f"  Hit@5: {main_metrics['val_ranking/main/hit_at_5']:.4f}")
    print(f"  Hit@10: {main_metrics['val_ranking/main/hit_at_10']:.4f}")
    print(f"  Hit@20: {main_metrics['val_ranking/main/hit_at_20']:.4f}")

    # Dataset-specific ranking validation if multiple datasets
    dataset_metrics = {}
    if dataset_sources_info['has_multiple']:
        print(f"\n=== DATASET-SPECIFIC VALIDATION RANKING ===")
        dataset_metrics = perform_dataset_specific_ranking(
            model, val_dataloader, device, epoch_num, subset_size, prefix='val_ranking/dataset'
        )

    # Log all metrics to wandb
    all_wandb_metrics = {**main_metrics, **dataset_metrics}
    wandb.log(all_wandb_metrics)

    print("=" * 60)
    return all_wandb_metrics


def rank_cross_encoder_subset(model, eeg_query, text_query, doc_list, doc_indices, device):
    """EFFICIENT: Batch process all documents at once for cross-encoder"""
    model.eval()

    with torch.no_grad():
        # Stack all documents in the subset for batch processing
        doc_input_ids = torch.stack([doc_list[doc_idx]['input_ids'] for doc_idx in doc_indices]).to(device)
        doc_attention_masks = torch.stack([doc_list[doc_idx]['attention_mask'] for doc_idx in doc_indices]).to(device)

        batch_size = len(doc_indices)

        # Replicate query for each document in the batch
        batch_eeg_queries = eeg_query.repeat(batch_size, 1, 1, 1)  # [batch_size, words, time, channels]
        batch_text_queries = {
            'input_ids': text_query['input_ids'].repeat(batch_size, 1),
            'attention_mask': text_query['attention_mask'].repeat(batch_size, 1)
        }
        batch_docs = {
            'input_ids': doc_input_ids,
            'attention_mask': doc_attention_masks
        }

        # Single forward pass for ALL documents at once
        batch_scores = model(batch_eeg_queries, batch_text_queries, batch_docs)
        scores = batch_scores.squeeze().cpu().tolist()

        # Handle single document case
        if isinstance(scores, float):
            scores = [scores]

    # Create (doc_idx, score) pairs and sort
    doc_score_pairs = list(zip(doc_indices, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    ranked_indices = [doc_idx for doc_idx, score in doc_score_pairs]
    ranked_scores = [score for doc_idx, score in doc_score_pairs]

    return ranked_indices, ranked_scores


# Helper function for dataset source detection
def detect_dataset_sources(val_dataloader):
    """Detect which dataset sources are present in validation data"""
    dataset_sources = set()

    for batch in val_dataloader:
        for metadata in batch['metadata']:
            dataset_source = metadata.get('dataset_source', 'unknown')
            dataset_sources.add(dataset_source)

    dataset_sources = list(dataset_sources)
    print(f"Detected dataset sources: {dataset_sources}")

    # Check if we have multiple datasets
    has_multiple_datasets = len(dataset_sources) > 1

    return {
        'all_sources': dataset_sources,
        'has_multiple': has_multiple_datasets,
    }


# ==========================================
# MAIN TRAINING FUNCTIONS WITH DYNAMIC MULTI-MASKING
# ==========================================

def train_epoch(model, dataloader, optimizer, device, epoch_num, total_epochs, debug=False):
    """Train for a single epoch"""

    model.train()
    total_loss = 0
    num_batches = 0

    # Determine if cross-encoder for metric tracking
    is_cross_encoder = hasattr(model, 'cross_attention')

    # Accumulate metrics
    if is_cross_encoder:
        epoch_accuracies = []
    else:
        epoch_similarities = []

    epoch_grad_norms = []

    for batch_idx, batch in enumerate(dataloader):
        # Debug first batch of first epoch
        debug_this_batch = debug and epoch_num == 1 and batch_idx == 0

        # Calculate global step
        step_num = (epoch_num - 1) * len(dataloader) + batch_idx

        # Training step
        loss, metrics, grad_norm = train_step(
            model, batch, optimizer, device, step_num, debug=debug_this_batch
        )

        total_loss += loss
        num_batches += 1

        if is_cross_encoder:
            epoch_accuracies.append(metrics['accuracy'])
        else:
            # Use appropriate similarity metric based on query type
            if hasattr(model, 'query_type') and model.query_type == 'text':
                similarity_key = 'text_query_similarity'
                display_name = 'Text-Query Sim'
            else:
                similarity_key = 'eeg_query_similarity'
                display_name = 'EEG-Query Sim'

            epoch_similarities.append(metrics[similarity_key])

        epoch_grad_norms.append(grad_norm)

        # Progress logging with appropriate labels
        if batch_idx % 20 == 0:
            if is_cross_encoder:
                print(f"Epoch {epoch_num}/{total_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_display = f"Text-Query Sim: {metrics['text_query_similarity']:.4f}"
                else:
                    sim_display = f"EEG-Query Sim: {metrics['eeg_query_similarity']:.4f}"
                print(f"Epoch {epoch_num}/{total_epochs}, Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss:.4f}, {sim_display}")

    # Compute epoch statistics
    avg_loss = total_loss / num_batches
    if is_cross_encoder:
        avg_metric = np.mean(epoch_accuracies)
        metric_name = "Accuracy"
    else:
        avg_metric = np.mean(epoch_similarities)
        if hasattr(model, 'query_type') and model.query_type == 'text':
            metric_name = "Text-Query Sim"
        else:
            metric_name = "EEG-Query Sim"

    avg_grad_norm = np.mean(epoch_grad_norms)

    print(f"Epoch {epoch_num} training completed. "
          f"Avg Loss: {avg_loss:.4f}, Avg {metric_name}: {avg_metric:.4f}")

    return avg_loss, avg_metric, avg_grad_norm


def validate_epoch(model, val_dataloader, device, epoch_num,
                   enable_multi_masking_validation=False,
                   validation_masking_levels=[0, 25, 50, 75, 90, 100],
                   multi_masking_frequency=3,
                   primary_masking_level=90):
    """
    Enhanced validate_epoch with DYNAMIC multi-masking support + dataset-specific validation

    Args:
        model: The model to validate
        val_dataloader: Single validation dataloader with dynamic masking capability
        device: Device to run validation on
        epoch_num: Current epoch number
        enable_multi_masking_validation: Whether to enable multi-masking validation
        validation_masking_levels: List of masking percentages to test
        multi_masking_frequency: Run multi-masking validation every N epochs
        primary_masking_level: Masking level to use for early stopping (default: 90)
    """

    # Always validate on primary masking level (90% by default)
    print(f"Running standard validation for epoch {epoch_num}...")

    primary_masking_prob = primary_masking_level / 100.0
    primary_metrics, primary_main_metric = validate_single_masking_level(
        model, val_dataloader, device, primary_masking_prob, f"masking_{primary_masking_level}%", is_primary=True
    )

    # Log primary validation metrics to wandb
    log_dict = {}
    for metric_name, value in primary_metrics.items():
        log_dict[f'val/{metric_name}'] = value
    log_dict['val/epoch_num'] = epoch_num

    wandb.log(log_dict)

    # Determine if cross-encoder for display
    is_cross_encoder = hasattr(model, 'cross_attention')

    if is_cross_encoder:
        print(f"Standard validation completed. Val Loss: {primary_metrics['loss']:.4f}, "
              f"Val Accuracy: {primary_metrics['accuracy']:.4f}")
    else:
        if hasattr(model, 'query_type') and model.query_type == 'text':
            display_name = "Text-Query Sim"
            similarity_key = 'text_query_similarity'
        else:
            display_name = "EEG-Query Sim"
            similarity_key = 'eeg_query_similarity'

        print(f"Standard validation completed. Val Loss: {primary_metrics['loss']:.4f}, "
              f"{display_name}: {primary_metrics[similarity_key]:.4f}")

    # NEW: Dataset-specific validation (every epoch, but only at primary masking level)
    dataset_sources_info = detect_dataset_sources(val_dataloader)
    if dataset_sources_info['has_multiple']:
        print(f"\nDetected multiple datasets - running dataset-specific validation...")

        # Split validation data by dataset source
        dataset_batches = split_validation_data_by_dataset(val_dataloader)

        # Run dataset-specific validation at primary masking level only
        dataset_results = validate_dataset_specific(model, dataset_batches, device, primary_masking_level)

        # Log dataset-specific results to wandb under "dataset_perf" section
        dataset_log_dict = {}
        for dataset_name, result_data in dataset_results.items():
            metrics = result_data['metrics']
            for metric_name, value in metrics.items():
                dataset_log_dict[f'dataset_perf/{dataset_name}_{metric_name}'] = value

        dataset_log_dict['dataset_perf/epoch_num'] = epoch_num
        dataset_log_dict['dataset_perf/masking_level'] = primary_masking_level

        wandb.log(dataset_log_dict)

        print(f"Dataset-specific validation logged to wandb under 'dataset_perf' section")
    else:
        print(f"Single dataset detected - skipping dataset-specific validation")

    # DYNAMIC Multi-masking validation (if enabled and appropriate epoch)
    if (enable_multi_masking_validation and epoch_num % multi_masking_frequency == 0):

        print(f"\n=== DYNAMIC MULTI-MASKING VALIDATION (Epoch {epoch_num}) ===")
        print(f"Testing masking levels: {validation_masking_levels}%")
        print(f"Using SINGLE dataloader with dynamic masking probability")

        all_masking_metrics = {}

        for masking_level in validation_masking_levels:
            masking_probability = masking_level / 100.0
            masking_level_name = f"{masking_level}%"

            print(f"  Validating on {masking_level_name} masking...")

            masking_metrics, masking_main_metric = validate_single_masking_level(
                model, val_dataloader, device, masking_probability, masking_level_name
            )

            # Store metrics for this masking level
            all_masking_metrics[masking_level_name] = {
                'metrics': masking_metrics,
                'main_metric': masking_main_metric
            }

            # Log to wandb with masking level prefix
            masking_log_dict = {}
            for metric_name, value in masking_metrics.items():
                masking_log_dict[f'val/masking_{masking_level}_{metric_name}'] = value
            masking_log_dict[f'val/masking_{masking_level}_epoch_num'] = epoch_num

            wandb.log(masking_log_dict)

            # Print summary for this masking level
            if is_cross_encoder:
                print(f"    {masking_level_name}: Loss {masking_metrics['loss']:.4f}, "
                      f"Accuracy {masking_metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_key = 'text_query_similarity'
                else:
                    sim_key = 'eeg_query_similarity'

                print(f"    {masking_level_name}: Loss {masking_metrics['loss']:.4f}, "
                      f"Similarity {masking_metrics[sim_key]:.4f}")

        # Print multi-masking summary
        print(f"\n  DYNAMIC multi-masking validation summary:")
        for masking_level, data in all_masking_metrics.items():
            metrics = data['metrics']
            main_metric = data['main_metric']

            if is_cross_encoder:
                print(f"    {masking_level}: Accuracy {metrics['accuracy']:.4f}")
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    sim_key = 'text_query_similarity'
                else:
                    sim_key = 'eeg_query_similarity'
                print(f"    {masking_level}: Similarity {metrics[sim_key]:.4f}")

        print("=" * 60)

    # Standard ranking validation every 3rd epoch
    if epoch_num % 3 == 0:
        if is_cross_encoder:
            print(f"Running cross-encoder ranking validation...")
            ranking_metrics = perform_cross_encoder_ranking_validation(
                model, val_dataloader, device, epoch_num, subset_size=100
            )
        else:
            print(f"Running dual-encoder ranking validation...")
            ranking_metrics = perform_ranking_validation(
                model, val_dataloader, device, epoch_num, subset_size=None
            )

    # Return primary metrics for early stopping
    return primary_metrics['loss'], primary_main_metric


def initialize_wandb(config):
    """Initialize wandb logging"""

    # Create descriptive run name
    encoder_type = config.get('encoder_type', 'dual')
    query_type = config.get('query_type', 'eeg')
    holdout_subjects = config.get('holdout_subjects', False)
    dataset_name = config.get('dataset_name', 'unknown')
    fold = config.get('fold', None)

    if encoder_type == 'cross':
        pooling_strategy = 'cross'
        eeg_arch = config.get('eeg_arch', 'simple')
        use_lora = config.get('use_lora', True)
        model_suffix = f"_lora{config.get('lora_r', 16)}" if use_lora else "_frozen"

        if holdout_subjects and fold is not None:
            split_suffix = f"_holdout_fold{fold}"
        elif holdout_subjects:
            split_suffix = "_holdout"
        else:
            split_suffix = "_random"

        run_name = f"cross_{query_type}_{eeg_arch}{model_suffix}{split_suffix}_{dataset_name}"
    else:
        pooling_strategy = config.get('pooling_strategy', 'multi')
        eeg_arch = config.get('eeg_arch', 'simple')
        use_lora = config.get('use_lora', True)
        model_suffix = f"_lora{config.get('lora_r', 16)}" if use_lora else "_frozen"

        if holdout_subjects and fold is not None:
            split_suffix = f"_holdout_fold{fold}"
        elif holdout_subjects:
            split_suffix = "_holdout"
        else:
            split_suffix = "_random"

        run_name = f"dual_{query_type}_{pooling_strategy}_{eeg_arch}{model_suffix}{split_suffix}_{dataset_name}"

    # Add multi-masking info to run name if enabled
    if config.get('enable_multi_masking_validation', False):
        run_name += "_dynamic_mask"

    # NEW: Add baseline indicator for text queries
    if query_type == 'text':
        run_name += "_TEXT_BASELINE"

    # NEW: Updated tags to include text baseline
    tags = ['brain-retrieval', 'colbert', 'lora' if use_lora else 'frozen',
            f'arch-{eeg_arch}', f'encoder-{encoder_type}',
            f'pooling-{pooling_strategy}' if encoder_type == 'dual' else 'cross-encoder',
            'holdout-subjects' if holdout_subjects else 'random-split',
            f'dataset-{dataset_name}',
            'dynamic-multi-masking-validation']

    # NEW: Add query type specific tags
    if query_type == 'text':
        tags.extend(['text-baseline', 'text-query'])
    else:
        tags.extend(['eeg', 'eeg-query'])

    wandb.init(
        project="ECIR2026",
        name=run_name,
        config={
            'encoder_type': encoder_type,
            'query_type': query_type,
            'pooling_strategy': pooling_strategy,
            'holdout_subjects': holdout_subjects,
            'fold': fold,
            'split_method': 'holdout_subjects' if holdout_subjects else 'random_samples',
            'dataset_name': dataset_name,

            # Model config
            'colbert_model_name': config.get('colbert_model_name', 'colbert-ir/colbertv2.0'),
            'eeg_arch': eeg_arch,
            'hidden_dim': config.get('hidden_dim', 768),
            'num_vectors': config.get('num_vectors', 32),

            # LoRA config
            'use_lora': use_lora,
            'lora_r': config.get('lora_r', 16),
            'lora_alpha': config.get('lora_alpha', 32),
            'use_pretrained_text': config.get('use_pretrained_text', True),  # NEW
            'trainable_params': config.get('trainable_params', 0),
            'total_params': config.get('total_params', 0),

            # Training config
            'batch_size': config.get('batch_size', 8),
            'learning_rate': config.get('learning_rate', 1e-4),
            'epochs': config.get('epochs', 50),
            'patience': config.get('patience', 10),

            # Data config
            'max_text_len': config.get('max_text_len', 256),
            'max_eeg_len': config.get('max_eeg_len', 50),
            'train_samples': config.get('train_samples', 0),
            'val_samples': config.get('val_samples', 0),
            'train_subjects': config.get('train_subjects', 0),
            'val_subjects': config.get('val_subjects', 0),

            # Dynamic Multi-masking validation config
            'enable_multi_masking_validation': config.get('enable_multi_masking_validation', False),
            'validation_masking_levels': config.get('validation_masking_levels', []),
            'multi_masking_frequency': config.get('multi_masking_frequency', 3),
            'primary_masking_level': config.get('primary_masking_level', 90),
            'training_masking_level': config.get('training_masking_level', 90),
            'dynamic_masking': True,

            # Experiment config
            'seed': config.get('seed', 42),
            'loss_type': 'bce' if encoder_type == 'cross' else 'contrastive',
            'similarity_function': f'{pooling_strategy}_similarity',
            'is_text_baseline': query_type == 'text'  # NEW
        },
        tags=tags
    )


def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs,
                patience=10, device='cuda', debug=False, config=None,
                enable_multi_masking_validation=False, multi_masking_frequency=3,
                validation_masking_levels=[0, 25, 50, 75, 90, 100],
                primary_masking_level=90):
    """
    Complete training loop with validation and early stopping + DYNAMIC MULTI-MASKING VALIDATION

    Args:
        model: Model to train
        train_dataloader: Training dataloader (with dynamic masking)
        val_dataloader: Single validation dataloader (with dynamic masking capability)
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on
        debug: Enable debug prints
        config: Configuration dict for wandb
        enable_multi_masking_validation: Enable dynamic multi-masking validation
        multi_masking_frequency: Run multi-masking validation every N epochs
        validation_masking_levels: List of masking percentages to test
        primary_masking_level: Masking level to use for early stopping decisions
    """

    # Initialize wandb
    if config:
        initialize_wandb(config)

    is_cross_encoder = hasattr(model, 'cross_attention')
    encoder_type = 'Cross-Encoder' if is_cross_encoder else 'Dual-Encoder'
    pooling_info = model.pooling_strategy if hasattr(model, 'pooling_strategy') else 'unknown'

    print(f"Starting {encoder_type} training with early stopping (patience={patience})...")
    print(f"Pooling strategy: {pooling_info}")
    print(f"Training masking level: {config.get('training_masking_level', 90)}%")

    # DYNAMIC Multi-masking validation info
    if enable_multi_masking_validation:
        print(f"DYNAMIC multi-masking validation: ENABLED")
        print(f"  Validation masking levels: {validation_masking_levels}%")
        print(f"  Multi-masking frequency: every {multi_masking_frequency} epochs")
        print(f"  Primary masking level: {primary_masking_level}% (for early stopping)")
        print(f"  Memory efficient: Uses SINGLE dataloader with dynamic masking")
    else:
        print(f"Multi-masking validation: DISABLED")

    # Early stopping variables
    best_val_loss = float('inf')
    best_val_metric = -1.0 if not is_cross_encoder else 0.0  # similarity vs accuracy
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0
    early_stopped = False

    for epoch in range(num_epochs):
        epoch_num = epoch + 1

        # Training epoch
        train_loss, train_metric, train_grad_norm = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch_num,
            total_epochs=num_epochs,
            debug=debug
        )

        # Enhanced validation epoch with DYNAMIC multi-masking support
        val_loss, val_metric = validate_epoch(
            model=model,
            val_dataloader=val_dataloader,
            device=device,
            epoch_num=epoch_num,
            enable_multi_masking_validation=enable_multi_masking_validation,
            validation_masking_levels=validation_masking_levels,
            multi_masking_frequency=multi_masking_frequency,
            primary_masking_level=primary_masking_level
        )

        # Early stopping logic - use loss for primary criterion
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metric = val_metric
            best_epoch = epoch_num
            epochs_without_improvement = 0

            # Save best model state
            best_model_state = model.state_dict().copy()

            print(f"New best validation loss: {best_val_loss:.4f} (epoch {epoch_num})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch_num} epochs!")
                print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                early_stopped = True

        # Log epoch-level metrics
        metric_name = 'accuracy' if is_cross_encoder else 'similarity'
        wandb.log({
            'epoch/train_loss': train_loss,
            'epoch/val_loss': val_loss,
            f'epoch/train_{metric_name}': train_metric,
            f'epoch/val_{metric_name}': val_metric,
            'epoch/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch/epoch_num': epoch_num,
            'epoch/epochs_without_improvement': epochs_without_improvement,
            'epoch/best_val_loss': best_val_loss
        })

        # Print epoch summary
        metric_display = 'Accuracy' if is_cross_encoder else 'EEG-Query Sim'
        print(f"\nEpoch {epoch_num}/{num_epochs} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, {metric_display}: {train_metric:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, {metric_display}: {val_metric:.4f}")
        print(f"  Best  - Val Loss: {best_val_loss:.4f} (epoch {best_epoch})")
        print(f"  Early Stopping: {epochs_without_improvement}/{patience} epochs without improvement")

        # Enhanced logging for multi-masking epochs
        if (enable_multi_masking_validation and epoch_num % multi_masking_frequency == 0):
            print(f"  Note: DYNAMIC multi-masking validation performed this epoch ({validation_masking_levels}%)")
        elif epoch_num % 1 == 0:
            print(f"  Note: Standard ranking validation performed this epoch")

        print("-" * 70)

        if early_stopped:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch}")

    # Log training summary
    wandb.log({
        'training/completed_epochs': epoch_num,
        'training/early_stopped': early_stopped,
        'training/best_epoch': best_epoch,
        'training/best_val_loss': best_val_loss,
        'training/final_val_loss': val_loss
    })

    print(f"\n{encoder_type} training completed!")
    if early_stopped:
        print(f"Stopped early after {epoch_num} epochs")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    if enable_multi_masking_validation:
        print(f"DYNAMIC multi-masking validation was performed every {multi_masking_frequency} epochs")
        print(f"Memory efficient: Used single dataloader with dynamic masking")
        print(f"Check wandb logs for performance trends across masking levels: {validation_masking_levels}%")

    return model


def test_model(model, test_dataloader, device, debug=False,
               test_masking_levels=[0, 25, 50, 75, 90, 100],
               primary_masking_level=90, wandb_namespace=None):
    """
    Comprehensive test evaluation with MULTI-MASKING support

    Args:
        model: Trained model to test
        test_dataloader: Test dataloader with dynamic masking capability
        device: Device to run testing on
        debug: Enable debug prints
        test_masking_levels: List of masking levels to test (0-100)
        primary_masking_level: Primary masking level for main results
    """

    print(f"=== COMPREHENSIVE MULTI-MASKING TEST SET EVALUATION ===")
    print(f"Test set size: {len(test_dataloader.dataset)} samples")
    print(f"Test subjects: {len(test_dataloader.dataset.unique_subjects)} unique subjects")
    print(f"Test masking levels: {test_masking_levels}%")
    print(f"Primary masking level: {primary_masking_level}%")

    model.eval()
    is_cross_encoder = hasattr(model, 'cross_attention')

    # Store original masking probability
    original_probability = test_dataloader.dataset.get_current_masking_probability()
    all_test_metrics = {}

    try:
        # Test at each masking level
        for masking_level in test_masking_levels:
            masking_probability = masking_level / 100.0
            is_primary = (masking_level == primary_masking_level)

            print(f"\n=== TESTING AT {masking_level}% MASKING {'(PRIMARY)' if is_primary else ''} ===")

            # Set masking probability for this level
            test_dataloader.dataset.set_masking_probability(masking_probability)

            # 1. Main ranking evaluation at this masking level
            if is_cross_encoder:
                main_ranking_results = perform_cross_encoder_test_ranking_at_masking_level(
                    model, test_dataloader, device, masking_level
                )
            else:
                main_ranking_results = perform_dual_encoder_test_ranking_at_masking_level(
                    model, test_dataloader, device, masking_level
                )

            # 2. Dataset-specific ranking evaluation at this masking level
            dataset_sources_info = detect_dataset_sources(test_dataloader)
            dataset_ranking_results = {}

            if dataset_sources_info['has_multiple']:
                print(f"  Running dataset-specific evaluation at {masking_level}% masking...")
                dataset_ranking_results = perform_dataset_specific_ranking(
                    model, test_dataloader, device, epoch_num=0, subset_size=100,
                    prefix=f'test_ranking/masking_{masking_level}/dataset'
                )

            # 3. Add masking level to all metric names
            masking_prefix = f'test_ranking/masking_{masking_level}'

            # Rename main results
            main_masking_results = {}
            for key, value in main_ranking_results.items():
                new_key = key.replace('test_ranking/main', masking_prefix)
                main_masking_results[new_key] = value

            # Store results for this masking level
            level_results = {**main_masking_results, **dataset_ranking_results}
            all_test_metrics.update(level_results)

            # Print summary for this masking level
            if main_ranking_results:
                main_mrr = main_ranking_results.get('test_ranking/main/rr', 0)
                main_hit1 = main_ranking_results.get('test_ranking/main/hit_at_1', 0)
                main_hit10 = main_ranking_results.get('test_ranking/main/hit_at_10', 0)
                print(f"  {masking_level}% Masking: MRR {main_mrr:.4f}, Hit@1 {main_hit1:.4f}, Hit@10 {main_hit10:.4f}")

        # 4. Noise baseline evaluation (at primary masking level)
        print(f"\n=== NOISE BASELINE AT {primary_masking_level}% MASKING ===")
        test_dataloader.dataset.set_masking_probability(primary_masking_level / 100.0)
        noise_ranking_results = perform_noise_ranking_evaluation(
            model, test_dataloader, device, primary_masking_level
        )
        all_test_metrics.update(noise_ranking_results)

        # 5. Calculate cross-masking performance comparisons
        masking_comparison_metrics = calculate_masking_performance_trends(
            all_test_metrics, test_masking_levels
        )
        all_test_metrics.update(masking_comparison_metrics)

        # 6. Add comprehensive summary statistics
        summary_metrics = {
            'test_ranking/summary/test_masking_levels': test_masking_levels,
            'test_ranking/summary/primary_masking_level': primary_masking_level,
            'test_ranking/summary/num_test_samples': len(test_dataloader.dataset),
            'test_ranking/summary/num_test_subjects': len(test_dataloader.dataset.unique_subjects),
            'test_ranking/summary/has_multiple_datasets': dataset_sources_info['has_multiple'],
            'test_ranking/summary/encoder_type': 'cross' if is_cross_encoder else 'dual'
        }

        if hasattr(model, 'query_type'):
            summary_metrics['test_ranking/summary/query_type'] = model.query_type

        all_test_metrics.update(summary_metrics)

        # Log all metrics to wandb
        if wandb_namespace:
            wandb.log({f'{wandb_namespace}/{key}': value for key, value in all_test_metrics.items()})
        else:
            wandb.log(all_test_metrics)

        # Print comprehensive multi-masking summary
        print_multi_masking_test_summary(all_test_metrics, test_masking_levels, primary_masking_level)

        return all_test_metrics

    finally:
        # Always restore original masking probability
        test_dataloader.dataset.set_masking_probability(original_probability)


def perform_dual_encoder_test_ranking_at_masking_level(model, test_dataloader, device, masking_level):
    """Perform ranking evaluation at specific masking level for dual encoder"""

    doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        return {}

    subset_size = len(doc_list)
    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(test_dataloader, device)

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_dual_encoder_subset(
            model, eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

    if not all_metrics:
        return {}

    # Aggregate metrics
    test_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        test_metrics[f'test_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            test_metrics[f'test_ranking/main/{metric_name}_std'] = np.std(values)

    test_metrics.update({
        'test_ranking/main/num_unique_documents': len(doc_list),
        'test_ranking/main/num_queries_evaluated': len(all_metrics),
        'test_ranking/main/subset_size': subset_size,
        'test_ranking/main/masking_level': masking_level,
        'test_ranking/main/query_type': model.query_type
    })

    return test_metrics


def perform_cross_encoder_test_ranking_at_masking_level(model, test_dataloader, device, masking_level):
    """Perform ranking evaluation at specific masking level for cross encoder"""

    doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        return {}

    subset_size = len(doc_list)
    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(test_dataloader, device)

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_cross_encoder_subset(
            model, eeg_query, text_query, doc_list, doc_subset_indices, device
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

    if not all_metrics:
        return {}

    # Aggregate metrics
    test_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        test_metrics[f'test_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            test_metrics[f'test_ranking/main/{metric_name}_std'] = np.std(values)

    test_metrics.update({
        'test_ranking/main/num_unique_documents': len(doc_list),
        'test_ranking/main/num_queries_evaluated': len(all_metrics),
        'test_ranking/main/subset_size': subset_size,
        'test_ranking/main/masking_level': masking_level,
        'test_ranking/main/encoder_type': 'cross'
    })

    return test_metrics


def calculate_masking_performance_trends(all_metrics, test_masking_levels):
    """Calculate performance trends across masking levels"""

    trend_metrics = {}

    # Extract MRR and Hit@1 across all masking levels
    mrr_values = []
    hit1_values = []
    hit10_values = []

    for masking_level in test_masking_levels:
        mrr_key = f'test_ranking/masking_{masking_level}/rr'
        hit1_key = f'test_ranking/masking_{masking_level}/hit_at_1'
        hit10_key = f'test_ranking/masking_{masking_level}/hit_at_10'

        if mrr_key in all_metrics:
            mrr_values.append(all_metrics[mrr_key])
        if hit1_key in all_metrics:
            hit1_values.append(all_metrics[hit1_key])
        if hit10_key in all_metrics:
            hit10_values.append(all_metrics[hit10_key])

    if mrr_values:
        trend_metrics.update({
            'test_ranking/trends/mrr_mean': np.mean(mrr_values),
            'test_ranking/trends/mrr_std': np.std(mrr_values),
            'test_ranking/trends/mrr_min': np.min(mrr_values),
            'test_ranking/trends/mrr_max': np.max(mrr_values),
            'test_ranking/trends/mrr_range': np.max(mrr_values) - np.min(mrr_values)
        })

    if hit1_values:
        trend_metrics.update({
            'test_ranking/trends/hit1_mean': np.mean(hit1_values),
            'test_ranking/trends/hit1_std': np.std(hit1_values),
            'test_ranking/trends/hit1_min': np.min(hit1_values),
            'test_ranking/trends/hit1_max': np.max(hit1_values),
            'test_ranking/trends/hit1_range': np.max(hit1_values) - np.min(hit1_values)
        })

    if hit10_values:
        trend_metrics.update({
            'test_ranking/trends/hit10_mean': np.mean(hit10_values),
            'test_ranking/trends/hit10_std': np.std(hit10_values),
            'test_ranking/trends/hit10_min': np.min(hit10_values),
            'test_ranking/trends/hit10_max': np.max(hit10_values),
            'test_ranking/trends/hit10_range': np.max(hit10_values) - np.min(hit10_values)
        })

    return trend_metrics


def print_multi_masking_test_summary(all_metrics, test_masking_levels, primary_masking_level):
    """Print comprehensive multi-masking test summary"""

    print(f"\n=== MULTI-MASKING TEST EVALUATION SUMMARY ===")
    print(f"Test masking levels: {test_masking_levels}%")
    print(f"Primary masking level: {primary_masking_level}%")

    print(f"\nPerformance across masking levels:")
    print(f"{'Masking':>8} {'MRR':>8} {'Hit@1':>8} {'Hit@5':>8} {'Hit@10':>8}")
    print(f"{'-' * 45}")

    for masking_level in test_masking_levels:
        mrr = all_metrics.get(f'test_ranking/masking_{masking_level}/rr', 0)
        hit1 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_1', 0)
        hit5 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_5', 0)
        hit10 = all_metrics.get(f'test_ranking/masking_{masking_level}/hit_at_10', 0)

        primary_marker = " *" if masking_level == primary_masking_level else ""
        print(f"{masking_level:>6}%{primary_marker:>2} {mrr:>8.4f} {hit1:>8.4f} {hit5:>8.4f} {hit10:>8.4f}")

    # Noise baseline comparison
    noise_mrr = all_metrics.get('test_ranking/noise/rr', 0)
    noise_hit1 = all_metrics.get('test_ranking/noise/hit_at_1', 0)

    if noise_mrr > 0:
        primary_mrr = all_metrics.get(f'test_ranking/masking_{primary_masking_level}/rr', 0)
        primary_hit1 = all_metrics.get(f'test_ranking/masking_{primary_masking_level}/hit_at_1', 0)

        print(f"\nNoise Baseline: MRR {noise_mrr:.4f}, Hit@1 {noise_hit1:.4f}")
        print(f"Primary vs Noise: MRR +{primary_mrr - noise_mrr:.4f}, Hit@1 +{primary_hit1 - noise_hit1:.4f}")

    # Performance trends
    if 'test_ranking/trends/mrr_range' in all_metrics:
        mrr_range = all_metrics['test_ranking/trends/mrr_range']
        hit1_range = all_metrics['test_ranking/trends/hit1_range']
        print(f"\nRobustness Analysis:")
        print(f"  MRR range across masking levels: {mrr_range:.4f}")
        print(f"  Hit@1 range across masking levels: {hit1_range:.4f}")

    print(f"\nAll multi-masking test results logged to wandb under:")
    for masking_level in test_masking_levels:
        print(f"  - test_ranking/masking_{masking_level}/ (performance at {masking_level}%)")
    print(f"  - test_ranking/trends/ (cross-masking analysis)")
    print(f"  - test_ranking/noise/ (noise baseline)")
    print(f"  - test_ranking/summary/ (summary statistics)")
    print("=" * 60)


def perform_dataset_specific_ranking(model, dataloader, device, epoch_num, subset_size, prefix='ranking/dataset'):
    """
    Perform dataset-specific ranking evaluation

    Args:
        model: Model to evaluate
        dataloader: Dataloader with multiple datasets
        device: Device to run on
        epoch_num: Current epoch (for logging)
        subset_size: Size of document subset for ranking
        prefix: wandb logging prefix (e.g., 'val_ranking/dataset' or 'test_ranking/dataset')
    """

    print(f"Running dataset-specific ranking evaluation...")

    # Split data by dataset
    dataset_batches = split_validation_data_by_dataset(dataloader)

    all_dataset_metrics = {}
    is_cross_encoder = hasattr(model, 'cross_attention')

    for dataset_name, samples in dataset_batches.items():
        if len(samples) == 0:
            continue

        print(f"  Ranking evaluation on {dataset_name} ({len(samples)} samples)...")

        # Build document database for this dataset only
        dataset_doc_list = []
        dataset_query_to_doc_mapping = {}

        for query_idx, sample in enumerate(samples):
            doc_text = sample['metadata'][0]['document_text'].strip()

            if doc_text:
                # For dataset-specific evaluation, each query has its own document
                doc_info = {
                    'idx': query_idx,
                    'text': doc_text,
                    'input_ids': sample['doc']['input_ids'].squeeze(0),
                    'attention_mask': sample['doc']['attention_mask'].squeeze(0)
                }
                dataset_doc_list.append(doc_info)
                dataset_query_to_doc_mapping[query_idx] = query_idx

        if len(dataset_doc_list) == 0:
            continue

        # Generate subsets for this dataset
        actual_subset_size = min(subset_size, len(dataset_doc_list))
        dataset_query_subsets = generate_consistent_subsets(
            dataset_doc_list, dataset_query_to_doc_mapping, actual_subset_size
        )

        # Collect queries for this dataset
        dataset_queries = []
        for sample in samples:
            eeg_query = sample['eeg_query'].to(device)
            eeg_mv_mask = sample['eeg_mv_mask'].to(device)
            text_query = {
                'input_ids': sample['text_query']['input_ids'].to(device),
                'attention_mask': sample['text_query']['attention_mask'].to(device)
            }
            dataset_queries.append((eeg_query, eeg_mv_mask, text_query))

        # Perform ranking for this dataset
        dataset_all_metrics = []

        for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(dataset_queries):
            if query_idx not in dataset_query_to_doc_mapping:
                continue

            correct_doc_idx = dataset_query_to_doc_mapping[query_idx]
            doc_subset_indices = dataset_query_subsets[query_idx]

            if is_cross_encoder:
                ranked_indices, scores = rank_cross_encoder_subset(
                    model, eeg_query, text_query, dataset_doc_list, doc_subset_indices, device
                )
            else:
                ranked_indices, scores = rank_dual_encoder_subset(
                    model, eeg_query, eeg_mv_mask, dataset_doc_list, doc_subset_indices, device, text_query=text_query
                )

            query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
            dataset_all_metrics.append(query_metrics)

        if len(dataset_all_metrics) == 0:
            continue

        # Aggregate metrics for this dataset
        dataset_metrics = {}
        metric_names = dataset_all_metrics[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in dataset_all_metrics]
            dataset_metrics[f'{prefix}_{dataset_name}_{metric_name}'] = np.mean(values)
            if metric_name != 'rank_of_correct':
                dataset_metrics[f'{prefix}_{dataset_name}_{metric_name}_std'] = np.std(values)

        # Add metadata
        dataset_metrics.update({
            f'{prefix}_{dataset_name}_num_queries': len(dataset_all_metrics),
            f'{prefix}_{dataset_name}_subset_size': actual_subset_size,
            f'{prefix}_{dataset_name}_epoch_num': epoch_num
        })

        # Print results for this dataset
        print(f"    {dataset_name}: MRR {dataset_metrics[f'{prefix}_{dataset_name}_rr']:.4f}, "
              f"Hit@1 {dataset_metrics[f'{prefix}_{dataset_name}_hit_at_1']:.4f}, "
              f"Hit@10 {dataset_metrics[f'{prefix}_{dataset_name}_hit_at_10']:.4f}")

        all_dataset_metrics.update(dataset_metrics)

    return all_dataset_metrics


def perform_noise_ranking_evaluation(model, test_dataloader, device, primary_masking_level=90):
    """
    Perform ranking evaluation with noise queries as baseline

    Args:
        model: Model to evaluate
        test_dataloader: Test dataloader
        device: Device to run on
        primary_masking_level: Masking level to use
    """

    print(f"\n=== NOISE BASELINE RANKING EVALUATION ===")
    print(f"Testing ranking performance with Gaussian noise instead of EEG queries...")

    # Set masking probability
    original_probability = test_dataloader.dataset.get_current_masking_probability()
    test_dataloader.dataset.set_masking_probability(primary_masking_level / 100.0)

    try:
        # Build document database
        doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

        if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
            print("Warning: No valid query-document pairs found for noise ranking")
            return {}

        # Use all documents
        subset_size = len(doc_list)
        query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
        queries = collect_eeg_queries(test_dataloader, device)

        print(f"Noise ranking: {len(queries)} noise queries against {subset_size} documents each")

        all_metrics = []
        is_cross_encoder = hasattr(model, 'cross_attention')

        for query_idx, (real_eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
            if query_idx not in query_to_doc_mapping:
                continue

            # Replace EEG with noise
            noise_eeg_query = generate_eeg_noise_baseline(real_eeg_query).to(device)

            correct_doc_idx = query_to_doc_mapping[query_idx]
            doc_subset_indices = query_subsets[query_idx]

            # Perform ranking with noise query
            if is_cross_encoder:
                ranked_indices, scores = rank_cross_encoder_subset(
                    model, noise_eeg_query, text_query, doc_list, doc_subset_indices, device
                )
            else:
                ranked_indices, scores = rank_dual_encoder_subset(
                    model, noise_eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
                )

            query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
            all_metrics.append(query_metrics)

            if (query_idx + 1) % 50 == 0:
                print(f"  Processed {query_idx + 1}/{len(queries)} noise queries...")

        if not all_metrics:
            print("Warning: No noise ranking metrics computed")
            return {}

        # Aggregate noise ranking metrics
        noise_metrics = {}
        metric_names = all_metrics[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics]
            noise_metrics[f'test_ranking/noise/{metric_name}'] = np.mean(values)
            if metric_name != 'rank_of_correct':
                noise_metrics[f'test_ranking/noise/{metric_name}_std'] = np.std(values)

        noise_metrics.update({
            'test_ranking/noise/num_unique_documents': len(doc_list),
            'test_ranking/noise/num_queries_evaluated': len(all_metrics),
            'test_ranking/noise/subset_size': subset_size,
            'test_ranking/noise/masking_level': primary_masking_level
        })

        # Print noise baseline results
        print(f"Noise Baseline Ranking Results:")
        print(f"  Queries Evaluated: {len(all_metrics)}")
        print(f"  MRR: {noise_metrics['test_ranking/noise/rr']:.4f}")
        print(f"  Hit@1: {noise_metrics['test_ranking/noise/hit_at_1']:.4f}")
        print(f"  Hit@5: {noise_metrics['test_ranking/noise/hit_at_5']:.4f}")
        print(f"  Hit@10: {noise_metrics['test_ranking/noise/hit_at_10']:.4f}")
        print(f"  Hit@20: {noise_metrics['test_ranking/noise/hit_at_20']:.4f}")

        return noise_metrics

    finally:
        # Restore original masking probability
        test_dataloader.dataset.set_masking_probability(original_probability)


def perform_dual_encoder_test_ranking(model, test_dataloader, device):
    """Perform comprehensive ranking evaluation on test set for dual encoder"""

    print(f"\n=== TEST RANKING EVALUATION ===")

    doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for test ranking")
        return {}

    # Use all unique documents for test evaluation
    subset_size = len(doc_list)
    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(test_dataloader, device)

    print(f"Test ranking: {len(queries)} queries against {subset_size} documents each")
    print(f"Query type: {model.query_type}")

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_dual_encoder_subset(
            model, eeg_query, eeg_mv_mask, doc_list, doc_subset_indices, device, text_query=text_query
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} test queries...")

    if not all_metrics:
        print("Warning: No test ranking metrics computed")
        return {}

    # Aggregate test ranking metrics
    test_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        test_metrics[f'test_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            test_metrics[f'test_ranking/main/{metric_name}_std'] = np.std(values)

    test_metrics.update({
        'test_ranking/main/num_unique_documents': len(doc_list),
        'test_ranking/main/num_queries_evaluated': len(all_metrics),
        'test_ranking/main/subset_size': subset_size,
        'test_ranking/main/query_type': model.query_type
    })

    # Print results
    print(f"Test Ranking Results:")
    print(f"  Query Type: {model.query_type}")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {test_metrics['test_ranking/main/rr']:.4f}")
    print(f"  Hit@1: {test_metrics['test_ranking/main/hit_at_1']:.4f}")
    print(f"  Hit@5: {test_metrics['test_ranking/main/hit_at_5']:.4f}")
    print(f"  Hit@10: {test_metrics['test_ranking/main/hit_at_10']:.4f}")
    print(f"  Hit@20: {test_metrics['test_ranking/main/hit_at_20']:.4f}")
    print(f"  Precision@1: {test_metrics['test_ranking/main/precision_at_1']:.4f}")
    print(f"  Precision@10: {test_metrics['test_ranking/main/precision_at_10']:.4f}")

    return test_metrics



def perform_cross_encoder_test_ranking(model, test_dataloader, device):
    """Perform comprehensive ranking evaluation on test set for cross encoder"""

    print(f"\n=== CROSS-ENCODER TEST RANKING EVALUATION ===")

    doc_list, query_to_doc_mapping = build_document_database(test_dataloader)

    if len(doc_list) == 0 or len(query_to_doc_mapping) == 0:
        print("Warning: No valid query-document pairs found for test ranking")
        return {}

    # Use all unique documents for test evaluation
    subset_size = len(doc_list)
    query_subsets = generate_consistent_subsets(doc_list, query_to_doc_mapping, subset_size)
    queries = collect_eeg_queries(test_dataloader, device)

    print(f"Test ranking: {len(queries)} queries against {subset_size} documents each")

    all_metrics = []

    for query_idx, (eeg_query, eeg_mv_mask, text_query) in enumerate(queries):
        if query_idx not in query_to_doc_mapping:
            continue

        correct_doc_idx = query_to_doc_mapping[query_idx]
        doc_subset_indices = query_subsets[query_idx]

        ranked_indices, scores = rank_cross_encoder_subset(
            model, eeg_query, text_query, doc_list, doc_subset_indices, device
        )

        query_metrics = compute_ranking_metrics(ranked_indices, correct_doc_idx)
        all_metrics.append(query_metrics)

        if (query_idx + 1) % 50 == 0:
            print(f"  Processed {query_idx + 1}/{len(queries)} test queries...")

    if not all_metrics:
        print("Warning: No test ranking metrics computed")
        return {}

    # Aggregate test ranking metrics
    test_metrics = {}
    metric_names = all_metrics[0].keys()

    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        test_metrics[f'test_ranking/main/{metric_name}'] = np.mean(values)
        if metric_name != 'rank_of_correct':
            test_metrics[f'test_ranking/main/{metric_name}_std'] = np.std(values)

    test_metrics.update({
        'test_ranking/main/num_unique_documents': len(doc_list),
        'test_ranking/main/num_queries_evaluated': len(all_metrics),
        'test_ranking/main/subset_size': subset_size,
        'test_ranking/main/encoder_type': 'cross'
    })

    # Print results
    print(f"Cross-Encoder Test Ranking Results:")
    print(f"  Queries Evaluated: {len(all_metrics)}")
    print(f"  MRR: {test_metrics['test_ranking/main/rr']:.4f}")
    print(f"  Hit@1: {test_metrics['test_ranking/main/hit_at_1']:.4f}")
    print(f"  Hit@5: {test_metrics['test_ranking/main/hit_at_5']:.4f}")
    print(f"  Hit@10: {test_metrics['test_ranking/main/hit_at_10']:.4f}")
    print(f"  Hit@20: {test_metrics['test_ranking/main/hit_at_20']:.4f}")
    print(f"  Precision@1: {test_metrics['test_ranking/main/precision_at_1']:.4f}")
    print(f"  Precision@10: {test_metrics['test_ranking/main/precision_at_10']:.4f}")

    return test_metrics


def generate_eeg_noise_baseline(eeg_queries, noise_type='gaussian'):
    """Generate random noise with same shape as EEG queries for baseline comparison"""
    if noise_type == 'gaussian':
        # Generate Gaussian noise with same shape as real EEG
        noise = torch.randn_like(eeg_queries)
        # Optional: match the scale of real EEG data
        eeg_std = eeg_queries.std()
        eeg_mean = eeg_queries.mean()
        noise = noise * eeg_std + eeg_mean
    return noise


def test_noise_baseline(model, test_dataloader, device, primary_masking_level=90):
    """Test model performance with random noise instead of EEG queries"""

    print(f"\n=== NOISE BASELINE EVALUATION ===")
    print(f"Testing with Gaussian noise instead of EEG queries...")
    print(f"This verifies the model learns meaningful EEG semantics")

    model.eval()
    is_cross_encoder = hasattr(model, 'cross_attention')

    # Only test at primary masking level for noise baseline
    masking_probability = primary_masking_level / 100.0

    # Set masking probability
    original_probability = test_dataloader.dataset.get_current_masking_probability()
    test_dataloader.dataset.set_masking_probability(masking_probability)

    try:
        total_loss = 0
        num_batches = 0

        if is_cross_encoder:
            accuracies = []
            neg_accuracies = []
        else:
            similarities = []
            query_doc_sims = []
            eeg_doc_sims = []

        for batch in test_dataloader:
            # Replace EEG queries with Gaussian noise
            real_eeg_queries = batch['eeg_queries'].to(device)
            noise_eeg_queries = generate_eeg_noise_baseline(real_eeg_queries).to(device)

            # Replace the EEG queries in the batch with noise
            noise_batch = {
                'eeg_queries': noise_eeg_queries,
                'text_queries': {k: v.to(device) for k, v in batch['text_queries'].items()},
                'docs': {k: v.to(device) for k, v in batch['docs'].items()},
                'eeg_mv_masks': batch['eeg_mv_masks'].to(device),
                'metadata': batch['metadata']
            }

            # Run validation step with noise
            val_loss, val_metrics = validation_step(model, noise_batch, device)

            total_loss += val_loss
            num_batches += 1

            if is_cross_encoder:
                accuracies.append(val_metrics['accuracy'])
                neg_accuracies.append(val_metrics['negative_accuracy'])
            else:
                if hasattr(model, 'query_type') and model.query_type == 'text':
                    similarity_key = 'text_query_similarity'
                    doc_similarity_key = 'text_doc_similarity'
                else:
                    similarity_key = 'eeg_query_similarity'
                    doc_similarity_key = 'eeg_doc_similarity'

                similarities.append(val_metrics[similarity_key])
                query_doc_sims.append(val_metrics['query_doc_similarity'])
                eeg_doc_sims.append(val_metrics[doc_similarity_key])

        # Compute noise baseline statistics
        avg_loss = total_loss / num_batches

        if is_cross_encoder:
            avg_accuracy = np.mean(accuracies)
            avg_neg_accuracy = np.mean(neg_accuracies)
            main_metric = avg_accuracy

            noise_metrics = {
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'negative_accuracy': avg_neg_accuracy
            }
        else:
            avg_similarity = np.mean(similarities)
            avg_query_doc_sim = np.mean(query_doc_sims)
            avg_eeg_doc_sim = np.mean(eeg_doc_sims)
            main_metric = avg_similarity

            noise_metrics = {
                'loss': avg_loss,
                'query_doc_similarity': avg_query_doc_sim,
                'eeg_doc_similarity': avg_eeg_doc_sim
            }

            if hasattr(model, 'query_type') and model.query_type == 'text':
                noise_metrics['text_query_similarity'] = avg_similarity
                noise_metrics['text_doc_similarity'] = avg_eeg_doc_sim
            else:
                noise_metrics['eeg_query_similarity'] = avg_similarity
                noise_metrics['eeg_doc_similarity'] = avg_eeg_doc_sim

        return noise_metrics, main_metric

    finally:
        # Restore original masking probability
        test_dataloader.dataset.set_masking_probability(original_probability)


def finish_wandb():
    """Finish wandb run"""
    wandb.finish()