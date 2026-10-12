#!/usr/bin/env python3
"""
Simplified Dataloader for Brain Passage Retrieval with DYNAMIC RUNTIME MASKING
Single dataloader that can change masking probability on-the-fly
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import warnings


def create_positive_negative_pairs(batch):
    """Create positive and negative pairs for BCE training"""
    batch_size = len(batch['metadata'])
    positive_labels = torch.ones(batch_size)
    negative_pairs = []
    negative_labels = []

    for i in range(batch_size):
        available_indices = [j for j in range(batch_size) if j != i]
        if available_indices:
            neg_idx = random.choice(available_indices)
            negative_pairs.append({'eeg_idx': i, 'doc_idx': neg_idx})
            negative_labels.append(0)

    return positive_labels, negative_pairs, torch.tensor(negative_labels)


def detect_dataset_format(data_path):
    """Detect whether dataset is original or Nieuwland format"""
    try:
        dataset = np.load(data_path, allow_pickle=True).item()
        ict_pairs = dataset['ict_pairs']

        if len(ict_pairs) == 0:
            return 'original'

        first_pair = ict_pairs[0]

        # Nieuwland format has both query_eeg and doc_eeg, plus additional fields
        if (hasattr(first_pair, 'query_eeg') and hasattr(first_pair, 'doc_eeg') and
                hasattr(first_pair, 'query_words')):
            return 'nieuwland'
        elif 'query_eeg' in first_pair and 'doc_eeg' not in first_pair:
            return 'original'
        else:
            return 'original'
    except:
        return 'original'


def convert_nieuwland_to_original_format(ict_pairs):
    """Convert Nieuwland ICTPair objects to original dictionary format"""
    converted_pairs = []

    for pair in ict_pairs:
        if hasattr(pair, 'query_text'):  # ICTPair object
            converted_pair = {
                'query_text': pair.query_text,
                'query_eeg': pair.query_eeg,
                'doc_text': pair.doc_text,
                'participant_id': pair.participant_id,
                'sentence_id': pair.sentence_id,
                'is_masked': getattr(pair, 'is_masked', False),  # Legacy field
                'query_words': getattr(pair, 'query_words', []),
                'doc_words': getattr(pair, 'doc_words', []),
                'query_start_idx': getattr(pair, 'query_start_idx', 0),  # For runtime masking
                'query_end_idx': getattr(pair, 'query_end_idx', 0),  # For runtime masking
                'full_sentence_text': getattr(pair, 'full_sentence_text', pair.doc_text),
                'full_sentence_words': getattr(pair, 'full_sentence_words', getattr(pair, 'doc_words', [])),
                'fs': getattr(pair, 'fs', 500.0)
            }
        else:  # Already dictionary format
            converted_pair = pair

        converted_pairs.append(converted_pair)

    return converted_pairs


def apply_runtime_masking(doc_text: str, doc_words: List[str], query_start_idx: int,
                          query_end_idx: int, masking_probability: float,
                          random_state: Optional[int] = None) -> Tuple[str, List[str], bool]:
    """
    Apply runtime masking to document text based on query span and masking probability

    Args:
        doc_text: Full document text (unmasked)
        doc_words: Full document words (unmasked)
        query_start_idx: Start index of query span in document
        query_end_idx: End index of query span in document
        masking_probability: Probability of applying masking (0.0 = never mask, 1.0 = always mask)
        random_state: Random seed for reproducible masking decisions

    Returns:
        Tuple of (masked_doc_text, masked_doc_words, was_masked)
    """

    if random_state is not None:
        # Save current random state
        current_state = random.getstate()
        random.seed(random_state)

    try:
        # Decide whether to apply masking
        should_mask = random.random() < masking_probability

        if should_mask and query_start_idx < len(doc_words) and query_end_idx <= len(doc_words):
            # Remove query span from document
            masked_words = doc_words[:query_start_idx] + doc_words[query_end_idx:]
            masked_text = ' '.join(masked_words)
            return masked_text, masked_words, True
        else:
            # Return unmasked document
            return doc_text, doc_words, False

    finally:
        if random_state is not None:
            # Restore random state
            random.setstate(current_state)


def load_combined_datasets(data_paths, dataset_types=None):
    """Load and combine multiple datasets into a single format"""
    print(f"Loading {len(data_paths)} datasets for combination...")

    all_ict_pairs = []
    combined_metadata = {}

    for i, data_path in enumerate(data_paths):
        print(f"Loading dataset {i + 1}: {data_path}")

        # Determine dataset type
        dataset_type = dataset_types[i] if (dataset_types and i < len(dataset_types)) else 'auto'
        detected_format = detect_dataset_format(data_path) if dataset_type == 'auto' else dataset_type
        print(f"  Format: {detected_format}")

        # Load dataset
        dataset = np.load(data_path, allow_pickle=True).item()
        ict_pairs = dataset['ict_pairs']
        metadata = dataset.get('metadata', {})

        # Convert if needed
        if detected_format == 'nieuwland':
            print(f"  Converting {len(ict_pairs)} Nieuwland pairs...")
            ict_pairs = convert_nieuwland_to_original_format(ict_pairs)

        # Add source information
        for pair in ict_pairs:
            pair['dataset_source'] = f"dataset_{i + 1}_{detected_format}"

        all_ict_pairs.extend(ict_pairs)
        print(f"  Added {len(ict_pairs)} pairs (total: {len(all_ict_pairs)})")

        # Store metadata
        combined_metadata[f'dataset_{i + 1}'] = {
            'path': str(data_path), 'format': detected_format, 'count': len(ict_pairs),
            'original_metadata': metadata
        }

    print(f"Combined dataset: {len(all_ict_pairs)} total pairs")
    return all_ict_pairs, combined_metadata


def compute_combined_eeg_dimensions(all_ict_pairs, max_eeg_len=50):
    """Compute EEG dimensions for combined datasets"""
    global_max_words = 0
    global_max_time = 0
    global_max_channels = 0

    print(f"Computing global EEG dimensions across {len(all_ict_pairs)} combined samples...")

    for pair in all_ict_pairs:
        query_eeg = pair.get('query_eeg', None)
        if query_eeg is None:
            continue

        try:
            eeg_array = np.array(query_eeg, dtype=np.float32)

            if len(eeg_array.shape) == 3:
                num_words, time_samples, channels = eeg_array.shape
                if num_words > max_eeg_len:
                    num_words = max_eeg_len
                global_max_words = max(global_max_words, num_words)
                global_max_time = max(global_max_time, time_samples)
                global_max_channels = max(global_max_channels, channels)
            elif len(eeg_array.shape) == 2:
                time_samples, channels = eeg_array.shape
                global_max_words = max(global_max_words, 1)
                global_max_time = max(global_max_time, time_samples)
                global_max_channels = max(global_max_channels, channels)
        except Exception:
            continue

    print(f"Global EEG dimensions: {global_max_words}x{global_max_time}x{global_max_channels}")
    return global_max_words, global_max_time, global_max_channels


def compute_global_eeg_dimensions(data_path, max_eeg_len=50, dataset_type='auto'):
    """Compute global EEG dimensions across entire dataset with format detection"""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Detect dataset format
    detected_format = detect_dataset_format(data_path) if dataset_type == 'auto' else dataset_type
    print(f"Auto-detected dataset format: {detected_format}")

    dataset = np.load(data_path, allow_pickle=True).item()
    ict_pairs = dataset['ict_pairs']

    # Convert Nieuwland format if needed
    if detected_format == 'nieuwland':
        print(f"Converting {len(ict_pairs)} Nieuwland pairs to compatible format...")
        ict_pairs = convert_nieuwland_to_original_format(ict_pairs)

    return compute_combined_eeg_dimensions(ict_pairs, max_eeg_len)


class DynamicMaskingDataloader(Dataset):
    """
    Dataset with DYNAMIC RUNTIME MASKING support
    Single dataloader that can change masking probability on-the-fly
    """

    def __init__(self, data_path: str = None, tokenizer=None, max_text_len: int = 256,
                 max_eeg_len: int = 50, train_ratio: float = 0.8,
                 split: str = 'train', normalize_eeg: bool = True,
                 debug: bool = False, global_eeg_dims: tuple = None,
                 num_vectors: int = 32, dataset_type: str = 'auto',
                 holdout_subjects: bool = False,
                 initial_masking_probability: float = 0.9,
                 combined_dataset: dict = None,
                 fold: int = None):  # NEW PARAMETER

        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_eeg_len = max_eeg_len
        self.normalize_eeg = normalize_eeg
        self.debug = debug
        self.num_vectors = num_vectors
        self.holdout_subjects = holdout_subjects
        self.fold = fold  # NEW: Store fold number

        # DYNAMIC: Store masking probability as mutable attribute
        self._current_masking_probability = initial_masking_probability

        # Set global EEG dimensions
        if global_eeg_dims is not None:
            self.global_max_words, self.global_max_time, self.global_max_channels = global_eeg_dims
        else:
            self.global_max_words, self.global_max_time, self.global_max_channels = compute_global_eeg_dimensions(
                data_path, max_eeg_len, dataset_type)

        # Load and process ICT pairs
        if combined_dataset is not None:
            # MEMORY EFFICIENT: Use combined dataset directly (no file I/O)
            if self.debug:
                print("Using provided combined dataset (memory efficient - no temp files)")
            self.ict_pairs = combined_dataset['ict_pairs']
            self.metadata = combined_dataset.get('metadata', {})
            self.dataset_format = 'original'  # Combined datasets are already in original format
        else:
            # Original file path approach
            if data_path is None:
                raise ValueError("Either data_path or combined_dataset must be provided")

            data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")

            dataset = np.load(data_path, allow_pickle=True).item()
            self.ict_pairs = dataset['ict_pairs']
            self.metadata = dataset.get('metadata', {})

            # Handle dataset format conversion
            self.dataset_format = detect_dataset_format(data_path) if dataset_type == 'auto' else dataset_type
            if self.dataset_format == 'nieuwland':
                if self.debug:
                    print(f"Converting Nieuwland format to original format")
                self.ict_pairs = convert_nieuwland_to_original_format(self.ict_pairs)

        # Check if dataset supports runtime masking
        self.supports_runtime_masking = self.metadata.get('supports_runtime_masking', False)

        if self.debug:
            print(f"Loaded {len(self.ict_pairs)} ICT pairs")
            print(f"Initial masking probability: {initial_masking_probability}")
            print(f"Dataset supports runtime masking: {self.supports_runtime_masking}")
            if fold is not None:
                print(f"5-fold CV mode: Using fold {fold}")

        # Create train/val/test split
        random.seed(42)  # Fixed seed for reproducible splits

        if holdout_subjects:
            # Subject-based split (with optional 5-fold CV)
            if self.debug:
                if fold is not None:
                    print(f"Using 5-fold cross-validation: fold {fold}")
                else:
                    print("Using holdout subjects split (80% train, 10% val, 10% test)")
            self.indices, self.unique_subjects = self._create_subject_split(split, train_ratio, debug, fold)
        else:
            # Random sample split with train/val/test support
            if self.debug:
                print("Using random sample split")
            shuffled_indices = list(range(len(self.ict_pairs)))
            random.shuffle(shuffled_indices)

            # Support both 2-way and 3-way splits
            if split in ['train', 'val']:
                # Standard 2-way split during training
                split_idx = int(len(self.ict_pairs) * train_ratio)
                if split == 'train':
                    self.indices = shuffled_indices[:split_idx]
                elif split == 'val':
                    self.indices = shuffled_indices[split_idx:]
            elif split == 'test':
                # 3-way split: use validation portion and split it further
                val_start_idx = int(len(self.ict_pairs) * train_ratio)
                val_indices = shuffled_indices[val_start_idx:]

                # Split remaining data: 50% val, 50% test
                val_split_point = len(val_indices) // 2
                if split == 'test':
                    self.indices = val_indices[val_split_point:]  # Second half for test
                # Note: 'val' case handled above
            else:
                raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")

            # For random split, count unique subjects in this split
            self.unique_subjects = set()
            for idx in self.indices:
                participant_id = self.ict_pairs[idx].get('participant_id', 'unknown')
                self.unique_subjects.add(participant_id)

        self.pairs = [self.ict_pairs[i] for i in self.indices]

        # Process pairs ONCE - store raw data for dynamic masking
        self.processed_pairs = []
        self._process_pairs()

        if self.debug:
            print(
                f"{split} split: {len(self.processed_pairs)} samples from {len(self.unique_subjects)} unique subjects")
            if len(self.processed_pairs) > 0:
                self._debug_print_sample(0)

    def set_masking_probability(self, new_probability: float):
        """
        DYNAMIC: Change masking probability without recreating dataset

        Args:
            new_probability: New masking probability (0.0 to 1.0)
        """
        if not 0.0 <= new_probability <= 1.0:
            raise ValueError(f"Masking probability must be between 0.0 and 1.0, got {new_probability}")

        old_probability = self._current_masking_probability
        self._current_masking_probability = new_probability

        if self.debug:
            print(f"Changed masking probability: {old_probability:.2f} -> {new_probability:.2f}")

    def get_current_masking_probability(self) -> float:
        """Get current masking probability"""
        return self._current_masking_probability

    def _create_subject_split(self, split, train_ratio, debug, fold=None):
        """
        Create subject-based train/val/test split with optional 5-fold cross-validation

        Args:
            split: 'train', 'val', or 'test'
            train_ratio: Training ratio (used for original holdout, ignored for 5-fold CV)
            debug: Debug printing
            fold: If specified (1-5), use 5-fold cross-validation

        Returns:
            tuple: (selected_indices, unique_subjects_in_split)
        """

        # Extract all unique participants
        participant_to_indices = {}
        for idx, pair in enumerate(self.ict_pairs):
            participant_id = pair.get('participant_id', 'unknown')
            if participant_id not in participant_to_indices:
                participant_to_indices[participant_id] = []
            participant_to_indices[participant_id].append(idx)

        unique_participants = list(participant_to_indices.keys())
        random.shuffle(unique_participants)  # Ensures reproducible but random ordering

        if debug:
            print(f"Found {len(unique_participants)} unique participants")

        if fold is not None:
            # 5-FOLD CROSS-VALIDATION MODE
            if not (1 <= fold <= 5):
                raise ValueError(f"Fold must be between 1 and 5, got {fold}")

            # Divide participants into 5 roughly equal groups
            n_participants = len(unique_participants)
            fold_size = n_participants // 5
            remainder = n_participants % 5

            # Create 5 folds with roughly equal sizes
            folds = []
            start_idx = 0
            for fold_idx in range(5):
                # Add one extra participant to first 'remainder' folds
                current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
                end_idx = start_idx + current_fold_size
                folds.append(unique_participants[start_idx:end_idx])
                start_idx = end_idx

            # Select test participants for this fold (1-indexed)
            test_participants = folds[fold - 1]

            # Combine remaining 4 folds for train/val split
            train_val_participants = []
            for i, fold_participants in enumerate(folds):
                if i != (fold - 1):  # Skip the test fold
                    train_val_participants.extend(fold_participants)

            # Split train/val participants (87.5% train, 12.5% val of remaining subjects)
            # This gives approximately 70% train, 10% val, 20% test of total subjects
            n_train_val = len(train_val_participants)
            train_split_idx = int(n_train_val * 0.875)  # 87.5% of remaining 80%

            train_participants = train_val_participants[:train_split_idx]
            val_participants = train_val_participants[train_split_idx:]

            if debug:
                print(f"5-fold CV (fold {fold}):")
                print(
                    f"  Test participants: {len(test_participants)} ({len(test_participants) / n_participants * 100:.1f}%)")
                print(
                    f"  Train participants: {len(train_participants)} ({len(train_participants) / n_participants * 100:.1f}%)")
                print(
                    f"  Val participants: {len(val_participants)} ({len(val_participants) / n_participants * 100:.1f}%)")
                print(f"  Fold {fold} test subjects: {test_participants[:3]}..." if len(
                    test_participants) > 3 else f"  Fold {fold} test subjects: {test_participants}")

        else:
            # ORIGINAL HOLDOUT MODE (80% train, 10% val, 10% test)
            n_participants = len(unique_participants)
            train_end = int(n_participants * 0.8)
            val_end = int(n_participants * 0.9)

            train_participants = unique_participants[:train_end]
            val_participants = unique_participants[train_end:val_end]
            test_participants = unique_participants[val_end:]

            if debug:
                print(f"Original holdout split:")
                print(
                    f"  Train participants: {len(train_participants)} ({len(train_participants) / n_participants * 100:.1f}%)")
                print(
                    f"  Val participants: {len(val_participants)} ({len(val_participants) / n_participants * 100:.1f}%)")
                print(
                    f"  Test participants: {len(test_participants)} ({len(test_participants) / n_participants * 100:.1f}%)")

        # Get indices for requested split
        if split == 'train':
            selected_participants = train_participants
        elif split == 'val':
            selected_participants = val_participants
        elif split == 'test':
            selected_participants = test_participants
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")

        # Collect all indices for selected participants
        selected_indices = []
        for participant in selected_participants:
            selected_indices.extend(participant_to_indices[participant])

        if debug:
            print(f"Selected {len(selected_indices)} samples for {split} split")

        return selected_indices, set(selected_participants)
    def _process_pairs(self):
        """Process raw ICT pairs - store RAW data for dynamic masking"""
        successful_pairs = 0
        failed_pairs = 0

        for idx, pair in enumerate(self.pairs):
            try:
                processed = self._process_single_pair_raw(pair, idx)
                if processed is not None:
                    self.processed_pairs.append(processed)
                    successful_pairs += 1
                else:
                    failed_pairs += 1
            except Exception as e:
                failed_pairs += 1
                if self.debug:
                    print(f"Failed to process pair {idx}: {e}")
                continue

        if self.debug or failed_pairs > 0:
            print(f"Processed {successful_pairs} pairs successfully, {failed_pairs} failed")

    def _process_single_pair_raw(self, pair, idx):
        """Process single pair - store RAW masking info for dynamic application"""
        # Extract text components
        query_text = pair.get('query_text', '').strip()
        original_doc_text = pair.get('doc_text', '').strip()

        if not query_text or not original_doc_text:
            return None

        # Store RAW masking information (don't apply masking yet)
        query_start_idx = pair.get('query_start_idx', 0)
        query_end_idx = pair.get('query_end_idx', 0)
        full_sentence_words = pair.get('full_sentence_words', original_doc_text.split())

        # For datasets without runtime masking support, try to extract from existing data
        if not self.supports_runtime_masking:
            # Fallback: try to reconstruct query span info from query and doc text
            query_words = query_text.split()
            doc_words = original_doc_text.split()

            # Simple heuristic: find query in document
            query_start_idx = 0
            query_end_idx = len(query_words)

            # Try to find query span in document
            for i in range(len(doc_words) - len(query_words) + 1):
                if doc_words[i:i + len(query_words)] == query_words:
                    query_start_idx = i
                    query_end_idx = i + len(query_words)
                    break

            full_sentence_words = doc_words

        # Extract EEG data
        query_eeg = pair.get('query_eeg', None)
        if query_eeg is None:
            return None

        # Process EEG with global dimensions
        eeg_processed = self._process_eeg(query_eeg)
        if eeg_processed is None:
            return None

        return {
            'query_text': query_text,
            'original_doc_text': original_doc_text,
            'query_eeg': eeg_processed,
            'participant_id': pair.get('participant_id', 'unknown'),
            'sentence_id': pair.get('sentence_id', 0),
            'is_masked': pair.get('is_masked', False),
            'dataset_source': pair.get('dataset_source', self.metadata.get('version', 'unknown')),
            # FIXED: use pair's source first
            'original_idx': idx,
            # RAW masking info for dynamic application
            'query_start_idx': query_start_idx,
            'query_end_idx': query_end_idx,
            'full_sentence_words': full_sentence_words,
            'supports_runtime_masking': self.supports_runtime_masking
        }

    def _apply_dynamic_masking(self, processed_pair, idx):
        """Apply masking dynamically based on current masking probability"""
        if self._current_masking_probability <= 0:
            # No masking - return original document
            return processed_pair['original_doc_text'], processed_pair['full_sentence_words'], False

        # Apply runtime masking with current probability
        random_seed = hash((idx, self._current_masking_probability)) % (2 ** 31)

        doc_text, doc_words, was_masked = apply_runtime_masking(
            doc_text=processed_pair['original_doc_text'],
            doc_words=processed_pair['full_sentence_words'],
            query_start_idx=processed_pair['query_start_idx'],
            query_end_idx=processed_pair['query_end_idx'],
            masking_probability=self._current_masking_probability,
            random_state=random_seed
        )

        return doc_text, doc_words, was_masked

    def _process_eeg(self, eeg_data):
        """Process EEG data to consistent format with global padding"""
        try:
            eeg_array = np.array(eeg_data, dtype=np.float32)

            if len(eeg_array.shape) == 3:
                # 3D format: [num_words, time_samples, channels]
                num_words, time_samples, channels = eeg_array.shape
                if num_words > self.max_eeg_len:
                    eeg_array = eeg_array[:self.max_eeg_len]
                    num_words = self.max_eeg_len

                padded_eeg = np.zeros((self.global_max_words, self.global_max_time, self.global_max_channels),
                                      dtype=np.float32)
                padded_eeg[:num_words, :time_samples, :channels] = eeg_array
                return padded_eeg

            elif len(eeg_array.shape) == 2:
                # 2D format: [time_samples, channels]
                time_samples, channels = eeg_array.shape
                padded_eeg = np.zeros((self.global_max_words, self.global_max_time, self.global_max_channels),
                                      dtype=np.float32)
                padded_eeg[0, :time_samples, :channels] = eeg_array
                return padded_eeg

            else:
                # Try to reshape flattened data
                flattened = eeg_array.flatten()
                for channels in [32, 63, 64, 128, 256]:
                    if len(flattened) % channels == 0:
                        time_samples = len(flattened) // channels
                        if time_samples >= 10:
                            reshaped = flattened.reshape(time_samples, channels)
                            padded_eeg = np.zeros(
                                (self.global_max_words, self.global_max_time, self.global_max_channels),
                                dtype=np.float32)
                            padded_eeg[0, :time_samples, :channels] = reshaped
                            return padded_eeg

                if self.debug:
                    print(f"Could not reshape EEG with shape {eeg_array.shape}")
                return None

        except Exception as e:
            if self.debug:
                print(f"EEG processing failed: {e}")
            return None

    def _normalize_eeg(self, eeg_tensor):
        """Normalize EEG tensor for better training stability"""
        if not self.normalize_eeg:
            return eeg_tensor

        if len(eeg_tensor.shape) == 3:
            mean = torch.mean(eeg_tensor, dim=1, keepdim=True)
            std = torch.std(eeg_tensor, dim=1, keepdim=True)
            std = torch.where(std == 0, torch.tensor(1e-6), std)
            return (eeg_tensor - mean) / std
        else:
            return eeg_tensor

    def _tokenize_text(self, text, add_special_tokens=True, is_query=False):
        """Tokenize text with ColBERT-style query augmentation for multi-vector"""
        if is_query:
            # Query processing with [MASK] augmentation
            encoded = self.tokenizer.encode_plus(
                text, add_special_tokens=add_special_tokens,
                max_length=self.max_text_len - 10, truncation=True, return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze()
            attention_mask = encoded['attention_mask'].squeeze()

            # Add [MASK] tokens
            mask_token_id = self.tokenizer.mask_token_id
            current_len = attention_mask.sum().item()

            if current_len < self.max_text_len:
                num_masks = self.max_text_len - current_len
                sep_pos = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                insert_pos = sep_pos[0].item() if len(sep_pos) > 0 else current_len

                new_input_ids = torch.zeros(self.max_text_len, dtype=input_ids.dtype)
                new_attention_mask = torch.zeros(self.max_text_len, dtype=attention_mask.dtype)

                new_input_ids[:insert_pos] = input_ids[:insert_pos]
                new_attention_mask[:insert_pos] = attention_mask[:insert_pos]
                new_input_ids[insert_pos:insert_pos + num_masks] = mask_token_id
                new_attention_mask[insert_pos:insert_pos + num_masks] = 1

                remaining_len = current_len - insert_pos
                if remaining_len > 0:
                    new_input_ids[insert_pos + num_masks:insert_pos + num_masks + remaining_len] = input_ids[
                                                                                                   insert_pos:current_len]
                    new_attention_mask[insert_pos + num_masks:insert_pos + num_masks + remaining_len] = attention_mask[
                                                                                                        insert_pos:current_len]

                input_ids = new_input_ids
                attention_mask = new_attention_mask
            else:
                padded_input_ids = torch.zeros(self.max_text_len, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros(self.max_text_len, dtype=attention_mask.dtype)
                padded_input_ids[:len(input_ids)] = input_ids
                padded_attention_mask[:len(attention_mask)] = attention_mask
                input_ids = padded_input_ids
                attention_mask = padded_attention_mask
        else:
            # Document processing
            encoded = self.tokenizer.encode_plus(
                text, add_special_tokens=add_special_tokens, max_length=self.max_text_len,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze()
            attention_mask = encoded['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def _debug_print_sample(self, idx):
        """Print detailed debug information for a sample"""
        sample = self.processed_pairs[idx]
        print(f"\nSample {idx}:")
        print(f"  Query text: '{sample['query_text'][:100]}...'")
        print(f"  Original doc: '{sample['original_doc_text'][:100]}...'")
        print(f"  EEG shape: {sample['query_eeg'].shape}")
        print(f"  Participant: {sample['participant_id']}")
        print(f"  Current masking probability: {self._current_masking_probability}")
        print(f"  Supports runtime masking: {sample['supports_runtime_masking']}")

    def __len__(self):
        return len(self.processed_pairs)

    def __getitem__(self, idx):
        """Get sample with DYNAMIC masking applied"""
        sample = self.processed_pairs[idx]

        # DYNAMIC: Apply masking based on current probability
        doc_text, doc_words, was_masked = self._apply_dynamic_masking(sample, idx)

        # Tokenize texts (using dynamically masked document text)
        query_tokens = self._tokenize_text(sample['query_text'], is_query=True)
        doc_tokens = self._tokenize_text(doc_text, is_query=False)

        # Process EEG
        eeg_tensor = torch.tensor(sample['query_eeg'], dtype=torch.float32)
        eeg_tensor = self._normalize_eeg(eeg_tensor)

        # Multi-vector mask
        mv_mask = torch.ones(self.num_vectors, dtype=torch.float32)

        return {
            'eeg_query': eeg_tensor,
            'text_query': {
                'input_ids': query_tokens['input_ids'],
                'attention_mask': query_tokens['attention_mask'],
                'mv_mask': mv_mask
            },
            'doc': {
                'input_ids': doc_tokens['input_ids'],
                'attention_mask': doc_tokens['attention_mask'],
                'mv_mask': mv_mask
            },
            'eeg_mv_mask': mv_mask,
            'metadata': {
                'participant_id': sample['participant_id'],
                'sentence_id': sample['sentence_id'],
                'is_masked': sample['is_masked'],
                'was_masked': was_masked,  # NEW: Runtime masking result
                'current_masking_probability': self._current_masking_probability,  # NEW
                'dataset_source': sample['dataset_source'],
                'original_idx': sample['original_idx'],
                'query_text': sample['query_text'],
                'document_text': doc_text,  # Dynamically masked
                'original_document_text': sample['original_doc_text'],  # Always unmasked
                'num_vectors': self.num_vectors
            }
        }


def simple_collate_fn(batch):
    """Simple collate function for batching"""
    # Stack tensors
    eeg_queries = torch.stack([item['eeg_query'] for item in batch])

    text_queries = {
        'input_ids': torch.stack([item['text_query']['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['text_query']['attention_mask'] for item in batch]),
        'mv_mask': torch.stack([item['text_query']['mv_mask'] for item in batch])
    }

    docs = {
        'input_ids': torch.stack([item['doc']['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['doc']['attention_mask'] for item in batch]),
        'mv_mask': torch.stack([item['doc']['mv_mask'] for item in batch])
    }

    eeg_mv_masks = torch.stack([item['eeg_mv_mask'] for item in batch])
    metadata = [item['metadata'] for item in batch]

    return {
        'eeg_queries': eeg_queries, 'text_queries': text_queries, 'docs': docs,
        'eeg_mv_masks': eeg_mv_masks, 'metadata': metadata
    }


def debug_batch(batch, print_texts=True, print_shapes=True):
    """Debug utility to inspect a batch"""
    batch_size = len(batch['metadata'])
    print(f"\nBatch size: {batch_size}")

    if print_shapes:
        print("Tensor shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape}")

    if print_texts and batch_size > 0:
        meta = batch['metadata'][0]
        print(f"Sample texts:")
        print(f"  Query: '{meta['query_text'][:100]}...'")
        print(f"  Document: '{meta['document_text'][:100]}...'")
        if meta.get('original_document_text') != meta['document_text']:
            print(f"  Original Document: '{meta['original_document_text'][:100]}...'")
        print(f"  Participant: {meta['participant_id']}")
        print(f"  Current masking probability: {meta.get('current_masking_probability', 'N/A')}")
        print(f"  Was masked: {meta.get('was_masked', 'N/A')}")


# Convenience alias for backward compatibility
SimplifiedDataloader = DynamicMaskingDataloader