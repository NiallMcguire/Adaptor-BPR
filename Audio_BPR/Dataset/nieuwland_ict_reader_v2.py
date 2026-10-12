#!/usr/bin/env python3
"""
CORRECTED Complete Word-Level EEG Alignment Dataloader with RUNTIME MASKING SUPPORT
✅ Fixed trigger extraction bug (unique triggers only)
✅ Removed duplicate methods
✅ Now generates unmasked ICT pairs with metadata for runtime masking
✅ Compatible with multi-masking validation (0%, 25%, 50%, 75%, 90%, 100%)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import mne
import torch
from typing import Dict, List, Tuple, Optional
import warnings
from collections import defaultdict
from tqdm import tqdm
import random
from dataclasses import dataclass
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Set global seeds for reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)


@dataclass
class ICTPair:
    """Data class for ICT query-document pairs with RUNTIME MASKING SUPPORT"""
    query_text: str
    query_eeg: np.ndarray
    query_words: List[str]
    doc_text: str  # ALWAYS UNMASKED - full sentence
    doc_eeg: np.ndarray  # ALWAYS UNMASKED - full sentence EEG
    doc_words: List[str]  # ALWAYS UNMASKED - full sentence words
    participant_id: str
    sentence_id: int
    query_start_idx: int  # For runtime masking
    query_end_idx: int  # For runtime masking
    full_sentence_text: str  # Store original full sentence
    full_sentence_words: List[str]  # Store original full sentence words
    fs: float


class CorrectedCompleteWordEEGAligner:
    """
    CORRECTED VERSION - Fixed trigger extraction + RUNTIME MASKING SUPPORT

    Key corrections:
    1. ✅ FIXED trigger extraction (unique triggers only)
    2. ✅ Removed duplicate methods
    3. ✅ Added runtime masking support for multi-masking validation
    4. ✅ Realistic expectations (~51 participants, ~4,067 sentences)
    """

    def __init__(self, data_dir: str, sentence_materials_dir: str = None, random_seed: int = 42,
                 limit_participants: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.sentence_materials_dir = Path(sentence_materials_dir) if sentence_materials_dir else None
        self.random_seed = random_seed
        self.limit_participants = limit_participants

        # Set seeds for reproducibility
        self._set_seeds(random_seed)

        # Nieuwland timing parameters
        self.word_display_ms = 200
        self.blank_screen_ms = 300
        self.total_word_duration_ms = 500

        # Data containers
        self.participants = []
        self.sentence_mapping = {}
        self.all_word_eeg_pairs = []

        # Metadata for reproducibility - ENHANCED for runtime masking
        self.metadata = {
            'creation_date': datetime.now().isoformat(),
            'random_seed': random_seed,
            'data_dir': str(data_dir),
            'sentence_materials_dir': str(sentence_materials_dir) if sentence_materials_dir else None,
            'word_display_ms': self.word_display_ms,
            'blank_screen_ms': self.blank_screen_ms,
            'total_word_duration_ms': self.total_word_duration_ms,
            'limit_participants': limit_participants,
            'version': 'CORRECTED_RUNTIME_MASKING_v1.0',
            'supports_runtime_masking': True,
            'masking_method': 'runtime_query_span_removal'
        }

        print(f"📧 CORRECTED Nieuwland EEG Aligner - Runtime Masking Support!")
        print(f"🎲 Random seed: {random_seed}")

        # Scan for participants
        self._scan_participants(limit_participants)

        # Load sentence materials
        if self.sentence_materials_dir and self.sentence_materials_dir.exists():
            self._load_sentence_materials()

    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _scan_participants(self, limit_participants: Optional[int] = None):
        """Scan for participants"""
        print("📝 Scanning for participants...")

        participant_files = defaultdict(dict)

        # Use sorted() to ensure consistent order
        for file_path in sorted(self.data_dir.glob("seg_*")):
            if file_path.is_file():
                filename = file_path.name
                participant_id = filename[4:].split('.')[0]
                ext = filename.split('.')[-1]
                participant_files[participant_id][ext] = file_path

        # Process participants in deterministic order
        participants_added = 0
        for pid in sorted(participant_files.keys()):
            if limit_participants is not None and participants_added >= limit_participants:
                break

            files = participant_files[pid]
            has_dat = 'dat' in files
            has_triggers = any(trigger_type in files for trigger_type in ['vmrk', 'ehst2', 'vhdr'])

            if has_dat and has_triggers:
                # Determine trigger file format
                trigger_format = None
                if 'vmrk' in files:
                    trigger_format = 'vmrk'
                elif 'ehst2' in files:
                    trigger_format = 'ehst2'
                elif 'vhdr' in files:
                    trigger_format = 'vhdr'

                participant_data = {
                    'id': pid,
                    'files': files,
                    'trigger_format': trigger_format
                }

                self.participants.append(participant_data)
                print(f"✅ Found participant: {pid} with {trigger_format} triggers")
                participants_added += 1

        print(f"📊 Total participants loaded: {len(self.participants)}")

        # Update metadata
        if self.participants:
            self.metadata['participants_processed'] = len(self.participants)

    def _load_sentence_materials(self):
        """Load sentence materials"""
        replication_items_path = self.sentence_materials_dir / "REPLICATION_ITEMS.xlsx"
        if replication_items_path.exists():
            try:
                replication_items = pd.read_excel(replication_items_path)
                print(f"✅ Loaded sentence materials: {replication_items.shape}")

                # Create sentence mapping - use sorted iteration for reproducibility
                for idx in sorted(replication_items.index):
                    row = replication_items.iloc[idx]
                    try:
                        item_num = int(row['Item Number'])

                        # Build full sentence
                        context = str(row['Sentence context']).strip()
                        expected_article = str(row['Expected']).strip()
                        expected_noun = str(row['Expected.1']).strip()
                        ending = str(row['Sentence Ending']).strip() if pd.notna(row['Sentence Ending']) else ""

                        full_sentence = f"{context} {expected_article} {expected_noun} {ending}".strip()
                        words = full_sentence.split()

                        self.sentence_mapping[item_num] = {
                            'full_sentence': full_sentence,
                            'words': words,
                            'word_count': len(words)
                        }
                    except Exception as e:
                        continue

                print(f"✅ Created sentence mapping for {len(self.sentence_mapping)} items")
                self.metadata['sentence_count'] = len(self.sentence_mapping)

            except Exception as e:
                print(f"⚠ Error loading sentence materials: {e}")

    # 📧 FIXED TRIGGER EXTRACTION METHODS (extract unique triggers only)

    def _extract_sentence_triggers_vmrk(self, vmrk_path: Path) -> List[Tuple[int, int, int]]:
        """FIXED: Extract UNIQUE sentence triggers from .vmrk file"""
        sentence_triggers = []
        seen_triggers = set()  # ✅ KEY FIX: Track to avoid duplicates

        try:
            with open(vmrk_path, 'r', encoding='latin-1') as f:
                for line in f:
                    if line.startswith('Mk') and '=' in line:
                        parts = line.split('=')[1].split(',')
                        if len(parts) >= 3:
                            trigger_code = parts[1].strip()
                            sample_pos = int(parts[2]) if parts[2].isdigit() else 0

                            if trigger_code.startswith('S'):
                                try:
                                    code = int(trigger_code[1:])
                                    if 101 <= code <= 180:
                                        item_num = code - 100

                                        # ✅ KEY FIX: Only first occurrence of each item
                                        if item_num not in seen_triggers:
                                            sentence_triggers.append((sample_pos, code, item_num))
                                            seen_triggers.add(item_num)

                                except ValueError:
                                    continue
        except Exception as e:
            print(f"⚠ Error reading .vmrk: {e}")

        return sentence_triggers

    def _extract_sentence_triggers_ehst2(self, ehst2_path: Path) -> List[Tuple[int, int, int]]:
        """FIXED: Extract UNIQUE sentence triggers from .ehst2 file"""
        sentence_triggers = []
        seen_triggers = set()  # ✅ KEY FIX: Track to avoid duplicates

        try:
            with open(ehst2_path, 'rb') as f:
                raw_data = f.read()

            int16_data = np.frombuffer(raw_data, dtype=np.int16)

            for i, val in enumerate(int16_data):
                if 101 <= val <= 180:
                    sample_pos = i * 2
                    code = int(val)
                    item_num = code - 100

                    # ✅ KEY FIX: Only first occurrence of each item
                    if item_num not in seen_triggers:
                        sentence_triggers.append((sample_pos, code, item_num))
                        seen_triggers.add(item_num)

        except Exception as e:
            print(f"⚠ Error reading .ehst2: {e}")

        return sentence_triggers

    def _extract_sentence_triggers_vhdr(self, vhdr_path: Path) -> List[Tuple[int, int, int]]:
        """FIXED: Extract UNIQUE sentence triggers from .vhdr file"""
        sentence_triggers = []
        seen_triggers = set()  # ✅ KEY FIX: Track to avoid duplicates

        try:
            with open(vhdr_path, 'r', encoding='latin-1') as f:
                content = f.read()

            in_marker_section = False
            for line in content.split('\n'):
                line = line.strip()

                if line.startswith('[Marker Infos]'):
                    in_marker_section = True
                    continue
                elif line.startswith('[') and in_marker_section:
                    break

                if in_marker_section and '=' in line and line.startswith('Mk'):
                    try:
                        parts = line.split('=')[1].split(',')
                        if len(parts) >= 3:
                            trigger_part = parts[1].strip()
                            sample_pos = int(parts[2]) if parts[2].isdigit() else 0

                            if trigger_part.startswith('S'):
                                code = int(trigger_part[1:])
                                if 101 <= code <= 180:
                                    item_num = code - 100

                                    # ✅ KEY FIX: Only first occurrence of each item
                                    if item_num not in seen_triggers:
                                        sentence_triggers.append((sample_pos, code, item_num))
                                        seen_triggers.add(item_num)

                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            print(f"⚠ Error reading .vhdr: {e}")

        return sentence_triggers

    def _read_dat_file(self, dat_path: Path) -> Optional[np.ndarray]:
        """Read EEG data from .dat file"""
        channel_options = [32, 64, 128, 16, 8]
        dtype_options = [np.int16, np.float32, np.float64]

        for channels in channel_options:
            for dtype in dtype_options:
                try:
                    with open(dat_path, 'rb') as f:
                        data = np.frombuffer(f.read(), dtype=dtype)

                    if len(data) % channels == 0:
                        n_samples = len(data) // channels
                        if n_samples > 1000:
                            eeg_data = data.reshape((n_samples, channels)).T

                            if not np.all(eeg_data == 0) and np.isfinite(eeg_data).all():
                                print(f"   ✅ EEG data loaded: {eeg_data.shape} ({channels}ch, {dtype})")
                                return eeg_data

                except Exception:
                    continue

        print(f"   ⚠ Failed to load EEG data")
        return None

    def _calculate_word_timings(self, sentence_start_sample: int, word_count: int, fs: float) -> List[int]:
        """Calculate start samples for each word using fixed 500ms timing"""
        word_duration_samples = int(self.total_word_duration_ms * fs / 1000)
        return [sentence_start_sample + (i * word_duration_samples) for i in range(word_count)]

    def _extract_word_eeg(self, eeg_data: np.ndarray, word_start_sample: int,
                          window_ms: int = 500, fs: float = 500.0) -> Optional[np.ndarray]:
        """Extract EEG segment for a specific word"""
        window_samples = int(window_ms * fs / 1000)

        start_sample = max(0, word_start_sample)
        end_sample = min(eeg_data.shape[1], word_start_sample + window_samples)

        if end_sample <= start_sample:
            return None

        word_eeg = eeg_data[:, start_sample:end_sample]
        return word_eeg.T  # [samples, channels]

    def align_all_words_to_eeg(self, max_sentences_per_participant: Optional[int] = None):
        """CORRECTED: Word-EEG alignment using FIXED trigger extraction"""

        if not self.participants:
            print("⚠ No participants found")
            return

        print(f"\n🧠 CORRECTED WORD-EEG ALIGNMENT")
        print("=" * 60)
        print(f"📊 Participants: {len(self.participants)}")
        print(f"🎲 Random seed: {self.random_seed}")

        total_sentences_processed = 0
        total_words_aligned = 0

        for participant_idx, participant in enumerate(self.participants):
            print(f"\n{'=' * 20} PARTICIPANT {participant_idx + 1}/{len(self.participants)} {'=' * 20}")
            print(f"Participant: {participant['id']}")

            # Define sampling rate
            fs = 500.0

            # Extract sentence triggers using FIXED methods
            trigger_format = participant['trigger_format']
            if trigger_format == 'vmrk':
                sentence_triggers = self._extract_sentence_triggers_vmrk(participant['files']['vmrk'])
            elif trigger_format == 'ehst2':
                sentence_triggers = self._extract_sentence_triggers_ehst2(participant['files']['ehst2'])
            elif trigger_format == 'vhdr':
                sentence_triggers = self._extract_sentence_triggers_vhdr(participant['files']['vhdr'])
            else:
                print("⚠ Unknown trigger format")
                continue

            print(f"📊 Unique sentence triggers found: {len(sentence_triggers)}")

            # Validate trigger extraction
            if len(sentence_triggers) == 0:
                print("⚠️ No triggers found - skipping participant")
                continue
            elif len(sentence_triggers) > 100:
                print(f"⚠️ Too many triggers ({len(sentence_triggers)}) - check extraction logic")
                continue

            # Load EEG data
            eeg_data = self._read_dat_file(participant['files']['dat'])
            if eeg_data is None:
                print("⚠ Failed to load EEG data")
                continue

            # Process sentences in deterministic order
            participant_sentences_processed = 0
            participant_words_aligned = 0

            for sample_pos, trigger_code, item_num in sorted(sentence_triggers):
                if max_sentences_per_participant is not None and participant_sentences_processed >= max_sentences_per_participant:
                    break

                if item_num not in self.sentence_mapping:
                    continue

                sentence_info = self.sentence_mapping[item_num]
                full_sentence = sentence_info['full_sentence']
                words = sentence_info['words']
                word_count = sentence_info['word_count']

                # Calculate timing for ALL words
                word_timings = self._calculate_word_timings(sample_pos, word_count, fs)

                # Extract EEG for every word
                sentence_word_data = []
                for word_idx, (word, word_start_sample) in enumerate(zip(words, word_timings)):
                    word_eeg = self._extract_word_eeg(eeg_data, word_start_sample, window_ms=500, fs=fs)

                    if word_eeg is not None:
                        word_time = word_start_sample / fs

                        word_data = {
                            'word': word,
                            'word_eeg': word_eeg,
                            'word_position': word_idx,
                            'word_time': word_time,
                            'sentence_id': item_num,
                            'full_sentence': full_sentence,
                            'participant_id': participant['id'],
                            'fs': fs
                        }
                        sentence_word_data.append(word_data)
                        participant_words_aligned += 1

                # Store sentence data for ICT generation
                if sentence_word_data:
                    sentence_data = {
                        'sentence_id': item_num,
                        'participant_id': participant['id'],
                        'full_sentence': full_sentence,
                        'words': words,
                        'word_data': sentence_word_data,
                        'fs': fs
                    }
                    self.all_word_eeg_pairs.append(sentence_data)

                participant_sentences_processed += 1
                total_sentences_processed += 1

            total_words_aligned += participant_words_aligned
            print(
                f"   ✅ Participant {participant['id']}: {participant_sentences_processed} sentences, {participant_words_aligned} words")

        # Update metadata
        self.metadata['processed_sentences'] = total_sentences_processed
        self.metadata['total_word_eeg_pairs'] = total_words_aligned

        print("\n" + "=" * 60)
        print("🎉 CORRECTED WORD-EEG ALIGNMENT COMPLETE!")
        print(f"📊 TOTAL STATS:")
        print(f"   👥 Participants processed: {len(self.participants)}")
        print(f"   📝 Total sentences: {total_sentences_processed}")
        print(f"   🧠 Total words aligned: {total_words_aligned}")
        print(f"🎲 Random seed: {self.random_seed}")

    def generate_ict_pairs(self, min_query_length: int = 2, max_query_length: int = 5,
                           query_length_ratio: float = 0.3, min_sentence_length: int = 8,
                           use_ratio_based_queries: bool = True, max_pairs_per_sentence: int = 2,
                           max_total_pairs: Optional[int] = None, random_seed: Optional[int] = None) -> List[ICTPair]:
        """Generate ICT pairs from word-level EEG alignments with RUNTIME MASKING SUPPORT"""

        if random_seed is None:
            random_seed = self.random_seed

        self._set_seeds(random_seed)

        if not self.all_word_eeg_pairs:
            print("⚠ No word-EEG data available. Run align_all_words_to_eeg() first.")
            return []

        print(
            f"📧 Generating ICT pairs with RUNTIME MASKING SUPPORT from {len(self.all_word_eeg_pairs)} sentences (seed: {random_seed})...")
        print(f"📊 PARAMETERS:")
        print(f"   min_query_length: {min_query_length}")
        print(f"   max_query_length: {max_query_length}")
        print(f"   query_length_ratio: {query_length_ratio}")
        print(f"   min_sentence_length: {min_sentence_length}")
        print(f"   use_ratio_based_queries: {use_ratio_based_queries}")
        print(f"   max_pairs_per_sentence: {max_pairs_per_sentence}")
        print(f"🎭 MASKING: Will be applied at RUNTIME (supports 0%-100%)")

        ict_pairs = []

        # Store generation parameters
        generation_params = {
            'min_query_length': min_query_length, 'max_query_length': max_query_length,
            'query_length_ratio': query_length_ratio, 'min_sentence_length': min_sentence_length,
            'use_ratio_based_queries': use_ratio_based_queries, 'max_pairs_per_sentence': max_pairs_per_sentence,
            'max_total_pairs': max_total_pairs, 'random_seed': random_seed,
            'masking_applied_at_generation': False,  # NEW: No masking at generation
            'supports_runtime_masking': True  # NEW: Runtime masking support
        }
        self.metadata['ict_generation_params'] = generation_params

        # Process sentences in deterministic order
        for sentence_data in tqdm(sorted(self.all_word_eeg_pairs, key=lambda x: x['sentence_id']),
                                  desc="Creating ICT pairs"):
            try:
                words = sentence_data['words']
                word_data = sentence_data['word_data']

                # Skip short sentences
                if len(words) < min_sentence_length:
                    continue

                # Generate multiple pairs per sentence
                sentence_pairs = 0
                attempts = 0
                max_attempts = max_pairs_per_sentence * 5

                while sentence_pairs < max_pairs_per_sentence and attempts < max_attempts:
                    pair = self._create_ict_pair_from_sentence(sentence_data, min_query_length, max_query_length,
                                                               query_length_ratio, use_ratio_based_queries)

                    if pair is not None:
                        ict_pairs.append(pair)
                        sentence_pairs += 1

                        # Check if we've reached the total limit
                        if max_total_pairs and len(ict_pairs) >= max_total_pairs:
                            break

                    attempts += 1

                # Early termination if max pairs reached
                if max_total_pairs and len(ict_pairs) >= max_total_pairs:
                    break

            except Exception as e:
                print(f"⚠ Error generating ICT pair: {e}")
                continue

        print(
            f"🎉 Generated {len(ict_pairs)} ICT pairs with RUNTIME MASKING SUPPORT (reproducible with seed {random_seed})")
        self._print_ict_statistics(ict_pairs)

        # Update metadata
        self.metadata['generated_ict_pairs'] = len(ict_pairs)
        return ict_pairs

    def _create_ict_pair_from_sentence(self, sentence_data: Dict, min_query_length: int, max_query_length: int,
                                       query_length_ratio: float, use_ratio_based_queries: bool) -> Optional[ICTPair]:
        """
        Create a single ICT pair with RUNTIME MASKING SUPPORT

        ✅ KEY CHANGES FOR RUNTIME MASKING:
        - Always stores UNMASKED document (full sentence)
        - Stores query span metadata for runtime masking
        - No masking applied at generation time
        """
        words = sentence_data['words']
        word_data = sentence_data['word_data']
        sentence_length = len(words)

        # Determine query length (same logic as before)
        if use_ratio_based_queries:
            query_length = max(min_query_length, int(sentence_length * query_length_ratio))
            query_length = min(query_length, sentence_length - 1)
        else:
            query_length = min(random.randint(min_query_length, max_query_length), sentence_length - 1)

        # Select query span (same logic as before)
        max_start_idx = sentence_length - query_length
        query_start_idx = random.randint(0, max_start_idx)
        query_end_idx = query_start_idx + query_length

        # Extract query text and words
        query_words = words[query_start_idx:query_end_idx]
        query_word_data = word_data[query_start_idx:query_end_idx]
        query_text = ' '.join(query_words)

        # ✅ Store ALL query word EEGs as array
        query_eegs = []
        for wd in query_word_data:
            if wd['word_eeg'] is not None:
                query_eegs.append(wd['word_eeg'])
            else:
                return None  # Skip if any word EEG is missing

        if not query_eegs:
            return None

        # Convert to numpy array: shape (num_words, 250, 32)
        query_eeg = np.array(query_eegs)

        # ✅ RUNTIME MASKING: ALWAYS store UNMASKED document (full sentence)
        doc_words = words[:]  # Full sentence words
        doc_word_data = word_data[:]  # Full sentence EEG data
        doc_text = ' '.join(doc_words)  # Full sentence text

        # ✅ Store ALL document word EEGs as array (UNMASKED)
        doc_eegs = []
        for wd in doc_word_data:
            if wd['word_eeg'] is not None:
                doc_eegs.append(wd['word_eeg'])

        if not doc_eegs:
            # Fallback: create dummy array matching query structure
            doc_eeg = np.zeros((len(words), query_eeg.shape[1], query_eeg.shape[2]))
        else:
            doc_eeg = np.array(doc_eegs)

        # ✅ Create ICT pair with RUNTIME MASKING SUPPORT
        return ICTPair(
            query_text=query_text,
            query_eeg=query_eeg,
            query_words=query_words,
            doc_text=doc_text,  # ✅ ALWAYS UNMASKED - full sentence
            doc_eeg=doc_eeg,  # ✅ ALWAYS UNMASKED - full sentence EEG
            doc_words=doc_words,  # ✅ ALWAYS UNMASKED - full sentence words
            participant_id=sentence_data['participant_id'],
            sentence_id=sentence_data['sentence_id'],
            query_start_idx=query_start_idx,  # ✅ For runtime masking
            query_end_idx=query_end_idx,  # ✅ For runtime masking
            full_sentence_text=doc_text,  # ✅ Store full sentence
            full_sentence_words=doc_words,  # ✅ Store full sentence words
            fs=sentence_data['fs']
        )

    def _print_ict_statistics(self, ict_pairs: List[ICTPair]):
        """Print statistics about generated ICT pairs - ENHANCED for runtime masking"""
        if not ict_pairs:
            return

        participants = set(pair.participant_id for pair in ict_pairs)

        query_lengths = [len(pair.query_words) for pair in ict_pairs]
        doc_lengths = [len(pair.doc_words) for pair in ict_pairs]

        query_eeg_lengths = [pair.query_eeg.shape[0] for pair in ict_pairs]
        doc_eeg_lengths = [pair.doc_eeg.shape[0] for pair in ict_pairs]

        print(f"\n📊 ICT PAIR STATISTICS (Runtime Masking Ready):")
        print(f"   👥 Participants: {len(participants)}")
        print(f"   🎭 All pairs UNMASKED (masking applied at runtime)")
        print(f"   📝 Query length: {np.mean(query_lengths):.1f} ± {np.std(query_lengths):.1f} words")
        print(f"   📄 Document length: {np.mean(doc_lengths):.1f} ± {np.std(doc_lengths):.1f} words")
        print(f"   🧠 Query EEG length: {np.mean(query_eeg_lengths):.1f} ± {np.std(query_eeg_lengths):.1f} samples")
        print(f"   🧠 Document EEG length: {np.mean(doc_eeg_lengths):.1f} ± {np.std(doc_eeg_lengths):.1f} samples")
        print(f"   ✅ Supports masking levels: 0%, 25%, 50%, 75%, 90%, 100%")

    def save_ict_pairs_with_metadata(self, ict_pairs: List[ICTPair], save_path: str):
        """Save ICT pairs to disk with comprehensive metadata - ENHANCED for runtime masking"""
        save_path = Path(save_path)

        print(f"💾 Saving {len(ict_pairs)} ICT pairs with RUNTIME MASKING SUPPORT to {save_path}")

        # Convert ICT pairs to serializable format
        pairs_data = []
        for pair in ict_pairs:
            pair_dict = {
                'query_text': pair.query_text,
                'query_eeg': pair.query_eeg,
                'query_words': pair.query_words,
                'doc_text': pair.doc_text,  # UNMASKED
                'doc_eeg': pair.doc_eeg,  # UNMASKED
                'doc_words': pair.doc_words,  # UNMASKED
                'participant_id': pair.participant_id,
                'sentence_id': pair.sentence_id,
                'query_start_idx': pair.query_start_idx,  # For runtime masking
                'query_end_idx': pair.query_end_idx,  # For runtime masking
                'full_sentence_text': pair.full_sentence_text,
                'full_sentence_words': pair.full_sentence_words,
                'fs': pair.fs
            }
            pairs_data.append(pair_dict)

        # Create comprehensive dataset
        dataset = {
            'ict_pairs': pairs_data,
            'metadata': self.metadata,
            'version': '2.0',
            'description': 'Nieuwland EEG-text ICT pairs with RUNTIME MASKING SUPPORT - compatible with multi-masking validation'
        }

        # Save as .npy file
        if save_path.suffix != '.npy':
            save_path = save_path.with_suffix('.npy')

        np.save(save_path, dataset)

        # Also save metadata as JSON for easy inspection
        metadata_path = save_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✅ Saved ICT pairs to: {save_path}")
        print(f"✅ Saved metadata to: {metadata_path}")
        print(f"🎲 Reproducible with seed: {self.metadata['random_seed']}")
        print(f"🎭 Ready for multi-masking validation (0%, 25%, 50%, 75%, 90%, 100%)")


# Main execution with RUNTIME MASKING SUPPORT
if __name__ == "__main__":
    print("🚀 CORRECTED NIEUWLAND ICT PAIR GENERATION - RUNTIME MASKING SUPPORT")
    print("=" * 60)

    # Configuration - RUNTIME MASKING READY
    RANDOM_SEED = 42
    LIMIT_PARTICIPANTS = None  # Use all available participants
    MAX_SENTENCES_PER_PARTICIPANT = None  # Use all sentences
    MAX_ICT_PAIRS = None  # Generate all possible ICT pairs

    # File paths
    data_dir = "/users/gxb18167/Processed segmented data"
    sentence_materials_dir = "/users/gxb18167/Sentence Materials"
    output_file = "nieuwland_ict_pairs_RUNTIME_MASKING.npy"

    print(f"📊 EXPECTED OUTPUT (based on your previous run):")
    print(f"   👥 Participants: ~51")
    print(f"   📝 Sentences: ~4,067")
    print(f"   🧠 Words: ~91,383")
    print(f"   🎯 ICT pairs: ~8,000-10,000")

    print(f"📧 RUNTIME MASKING FEATURES:")
    print(f"   ✅ No masking applied at generation time")
    print(f"   ✅ Stores query span metadata for runtime masking")
    print(f"   ✅ Compatible with any masking probability (0%-100%)")
    print(f"   ✅ Supports multi-masking validation during training")

    # Initialize the corrected aligner
    aligner = CorrectedCompleteWordEEGAligner(
        data_dir=data_dir,
        sentence_materials_dir=sentence_materials_dir,
        random_seed=RANDOM_SEED,
        limit_participants=LIMIT_PARTICIPANTS
    )

    # Step 1: Align words to EEG
    print(f"\n📬 Step 1: Aligning words to EEG...")
    aligner.align_all_words_to_eeg(max_sentences_per_participant=MAX_SENTENCES_PER_PARTICIPANT)

    # Step 2: Generate ICT pairs with RUNTIME MASKING SUPPORT
    print(f"\n📧 Step 2: Generating ICT pairs with runtime masking support...")
    ict_pairs = aligner.generate_ict_pairs(
        min_query_length=2,
        max_query_length=50,
        query_length_ratio=0.30,
        min_sentence_length=6,
        use_ratio_based_queries=True,
        max_pairs_per_sentence=2,  # 2 pairs per sentence
        max_total_pairs=MAX_ICT_PAIRS,
        random_seed=RANDOM_SEED
    )

    # Step 3: Save with metadata
    if ict_pairs:
        print(f"\n💾 Step 3: Saving ICT pairs with runtime masking support...")
        aligner.save_ict_pairs_with_metadata(ict_pairs, output_file)

        # Final statistics - RUNTIME MASKING FORMAT
        print(f"\n📊 FINAL RESULTS:")
        print("=" * 40)
        participants = set(pair.participant_id for pair in ict_pairs)
        sentences = set(pair.sentence_id for pair in ict_pairs)

        print(f"👥 Participants: {len(participants)}")
        print(f"📝 Unique sentences: {len(sentences)}")
        print(f"🎭 All pairs UNMASKED (ready for runtime masking)")
        print(f"🧠 Total ICT pairs: {len(ict_pairs)}")
        print(f"✅ Saved to: {output_file}")

        print(f"\n🎉 SUCCESS - RUNTIME MASKING SUPPORT COMPLETE!")
        print(f"✅ Generated {len(ict_pairs)} ICT pairs ready for runtime masking")
        print(f"✅ Compatible with masking levels: 0%, 25%, 50%, 75%, 90%, 100%")
        print(f"✅ Full reproducibility with seed: {RANDOM_SEED}")
        print(f"🚀 Ready for multi-masking validation during training!")

    else:
        print("⚠ No ICT pairs generated. Check data paths and parameters.")