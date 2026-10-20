# EEG-Text Brain Passage Retrieval

Scripts for generating Inverse Cloze Task (ICT) pairs from EEG-text datasets and training dual-encoder Brain Passage Retrieval models across auditory and visual modalities.

## Seen-Subject Adapter Experiment

Use `subject_adapter_experiment.py` for the matched combined-data baseline/SuLoRA experiment. The older `controller.py` workflow remains available, but its source-specific test pools and text-LoRA option are not the protocol for this experiment.

From the repository root, after generating the Alice and Nieuwland ICT `.npy` files:

```bash
python Audio_BPR/subject_adapter_experiment.py \
  --audio_path /path/to/alice_ict_pairs.npy \
  --visual_path /path/to/nieuwland_ict_pairs.npy \
  --output_dir results/combined_seen_subjects \
  --seed 1 \
  --split_seed 42 \
  --ranks 4 8 16 32 \
  --alpha_ratios 0.25 1 2 3 \
  --placements input_only eeg_backbone full_eeg
```

Defaults: jointly train on both datasets, transformer EEG encoder, CLS pooling, frozen BERT-base-uncased text encoder and identity text projection, AdamW (`lr=1e-5`, weight decay `0.01`), InfoNCE temperature `0.07`, batch size 32, 200 maximum epochs, 10 warmup epochs followed by linear decay, patience 10. No text-encoder LoRA is enabled. Pretrained model/tokenizer files must be available locally or downloadable from Hugging Face; local JSON logs do not require a W&B account.

Every subject is split independently into training, validation, and test pairs (approximately 80/10/10). Dataset + participant identities are unique across sources. Subjects need at least three usable pairs; processing failures that leave an evaluation subject absent from training cause the run to fail. `--split_seed` fixes the splits independently of training seeds. This runner does not expose an unseen-subject/LOSO option.

Shared EEG weights and subject-specific low-rank factors are trained jointly. The search covers three nested placements:

| Placement | Adapted EEG projections |
|---|---|
| `input_only` | Input projection only |
| `eeg_backbone` | Input/output projections, main transformer attention Q/K/V/output, and both feed-forward projections |
| `full_eeg` | Everything in `eeg_backbone`, plus CLS aggregation attention Q/K/V/output and both feed-forward projections |

For the default one-layer backbone and one-layer CLS aggregator, these adapt 1, 8, and 14 linear projections. BERT is never subject-adapted. Broader placements use an explicit attention/transformer forward path so PyTorch fast paths cannot bypass adapters. The full three-placement comparison requires transformer EEG encoding and CLS pooling. To use other pooling methods, explicitly request `--placements input_only eeg_backbone`; non-transformer EEG architectures support `input_only` only.

`--alpha_ratios` specifies alpha/r, so rank 8 and ratio 3 mean alpha 24. Exactly one matched seed is used (default 1; set `--seed` to change it). The default grid has 3 placements x 4 ranks x 4 ratios = 48 adapter fits plus one independently trained baseline, 49 fits total. Use a smaller grid for a pilot:

```bash
python Audio_BPR/subject_adapter_experiment.py \
  --audio_path /path/to/alice_ict_pairs.npy \
  --visual_path /path/to/nieuwland_ict_pairs.npy \
  --output_dir results/combined_pilot \
  --ranks 4 8 --alpha_ratios 1 3 --seed 1 \
  --epochs 5 --warmup_epochs 1
```

Each trial retains its best combined validation-MRR checkpoint, using 90% masking by default (`--primary_masking_level`). The best rank/scaling configuration is selected separately within each placement, and the overall preferred placement is selected by validation MRR as well. Only then are the baseline and three selected adapter versions tested. Test queries search the entire deduplicated combined test-passage index, not sampled negatives or separate source pools. Source breakdowns use that same combined index.

Test outputs include MRR, Hit@1, Hit@5, Hit@10, and Hit@20 at masking levels 0/25/50/75/90/100, plus mean across levels and per-placement differences against the matched baseline. Parameter counts, adapted layer names, fit times (training plus validation), and test times are recorded. A single seed gives zero across-seed standard deviation; this is not an uncertainty estimate. Use `--hit_k 1 3 5 10 20` to request other hit cutoffs. Masking levels are probabilities of removing the whole ICT query span, not percentages of individual passage tokens removed. Identical passage text is indexed once and shared as a positive by its corresponding queries.

Each new output directory contains:

| Artifact | Purpose |
|---|---|
| `experiment_config.json` | Complete command settings |
| `experiment_matrix.json` | Complete placement/rank/scaling grid and expected number of fits |
| `split_manifest.json` | Original combined-data row identities in each split |
| `subject_mapping.json` | Persisted dataset/participant-to-adapter routing |
| `experiment_log.jsonl` | Append-only lifecycle, epoch, selection, test, and failure events |
| `search_results.json` | Best validation MRR, parameter counts, and fit time for every configuration |
| `selected_adapter.json` | Best configuration per placement, overall validation-selected winner, and selection rule |
| `<trial>/seed_<seed>/best_model.pt` | Best checkpoint, model settings, routing, dimensions, and validation ranks |
| `<trial>/seed_<seed>/test_metrics.json` | Overall/source metrics, query ranks, and combined-index fingerprints for tested models |
| `comparison.json` | Baseline and each selected placement's metrics, paired differences, and resource costs |

Existing output directories must be empty; previous results are never overwritten. Failed runs retain their logs and completed artifacts, but automatic resume is not implemented. Full checkpoints include frozen text weights, so budget disk space for all 49 default fits. Pooling and EEG architecture can be changed with `--pooling_strategy` and `--eeg_arch`, subject to the placement restrictions above, but must remain fixed within a comparison. Earlier input-only checkpoints have a different schema from the new broader placements; do not mix old results into a new search.

### Sequential Slurm Search

The scheduler's explicit subject-adapter mode generates one job script that runs the complete search sequentially. Preparation never calls `sbatch`. The legacy `--use-lora` flag controls text LoRA and is not the subject-adapter search.

```bash
python Audio_BPR/experiment_scheduler.py \
  --subject-adapter-search \
  --audio-data /path/to/alice_ict_pairs.npy \
  --visual-data /path/to/nieuwland_ict_pairs.npy \
  --output-dir prepared_experiments_subject_adapters \
  --seed 1 \
  --placements input_only eeg_backbone full_eeg \
  --ranks 4 8 16 32 --alpha-ratios 0.25 1 2 3
```

This legacy mode writes one sequential `jobs/subject_adapter_search.sbatch` script and never submits it. The prepared script uses equal-probability source sampling; add `--balance_sources` to direct runner commands to use the same policy.

### Independent Slurm Array

For cluster sweeps, submit one independently retryable task per configuration and a dependent finalizer. Each task writes its checkpoint and validation result under its own trial directory. The finalizer only runs after every array task succeeds, selects the best configuration in each placement from validation MRR, then evaluates the baseline and three selected models.

```bash
python Audio_BPR/experiment_scheduler.py \
  --subject-adapter-array \
  --audio-data /path/to/alice_ict_pairs.npy \
  --visual-data /path/to/nieuwland_ict_pairs.npy \
  --output-dir prepared_experiments_subject_adapter_array \
  --time 2-00:01:00 --memory 128G --cpus 8 --gpu-request gpu:1 \
  --array-concurrency 2 \
  --seed 1 --placements input_only eeg_backbone full_eeg \
  --ranks 4 8 16 32 --alpha-ratios 0.25 1 2 3
```

This mode submits the 49-task array and its `afterok` finalizer. Set `--array-concurrency` to fit the cluster/user GPU and CPU limits. A task interrupted before completing writes no `validation_result.json`; after fixing the cause, requeue that array index. Completed trials are reused on retries. Each array task still has the requested per-job walltime cap.

Project-log wording and limitations are in [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md); implementation/experiment status is in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Requirements

Python 3.7 or higher

Install dependencies:
```bash
pip install -r requirements.txt
```


## Datasets

### Alice AudioBook Dataset (Auditory EEG)

Download: https://openneuro.org/datasets/ds002322


### Nieuwland Dataset (Visual EEG)

Download: https://osf.io/eyzaq/


## Reproducing Paper Experiments

### Step 1: Generate ICT Pairs

Generate ICT pairs for both datasets using the paper's configuration.

**Alice Dataset (Auditory):**
```bash
python alice_ict_reader.py \
  --text_path /path/to/AliceChapterOne-EEG.csv \
  --eeg_path /path/to/Subjects \
  --output alice_ict_pairs.npy \
  --seed 42 \
  --min_query_length 2 \
  --max_query_length 50 \
  --query_ratio 0.3 \
  --pairs_per_sentence 2
```

**Nieuwland Dataset (Visual):**
```bash
python nieuwland_ict_reader.py \
  --data_dir /path/to/Processed_segmented_data \
  --sentence_dir /path/to/Sentence_Materials \
  --output nieuwland_ict_pairs.npy \
  --seed 42 \
  --min_query_length 2 \
  --max_query_length 50 \
  --query_ratio 0.3 \
  --pairs_per_sentence 2
```

### Step 2: Train Models

The paper evaluates three training regimes (individual Alice, individual Nieuwland, combined) across four pooling strategies (CLS, MEAN, MAX, MULTI).

#### Individual Training - Alice (Auditory)

**CLS Pooling:**
```bash
python controller.py \
  --data_path alice_ict_pairs.npy \
  --pooling_strategy cls \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/alice_cls
```

**MEAN Pooling:**
```bash
python controller.py \
  --data_path alice_ict_pairs.npy \
  --pooling_strategy mean \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/alice_mean
```

**MAX Pooling:**
```bash
python controller.py \
  --data_path alice_ict_pairs.npy \
  --pooling_strategy max \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/alice_max
```

**MULTI Pooling:**
```bash
python controller.py \
  --data_path alice_ict_pairs.npy \
  --pooling_strategy multi \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/alice_multi
```

#### Individual Training - Nieuwland (Visual)

**CLS Pooling:**
```bash
python controller.py \
  --data_path nieuwland_ict_pairs.npy \
  --pooling_strategy cls \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/nieuwland_cls
```

**MEAN Pooling:**
```bash
python controller.py \
  --data_path nieuwland_ict_pairs.npy \
  --pooling_strategy mean \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/nieuwland_mean
```

**MAX Pooling:**
```bash
python controller.py \
  --data_path nieuwland_ict_pairs.npy \
  --pooling_strategy max \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/nieuwland_max
```

**MULTI Pooling:**
```bash
python controller.py \
  --data_path nieuwland_ict_pairs.npy \
  --pooling_strategy multi \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/nieuwland_multi
```

#### Combined Cross-Sensory Training

**CLS Pooling:**
```bash
python controller.py \
  --data_paths alice_ict_pairs.npy nieuwland_ict_pairs.npy \
  --pooling_strategy cls \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/combined_cls
```

**MEAN Pooling:**
```bash
python controller.py \
  --data_paths alice_ict_pairs.npy nieuwland_ict_pairs.npy \
  --pooling_strategy mean \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/combined_mean
```

**MAX Pooling:**
```bash
python controller.py \
  --data_paths alice_ict_pairs.npy nieuwland_ict_pairs.npy \
  --pooling_strategy max \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/combined_max
```

**MULTI Pooling:**
```bash
python controller.py \
  --data_paths alice_ict_pairs.npy nieuwland_ict_pairs.npy \
  --pooling_strategy multi \
  --encoder_type dual \
  --query_type eeg \
  --batch_size 32 \
  --lr 1e-5 \
  --epochs 200 \
  --patience 10 \
  --seed 42 \
  --holdout_subjects \
  --training_masking_level 90 \
  --enable_multi_masking_validation \
  --validation_masking_levels 0 25 50 75 90 100 \
  --output_dir results/combined_multi
```

### Step 3: Evaluation

Models are automatically evaluated during training across all masking levels (0%, 25%, 50%, 75%, 90%, 100%). Results are saved in the specified `--output_dir` including:

- `model_<pooling>_<arch>.pt`: Trained model checkpoint
- `experiment_config.json`: Full experiment configuration
- Console output with validation and test metrics (MRR, Hit@1, Hit@5, Hit@10)


## Command-Line Options

### ICT Generation Parameters

**Common (both scripts):**
- `--seed`: Random seed for reproducibility (default: 42)
- `--min_query_length`: Minimum query length in words (default: 2)
- `--max_query_length`: Maximum query length in words (default: 50)
- `--query_ratio`: Query length as fraction of sentence (default: 0.3)
- `--min_sentence_length`: Minimum sentence length to process (default: 6)
- `--pairs_per_sentence`: ICT pairs generated per sentence (default: 2)
- `--max_pairs`: Maximum total pairs to generate (default: unlimited)

**Alice-specific:**
- `--limit_subjects`: Limit number of subjects (default: all)
- `--target_freq`: Target sampling frequency in Hz (default: 128)
- `--target_channels`: Target number of channels (default: 128)
- `--no_preprocess`: Disable EEG preprocessing

**Nieuwland-specific:**
- `--limit_participants`: Limit number of participants (default: all)
- `--max_sentences_per_participant`: Maximum sentences per participant (default: all)

### Model Training Parameters

- `--data_path`: Path to single ICT pairs .npy file
- `--data_paths`: Paths to multiple ICT pairs .npy files (for combined training)
- `--pooling_strategy`: Semantic aggregation (choices: cls, mean, max, multi)
- `--encoder_type`: Encoder architecture (choices: dual, cross; default: dual)
- `--query_type`: Query type (choices: eeg, text; default: eeg)
- `--batch_size`: Training batch size (default: 32)
- `--lr`: Learning rate (default: 1e-5)
- `--epochs`: Maximum training epochs (default: 200)
- `--patience`: Early stopping patience (default: 10)
- `--holdout_subjects`: Use subject-based train/val/test split
- `--training_masking_level`: Masking probability during training (default: 90)
- `--enable_multi_masking_validation`: Enable multi-level masking validation
- `--validation_masking_levels`: Masking levels to evaluate (default: 0 25 50 75 90 100)
- `--seed`: Random seed (default: 42)
- `--output_dir`: Output directory for results

View all options:
```bash
python alice_ict_reader.py --help
python nieuwland_ict_reader.py --help
python controller.py --help
```

## Output Format

### ICT Pairs

Each ICT generation script produces:
- `.npy` file: ICT pairs with EEG data
- `.json` file: Metadata

Load the data:
```python
import numpy as np
data = np.load("alice_ict_pairs.npy", allow_pickle=True).item()
ict_pairs = data['ict_pairs']
metadata = data['metadata']

# Each ICT pair contains:
# - query_text, query_eeg, query_words
# - doc_text, doc_eeg, doc_words (unmasked full sentence)
# - query_start_idx, query_end_idx (for runtime masking)
# - participant_id, sentence_id, fs
```

### Model Outputs

Training produces:
- `model_<pooling>_<arch>.pt`: Trained model checkpoint
- `experiment_config.json`: Full configuration
- Console output with validation/test metrics (MRR, Hit@1, Hit@5, Hit@10)

## Data Attribution

- **Alice AudioBook**: Brennan et al. (2019) - https://openneuro.org/datasets/ds002322
- **Nieuwland**: Nieuwland et al. (2018) - https://osf.io/eyzaq/

Please cite the original dataset papers when using these resources.

## Authentic Query Mixed Experiments

The extracted task data can be converted into EEG-question-to-answer pairs. Query EEG is built from fixation epochs whose word index is confidently matched to the displayed `question_text`; the target passage is `ground_truth_response`. These task pairs are marked as non-ICT, so their complete answer text is never runtime-masked.

```bash
python Dataset/authentic_query_reader.py \
  --dataset-dir ../dataset \
  --output ../dataset/authentic_query_task_pairs.npy \
  --unaligned-clock-offset-seconds 0
```

P001 and P002 are included using the supplied offset as an **unverified assumption** because their PsychoPy clock is not synchronized to EEG. Their byte-identical source files also do not establish independent participant identity. The generated manifest records this, as well as trials skipped for missing answer rows or question-word EEG.

Prepare seven Slurm job files for the comparison matrix:

1. Task only
2. Audio only
3. Visual only
4. Audio + visual
5. Audio + task
6. Visual + task
7. Audio + visual + task

```bash
python experiment_scheduler.py \
  --audio-data /path/to/alice_ict_pairs.npy \
  --visual-data /path/to/nieuwland_ict_pairs.npy \
  --task-data ../dataset/authentic_query_task_pairs.npy \
  --output-dir prepared_experiments
```

This command only writes the experiment matrix and `.sbatch` files; it does not submit jobs. Pair-level train/validation/test assignments are deterministic and stratified per source, and the same source test pairs are reused across mixtures. Mixed-source training samples sources with equal probability. Test evaluation ranks each source's queries against only that source's held-out candidate passages; per-source metrics are written under each run's `test_by_source/` directory. The task source remains unmasked at every masking level, while Audio and Visual ICT passages use the configured span masking.

