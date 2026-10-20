# Experiment Log

## Current Status

**Active protocol:** Joint Alice + Nieuwland training; seen subjects only; three EEG adapter placements; exactly one matched seed. Search 48 adapter configurations and train one independent baseline, 49 fits total. Choose configurations using combined validation MRR and evaluate the baseline plus three placement-specific winners on one combined test index.

**Research status:** Implementation verified by 15 passing pipeline tests, including independent workers, finalizer completeness, and mocked Slurm dependencies. Original job `8205201` was cancelled by the user after 6 minutes 48 seconds. Both datasets loaded; baseline epoch 1 validation completed with MRR 0.0160, but no adapter fits or test evaluation ran. That partial value is not comparative evidence; its checkpoint remains in the old run directory.

**Run 8205201 (cancelled):** `/users/gxb18167/subject_adapter_placements_20261015/experiment_matrix.json`. Inputs were Alice and Nieuwland runtime-masking ICT files. Requested 49 sequential fits with one GPU, 8 CPUs, and 128 GB RAM. The initial 7-day limit exceeded GPU QOS (maximum 2 days), so the pending job's limit was reduced to `2-00:01:00`. It started on `gpu06` and was cancelled by the user after 6 minutes 48 seconds. It loaded the 16,266 pairs and completed baseline epoch 1 validation (MRR 0.0160), then was cancelled before baseline training completed; no adapter run or test evaluation started. Partial checkpoint: `/users/gxb18167/subject_adapter_placements_20261015/results/subject_adapter_search/baseline/seed_1/best_model.pt`. Do not use the partial metric/checkpoint as a final result.

**Replacement array (cancelled 2026-10-20):** `/users/gxb18167/subject_adapter_placements_20261015/individual_runs/experiment_matrix.json`. Array job `8205216` and finalizer `8205217` were cancelled by the user. Of 49 planned fits, 29 completed (baseline, all 16 input-only, and 12 EEG-backbone), two EEG-backbone runs were interrupted mid-fit, and 18 queued tasks never started. No test evaluation or final comparison ran. The completed-fit and Slurm accounting inventory is `/users/gxb18167/subject_adapter_placements_20261015/individual_runs/partial_run_inventory_20261020_112714.json`.

**Progress checkpoint, 2026-10-19:** 22/49 fits have completed: the baseline and all 16 input-only configs, plus five EEG-backbone configs. Baseline best validation MRR is 0.113600 (epoch 99). Best completed input-only is `sulora_input_only_r32_ratio1p0` (0.111405). Best completed EEG-backbone so far is `sulora_eeg_backbone_r8_ratio0p25` (0.105729). Task 22 (`eeg_backbone`, rank 8, alpha/r 1) was at epoch 108 with best MRR about 0.0991; task 23 (rank 8, alpha/r 2) was at epoch 24 with best MRR about 0.0211. Twenty-five tasks remained queued; no `full_eeg` config or test evaluation had started. These are validation-only progress values, not final comparisons.

**Progress checkpoint, 2026-10-20:** 29/49 fits have completed. All 16 `input_only` configs and 12 `eeg_backbone` configs are complete; baseline is complete. Best completed validation MRR remains baseline 0.113600, `input_only` rank 32 alpha/r 1 at 0.111405, and `eeg_backbone` rank 8 alpha/r 1 at 0.106264. Array tasks 29 and 30 are running `eeg_backbone` rank 32 with alpha/r 0.25 and 1; latest logs show epochs 58 and 47 and best MRR 0.0983 and 0.0788 respectively. About 43 GB RSS per active task was observed. Eighteen configurations (indices 31-48) remain queued. No `full_eeg` runs, test evaluation, or final comparison yet; finalizer remains dependency-pending. Active stderr contains only PyTorch's nested-tensor prototype warning.

**Cancellation snapshot, 2026-10-20:** After that progress check, the user requested cancellation. Slurm confirms 29 configurations completed, array tasks 29 and 30 cancelled while running, tasks 31-48 cancelled before start, and finalizer 8205217 cancelled. The interrupted rank-32 backbone fits retained best checkpoints and validation observations: alpha/r 0.25 at epoch 59 (MRR 0.100885), alpha/r 1 at epoch 47 (MRR 0.078756); these did not produce completed trial result files and are excluded from the 29 completed-fit count. The final inventory reconciles actual worker `trial_index` values, checkpoint presence, completed results, and Slurm accounting. No test metrics/comparison exist; no `full_eeg` fit started.

**Progress snapshot, 2026-10-15:** Task 0 (baseline) reached epoch 21 with best validation MRR 0.0744; task 1 (`input_only`, rank 4, alpha/r 0.25) reached epoch 20 with best validation MRR 0.0695. These are interim, non-comparable checkpoint-selection values; neither fit completed, so there are no completed search results or test metrics. Task stderr contains only the PyTorch nested-tensor prototype warning. Remaining tasks are still throttled behind the two active configurations.

**Documentation update, 2026-10-12:** Updated the project title and objectives to include placement optimization, added a current-protocol summary and execution status, and explicitly marked the initial plan below as historical. The authoritative current project description is [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md).

## 2026-10-12: Three Placements And One Seed

**Current protocol (supersedes the initial input-only/three-seed plan below):** Search `input_only`, `eeg_backbone`, and `full_eeg`, with one matched seed (default 1). Each placement searches rank {4, 8, 16, 32} and alpha/r {0.25, 1, 2, 3}: 48 adapter fits plus one baseline, 49 fits total.

**Placement definitions:** Input only adapts the input projection. EEG backbone additionally adapts the output projection, attention Q/K/V/output, and both feed-forward projections in the main EEG transformer. Full EEG additionally adapts the attention and feed-forward projections in CLS aggregation. Default architecture: 1/8/14 adapted projections. BERT remains frozen. Full comparison requires transformer EEG encoding and CLS pooling.

**Selection/evaluation:** Choose the best rank/scaling pair within each placement and the overall preferred placement using combined validation MRR, never test scores. Test the baseline and three selected versions on the same combined index. Report MRR, Hit@1/5/10/20, per-placement differences, parameter counts, adapted layer names, fit times, and test times. One seed does not provide a run-to-run uncertainty estimate.

**Implementation:** Subject-aware attention uses explicit Q/K/V/output projections and PyTorch scaled dot-product attention. Transformer residual, normalization, activation, masking, and dropout behavior are preserved; native fast paths cannot skip subject adapter updates. Search manifests enumerate every configuration before fitting. The scheduler's `--subject-adapter-search` mode prepares one sequential full-grid script; legacy `--use-lora` remains text LoRA only.

**Verification:** All 13 pipeline tests passed. Tests cover zero-correction equivalence against the original transformer on valid outputs with padding, mixed-subject routing, gradients through every adapted projection, increasing parameter costs, all 49 default configurations with one seed, validation-only per-placement selection, checkpoint reload and combined testing for all four selected models, and prepare-only scheduling without submission. Tests use synthetic data and tiny local models, not real performance measurements. Runtime imports succeeded; VS Code currently reports unresolved existing Torch/Transformers/PEFT imports in the model file, so editor environment resolution remains separate from the passing runtime checks.

**Submission status:** No jobs submitted. Only temporary test scripts and tiny CPU smoke fits were created. No real-data experiment was launched, and no Git commit or push was performed for this update.

## 2026-10-12: Initial Protocol (Superseded)

This section preserves the original input-only, three-seed plan and its verification history. Its seed counts and fit totals are not the current execution settings; use the current-status section above.

**Research question:** Does jointly trained subject-specific low-rank EEG adaptation improve cross-sensory BPR over a matched subject-agnostic model?

**Data:** Alice auditory EEG + Nieuwland visual EEG. Both sources are used for every training run. Seen subjects only, with per-dataset/per-subject pair splits and a fixed split seed.

**Comparison:** Independently trained combined-data baseline versus a combined-data SuLoRA-style model. Adapter placement is fixed at the EEG input projection; shared EEG weights and subject-specific factors are jointly trained. Text BERT weights are frozen; text LoRA is disabled.

**Search:** Rank {4, 8, 16, 32}; alpha/r {0.25, 1, 2, 3}; training seeds {1, 2, 3}. Best checkpoints and configuration selection use combined validation MRR at 90% masking. No test metrics are used for search selection.

**Evaluation:** All test queries search one combined, deduplicated Alice + Nieuwland passage index. Report MRR and Hit@1/5/10/20, overall and by source using the same index, at masking probabilities 0/25/50/75/90/100%. Record matched-seed means, standard deviations, differences, query ranks, and index fingerprints.

**Implemented artifacts:** Dedicated experiment runner, subject-routed low-rank projection, optional seen-subject split policy, full combined-index evaluator, regression/smoke coverage, usage instructions, and a project-log description. Existing controller experiments remain available separately.

**Verification so far:** Mixed-subject formula/scaling and joint-gradient checks passed. Simple and transformer input-projection integration passed. Four focused subject-experiment tests passed, including a complete CPU experiment with synthetic data and tiny locally initialized BERT models, all pooling modes, checkpoint reload, validation-only selection, and runtime logs. Synthetic test metrics are not research results.

**Final regression verification:** All 10 tests passed using `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s Audio_BPR/tests -p 'test_*.py' -v`. Python syntax/CLI checks and editor diagnostics passed for the changed implementation files. This also covers the existing seven-mixture pipeline, task reader, and scheduler preparation without job submission.

**Corrections during verification:** Original pair indices now refer to combined-data rows so split disjointness is auditable. Hit@k remains valid when k exceeds the passage-index size. Cross-source duplicate passage text is indexed once rather than counted as separate candidates.

**Real experiment status at initial implementation:** Not run. Real ICT file paths were not supplied, and no improvement claim had been made. The initial grid entailed 51 training fits; this has been superseded by the current three-placement, one-seed grid of 49 fits.

## Runtime Record

Each execution writes its own append-only `experiment_log.jsonl` in a new output directory. Events include experiment/data initialization, trial starts, every epoch's training loss and validation metrics, completed trials, adapter selection, final test evaluations, completion, and failures. Configuration, subject routing, split manifests, checkpoints, search results, and final comparisons are saved alongside it.

Do not overwrite previous output directories. Keep this document for protocol changes, run locations, and scientific interpretation; use the generated JSON/JSONL artifacts as the detailed record of each execution.