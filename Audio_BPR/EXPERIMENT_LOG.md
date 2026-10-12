# Experiment Log

## 2026-10-12: Protocol And Implementation

**Research question:** Does jointly trained subject-specific low-rank EEG adaptation improve cross-sensory BPR over a matched subject-agnostic model?

**Data:** Alice auditory EEG + Nieuwland visual EEG. Both sources are used for every training run. Seen subjects only, with per-dataset/per-subject pair splits and a fixed split seed.

**Comparison:** Independently trained combined-data baseline versus a combined-data SuLoRA-style model. Adapter placement is fixed at the EEG input projection; shared EEG weights and subject-specific factors are jointly trained. Text BERT weights are frozen; text LoRA is disabled.

**Search:** Rank {4, 8, 16, 32}; alpha/r {0.25, 1, 2, 3}; training seeds {1, 2, 3}. Best checkpoints and configuration selection use combined validation MRR at 90% masking. No test metrics are used for search selection.

**Evaluation:** All test queries search one combined, deduplicated Alice + Nieuwland passage index. Report MRR and Hit@1/5/10/20, overall and by source using the same index, at masking probabilities 0/25/50/75/90/100%. Record matched-seed means, standard deviations, differences, query ranks, and index fingerprints.

**Implemented artifacts:** Dedicated experiment runner, subject-routed low-rank projection, optional seen-subject split policy, full combined-index evaluator, regression/smoke coverage, usage instructions, and a project-log description. Existing controller experiments remain available separately.

**Verification so far:** Mixed-subject formula/scaling and joint-gradient checks passed. Simple and transformer input-projection integration passed. Four focused subject-experiment tests passed, including a complete CPU experiment with synthetic data and tiny locally initialized BERT models, all pooling modes, checkpoint reload, validation-only selection, and runtime logs. Synthetic test metrics are not research results.

**Final regression verification:** All 10 tests passed using `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s Audio_BPR/tests -p 'test_*.py' -v`. Python syntax/CLI checks and editor diagnostics passed for the changed implementation files. This also covers the existing seven-mixture pipeline, task reader, and scheduler preparation without job submission.

**Corrections during verification:** Original pair indices now refer to combined-data rows so split disjointness is auditable. Hit@k remains valid when k exceeds the passage-index size. Cross-source duplicate passage text is indexed once rather than counted as separate candidates.

**Real experiment status:** Not run. Real ICT file paths were not supplied, and no improvement claim has been made. Supply the generated Alice/Nieuwland files to the documented runner. The default grid entails 51 training fits and stores full checkpoints.

## Runtime Record

Each execution writes its own append-only `experiment_log.jsonl` in a new output directory. Events include experiment/data initialization, trial starts, every epoch's training loss and validation metrics, completed trials, adapter selection, final test evaluations, completion, and failures. Configuration, subject routing, split manifests, checkpoints, search results, and final comparisons are saved alongside it.

Do not overwrite previous output directories. Keep this document for protocol changes, run locations, and scientific interpretation; use the generated JSON/JSONL artifacts as the detailed record of each execution.