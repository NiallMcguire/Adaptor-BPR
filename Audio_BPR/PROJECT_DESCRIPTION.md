# Optimizing Subject-Specific Low-Rank Adapter Placement for Cross-Sensory Brain Passage Retrieval

## Objective

Investigate whether subject-specific low-rank adaptation improves Brain Passage Retrieval (BPR) relative to a subject-agnostic baseline when both models are trained jointly on auditory and visual EEG. This project combines the cross-sensory training approach of *Cross-Sensory Brain Passage Retrieval: Scaling Beyond Visual to Audio* with the shared-plus-subject-specific weight decomposition proposed in *Mitigating Subject Dependency in EEG Decoding with Subject-Specific Low-Rank Adapters*.

The primary research question is: **Does subject-conditioned low-rank adaptation improve retrieval over a jointly trained auditory + visual BPR baseline for subjects represented in training?**

The secondary research question is: **Which adapter placement, rank, and scaling combination provides the best validation retrieval performance, and how do the selected placements compare in test performance and computational cost?**

## Current Protocol

| Component | Specification |
|---|---|
| Training data | Alice auditory EEG and Nieuwland visual EEG, jointly trained |
| Subjects | Seen subjects only; held-out examples within each dataset/participant |
| Comparison | Independently trained baseline plus three nested EEG adapter placements |
| Search | 3 placements x 4 ranks x 4 scaling ratios = 48 adapter configurations |
| Training seed | Exactly one matched seed, default 1 |
| Total fits | 49, including the shared baseline comparison |
| Selection | Combined validation MRR; best configuration per placement and overall winner |
| Test models | Baseline plus one validation-selected model per placement |
| Candidate index | One combined, deduplicated Alice + Nieuwland test-passage index |
| Metrics | MRR and Hit@1/5/10/20, with parameter counts and execution times |

## Data And Task

Use Alice auditory EEG and Nieuwland visual EEG, paired with passage text through the existing Inverse Cloze Task (ICT) pipeline. EEG query spans are encoded directly into a text-passage embedding space; no intermediate EEG-to-text generation is performed. Both datasets contribute to every training run. Input EEG is padded to shared dimensions using the existing preprocessing and normalization pipeline.

Training, validation, and test examples are partitioned within each dataset/participant identity, approximately 80/10/10. Every evaluation subject must be represented in training. Participant identities are namespaced by dataset to prevent accidental adapter sharing. A fixed split seed separates data partitioning from training randomness. Subject holdout and leave-one-subject-out evaluation are explicitly excluded.

## Methods

Compare a subject-agnostic dual encoder against three nested subject-specific low-rank placements. The baseline is trained independently, not created by disabling adapters in a trained SuLoRA model. Use the repository's one-layer transformer with CLS aggregation for the full comparison:

1. Input only: adapt the EEG input projection.
2. EEG backbone: additionally adapt the output projection, attention Q/K/V/output projections, and both feed-forward projections in the main EEG transformer.
3. Full EEG path: additionally adapt attention Q/K/V/output and feed-forward projections in the CLS aggregation transformer.

The default architecture has 1, 8, and 14 adapted projections respectively. Broader placements explicitly route identity through attention and feed-forward computation rather than relying on native transformer fast paths. Mean, max, and multi-vector pooling can be used for separate input/backbone experiments, but do not have the CLS module that distinguishes the full EEG placement.

For a subject s, the adapted projection computes:

`output = shared_linear(input) + (alpha / rank) * input @ A_s @ B_s`.

All shared EEG weights are trained jointly with the adapters. Known subject identity deterministically selects the adapter at training and inference. Following SuLoRA, both low-rank factors have nonzero Gaussian initialization: A has standard deviation sqrt(2/r), and B has standard deviation 0.01. The full EEG placement applies the paper's broad linear-layer adaptation principle to BPR's EEG branch, without adapting the frozen text branch or claiming an exact reproduction of its architecture.

Keep the pretrained BERT text encoder frozen, use an identity text projection, and train EEG representations with the existing normalized similarity/InfoNCE objective and in-batch negatives. The text encoder remains in evaluation mode during training. Default optimization uses AdamW, learning rate 1e-5, weight decay 0.01, temperature 0.07, batch size 32, gradient clipping at 1.0, 10-epoch warmup, linear decay, up to 200 epochs, and patience 10. Use exactly one matched training seed (default 1), identical initial shared weights, the same splits, and the same training settings across variants.

## Hyperparameter Search

Search all three placements, adapter ranks {4, 8, 16, 32}, and scaling ratios alpha/r {0.25, 1, 2, 3}. Rank controls subject-specific capacity and parameter cost; scaling controls the strength of each correction relative to shared weights. Other model settings remain fixed. This entails 48 adapter configurations plus one independently trained baseline, 49 fits total using one seed.

For each configuration, retain the checkpoint with the highest MRR on the full combined validation index at 90% masking probability. Select the best rank/scaling pair separately within each placement, and select the overall preferred placement using validation MRR. Resolve ties by the declared grid order. Do not inspect test performance during configuration selection. Apply the same checkpoint-selection criterion to the independently trained baseline.

## Evaluation

Evaluate the baseline and the three placement-specific validation-selected SuLoRA checkpoints against one shared, deduplicated index of Alice and Nieuwland test passages. Every auditory and visual test query competes against all indexed passages from both datasets, with its paired ICT passage as the relevant document. Deduplicate identical passage text consistently across all four models. Testing all three selected versions supports the architectural comparison; the declared preferred version remains the validation-selected winner.

Report overall MRR and Hit@1, Hit@5, Hit@10, and Hit@20, with optional additional cutoffs. Evaluate at masking probabilities {0, 25, 50, 75, 90, 100}%, report each condition separately, and include a mean across conditions. These levels control the probability of removing the ICT query span; they are not percentages of independently removed words. Report each placement's matched-seed SuLoRA-minus-baseline differences, parameter costs, adapted layer names, fit time (training plus validation), and test time. Any across-seed standard deviation is zero with one seed and must not be interpreted as estimated uncertainty. Supplementary auditory/visual breakdowns retain the same shared candidate index.

Record candidate-index fingerprints, query-level ranks, sample/subject counts, parameter counts, splits, routing, settings, checkpoints, and append-only timestamped experiment events. Improvements must be established from the real dataset results; implementation smoke tests do not constitute evidence of a retrieval improvement.

## Execution Status

The three-placement search supports both sequential execution and an independently schedulable Slurm array with a dependent finalizer. All 15 pipeline tests passed using synthetic EEG pairs and tiny local models. Real-data array `8205216` and finalizer `8205217` were cancelled on 2026-10-20 at the user's request. Of 49 fits, 29 completed and were inventoried: baseline plus all 16 input-only and 12 EEG-backbone configurations. Two EEG-backbone fits were interrupted after saving best checkpoints (rank 32, alpha/r 0.25: validation MRR 0.1009 at epoch 59; rank 32, alpha/r 1: validation MRR 0.0788 at epoch 47). Eighteen configurations were still queued and never started. No full-EGEE fit, combined test evaluation, or final comparison ran. The checkpoint inventory is in `/users/gxb18167/subject_adapter_placements_20261015/individual_runs/partial_run_inventory_20261020_112714.json`; validation scores are exploratory and are not test results.

## Interpretation And Limitations

This experiment measures personalization of known subjects on held-out examples. It does not establish zero-shot transfer to new subjects, retrieval from genuine information-seeking EEG, or a causal benefit of sensory diversity independent of dataset differences. Pair-level splits do not guarantee unseen passage content: repeated stimuli, overlapping ICT spans, and related trials may occur across splits. Stronger content/session-disjoint controls would be separate follow-up experiments. Passage pool sizes can change with masking and are recorded; comparisons are paired within a masking condition. One seed does not characterize training variability or establish statistical significance. Broader placements add parameters, so retrieval gains must be considered alongside their capacity and computational cost.