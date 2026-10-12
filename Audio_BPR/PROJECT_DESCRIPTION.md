# Subject-Specific Low-Rank Adaptation for Cross-Sensory Brain Passage Retrieval

## Objective

Investigate whether subject-specific low-rank adaptation improves Brain Passage Retrieval (BPR) relative to a subject-agnostic baseline when both models are trained jointly on auditory and visual EEG. This project combines the cross-sensory training approach of *Cross-Sensory Brain Passage Retrieval: Scaling Beyond Visual to Audio* with the shared-plus-subject-specific weight decomposition proposed in *Mitigating Subject Dependency in EEG Decoding with Subject-Specific Low-Rank Adapters*.

The primary research question is: **Does subject-conditioned low-rank adaptation improve retrieval over a jointly trained auditory + visual BPR baseline for subjects represented in training?**

## Data And Task

Use Alice auditory EEG and Nieuwland visual EEG, paired with passage text through the existing Inverse Cloze Task (ICT) pipeline. EEG query spans are encoded directly into a text-passage embedding space; no intermediate EEG-to-text generation is performed. Both datasets contribute to every training run. Input EEG is padded to shared dimensions using the existing preprocessing and normalization pipeline.

Training, validation, and test examples are partitioned within each dataset/participant identity, approximately 80/10/10. Every evaluation subject must be represented in training. Participant identities are namespaced by dataset to prevent accidental adapter sharing. A fixed split seed separates data partitioning from training randomness. Subject holdout and leave-one-subject-out evaluation are explicitly excluded.

## Methods

Compare a subject-agnostic dual encoder against the same model augmented with a subject-specific low-rank correction at the EEG input projection. The baseline is trained independently, not created by disabling adapters in a trained SuLoRA model. The default EEG encoder is the repository's one-layer transformer with CLS aggregation; mean, max, and multi-vector pooling are also supported as separate controlled experiments.

For a subject s, the adapted projection computes:

`output = shared_linear(input) + (alpha / rank) * input @ A_s @ B_s`.

The shared projection and remaining EEG backbone are trained jointly with the adapters. Known subject identity deterministically selects the adapter at training and inference. Following SuLoRA, both low-rank factors have nonzero Gaussian initialization: A has standard deviation sqrt(2/r), and B has standard deviation 0.01. The fixed input-projection placement is a SuLoRA-style adaptation of BPR, not a reproduction of every layer replacement in the original SuLoRA paper.

Keep the pretrained BERT text encoder frozen, use an identity text projection, and train EEG representations with the existing normalized similarity/InfoNCE objective and in-batch negatives. The text encoder remains in evaluation mode during training. Default optimization uses AdamW, learning rate 1e-5, weight decay 0.01, temperature 0.07, batch size 32, gradient clipping at 1.0, 10-epoch warmup, linear decay, up to 200 epochs, and patience 10. Use matched training seeds {1, 2, 3}, identical initial shared weights, the same splits, and the same training settings across variants.

## Hyperparameter Search

Search adapter ranks {4, 8, 16, 32} and scaling ratios alpha/r {0.25, 1, 2, 3}. Rank controls subject-specific capacity and parameter cost; scaling controls the strength of each correction relative to shared weights. Placement and other model settings remain fixed throughout the search.

For each configuration and seed, retain the checkpoint with the highest MRR on the full combined validation index at 90% masking probability. Select one configuration by its average best validation MRR across seeds. Resolve ties by the declared grid order. Do not inspect test performance during configuration selection. Apply the same checkpoint-selection criterion to the independently trained baseline.

## Evaluation

Evaluate baseline and selected SuLoRA checkpoints against one shared, deduplicated index of Alice and Nieuwland test passages. Every auditory and visual test query competes against all indexed passages from both datasets, with its paired ICT passage as the relevant document. Deduplicate identical passage text consistently across both models.

Report overall MRR and Hit@1, Hit@5, Hit@10, and Hit@20, with optional additional cutoffs. Evaluate at masking probabilities {0, 25, 50, 75, 90, 100}%, report each condition separately, and include a mean across conditions. These levels control the probability of removing the ICT query span; they are not percentages of independently removed words. Summarize mean and population standard deviation across matched seeds and paired SuLoRA-minus-baseline differences. Supplementary auditory/visual breakdowns retain the same shared candidate index.

Record candidate-index fingerprints, query-level ranks, sample/subject counts, parameter counts, splits, routing, settings, checkpoints, and append-only timestamped experiment events. Improvements must be established from the real dataset results; implementation smoke tests do not constitute evidence of a retrieval improvement.

## Interpretation And Limitations

This experiment measures personalization of known subjects on held-out examples. It does not establish zero-shot transfer to new subjects, retrieval from genuine information-seeking EEG, or a causal benefit of sensory diversity independent of dataset differences. Pair-level splits do not guarantee unseen passage content: repeated stimuli, overlapping ICT spans, and related trials may occur across splits. Stronger content/session-disjoint controls would be separate follow-up experiments. Passage pool sizes can change with masking and are recorded; comparisons are paired within a masking condition. Three seeds provide a variability estimate, not a formal significance claim. Subject adaptation at other layers would require another declared architectural experiment.