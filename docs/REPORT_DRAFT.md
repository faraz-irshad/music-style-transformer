# Style-Conditioned Piano Accompaniment Generation

**BSc Thesis — ELTE Budapest**
*Draft — figures and numbers to be filled in after Colab runs*

---

## 1. Introduction

This thesis addresses **style-conditioned piano accompaniment generation**: given a fixed melody, automatically generate a piano accompaniment in a specified historical style (Baroque or Romantic). The melody is held fixed; only the accompaniment changes. This is distinct from full style transfer (which restyles an existing arrangement) and from cross-genre conversion (e.g. jazz-to-rock).

The accurate claim for this work is: **Style-Conditioned Piano Accompaniment Generation.**

---

## 2. Related Work

**Anticipatory Music Transformer (AMT)** [Thickstun et al., 2023]: A GPT-2-scale autoregressive model (~360M parameters) pretrained on **Lakh MIDI** (not MAESTRO). It uses a strict 3-token event grammar (arrival time, duration, pitch) and supports hard melodic constraints via control tokens (pitch + CONTROL_OFFSET). This is the backbone of our system.

**Groove2Groove** [Cífka et al., 2020]: Style transfer for accompaniment by disentangling content and style in a VAE. Operates on symbolic music; requires paired data. This is true style transfer (restyling an existing arrangement), which our work does not attempt.

**MuseMorphose** [Wu & Yang, 2023]: Transformer-based style transfer conditioned on bar-level attributes. Requires style tokens prepended to the sequence. We cannot use this approach because AMT's fixed 3-token grammar does not accommodate extra tokens.

**ImprovNet** [Kosta et al., 2024]: Iterative refinement for jazz improvisation style transfer. Operates on a different task (improvisation, not accompaniment).

**Compose & Embellish** [Wu et al., 2023]: Two-stage generation separating lead sheet from embellishment. Related in spirit to our melody-fixed accompaniment approach.

**Key difference from all above**: We do not restyle an existing arrangement. We generate a new accompaniment from scratch, conditioned on a fixed melody and a style label. Jazz-to-rock conversion is future work only.

---

## 3. Method

### 3.1 System Variants

| Label | Description |
|-------|-------------|
| `base_amt` | Pretrained AMT, melody as controls, no style conditioning. Neutral baseline. |
| `layer_a_scripted` | Rule-based generator. NOT a learned model. Its contrast must be reported from measured classifier probabilities and feature differences, not inferred from the coded constants. |
| `layer_c_lora` | Base AMT + LoRA adapter (r=8, alpha=16) trained on MAESTRO. |

### 3.2 Layer A (Rule-Based)

Baroque rules: walking bass (stepwise quarter notes), staccato articulation (40% of beat duration), constant pulse, slow harmonic rhythm (one chord per bar), inner voices on beats 2 and 4.

Romantic rules: octave bass on beat 1, rolling arpeggios on beats 2-4, legato articulation (95% of beat duration), rubato (+-5% timing jitter), rising dynamics (velocity increases linearly over the piece).

Chords are selected from I, IV, V, and vi per bar using duration-weighted
melody pitch-class weights in the estimated major key. Layer A is deterministic
given a fixed seed. **UNVERIFIED measurement:** render `twinkle`, `g_major`,
and `a_minor` with `python scripts/measure_layer_a.py`; include the reported
classifier probabilities and simple feature differences here after execution.

### 3.3 Layer C (LoRA Fine-Tuning)

LoRA configuration: r=8, lora_alpha=16, lora_dropout=0.05, target_modules=[c_attn, c_proj, c_fc], fan_in_fan_out=True (required for GPT-2 Conv1D layers). Base model frozen in fp16; LoRA weights in fp32 with AMP autocast and GradScaler.

Training: seq_len=512, micro_batch=4, gradient_accumulation=4, lr=2e-4 with 15-step warmup and cosine decay, 200 optimizer steps per adapter. ~50% of training windows use anticipatory interleaved form (melody as controls, loss masked on control tokens). 12-semitone transposition augmentation applied.

**Important constraint**: No new vocabulary tokens are added. AMT's 3-token grammar is preserved exactly.

---

## 4. Data

MAESTRO v3.0.0 (CC BY-NC-SA 4.0). Baroque group: Bach, Scarlatti, Handel. Romantic group: Chopin, Liszt. Rows with "/" in canonical_composer (transcriptions) are dropped.

**Honest note on Baroque data**: MAESTRO's Baroque representation is dominated by Bach and Scarlatti. Handel may have few or no entries. Counts are reported in `results/split_counts.json` (to be filled after data run).

Splitting: piece-disjoint GroupShuffleSplit by (composer, title). No piece appears in more than one split. At most 2 performances per work, at most 5 non-overlapping 15-second windows per performance.

---

## 5. Evaluation

### 5.1 Classifier

L2-regularized logistic regression (cross-checked with random forest) trained on real MAESTRO windows. Features evaluated on the **accompaniment only** (melody stripped).

Feature blocks:
1. **Harmony**: tonic-aligned pitch-class entropy, non-diatonic pitch ratio, vertical interval rates (major/minor thirds and sixths, dominant sevenths, diminished sevenths).
2. **Texture**: mean and 90th-percentile chord spread, mean polyphony per 50ms frame, fraction of accompaniment above melody.
3. **Rhythm**: coefficient of variation of IOIs, log-IOI ratio histogram (5 bins), legato/overlap ratio.

Excluded (shortcuts): raw BPM, raw IOI histograms, velocities, absolute-key pitch-class histograms, beat-grid syncopation.

Gate: held-out accuracy >= 75% required before proceeding. Result: **[TO BE FILLED]**

Block ablation results: **[TO BE FILLED — see results/classifier_report.json]**

### 5.2 Generation Metrics

For each generated clip (accompaniment only):
- Classifier style probability (Baroque / Romantic)
- Pitch-class entropy
- 4-gram repetition rate
- Skyline dominance (fraction of melody onsets where melody remains highest pitch)
- Melody-accompaniment clash rate

Distribution distances (generated vs. real test set):
- Jensen-Shannon divergence on key-normalized pitch-class histogram
- 1-Wasserstein distance on chord spread, log-IOI ratio, polyphony

### 5.3 Statistical Tests

Paired tests across identical melodies:
- Exact McNemar test on classifier predictions (base_amt vs. layer_c_lora)
- Paired Wilcoxon signed-rank test on style probability shift

95% confidence intervals: Wilson score for proportions.

### 5.4 Memorization Check

Longest transposition-invariant pitch-interval n-gram shared between generated clips and the fine-tuning corpus. Maximum and 95th percentile reported. **[TO BE FILLED]**

Note: AMT was pretrained on Lakh MIDI, which contains scraped commercial songs. Memorization of commercial melodies cannot be ruled out.

---

## 6. Results

**[All numbers to be filled after Colab runs. No invented figures.]**

### 6.1 Classifier Accuracy

| Block | LR Accuracy | LR Macro-F1 | RF Accuracy |
|-------|-------------|-------------|-------------|
| Harmony only | — | — | — |
| Texture only | — | — | — |
| Rhythm only | — | — | — |
| All features | — | — | — |

### 6.2 Style Classification of Generated Outputs

| Layer | Style | Classifier Accuracy | 95% CI |
|-------|-------|---------------------|--------|
| base_amt | baroque | — | — |
| base_amt | romantic | — | — |
| layer_a_scripted | baroque | — | — |
| layer_a_scripted | romantic | — | — |
| layer_c_lora | baroque | — | — |
| layer_c_lora | romantic | — | — |

### 6.3 Paired Tests (base_amt vs. layer_c_lora)

| Style | McNemar p | Wilcoxon p |
|-------|-----------|------------|
| baroque | — | — |
| romantic | — | — |

---

## 7. Limitations

- **Small Baroque corpus**: MAESTRO's Baroque representation is limited, mostly Bach and Scarlatti. This may cause the Baroque adapter to underfit.
- **Short training**: 200 optimizer steps is very short. If adapters show little effect over the base model, this is the most likely cause.
- **No listening test**: Perceptual evaluation was cut due to time constraints.
- **Layer A is not ML**: its style cues are hand-coded, not learned. Report measured probabilities/features from `scripts/measure_layer_a.py`; do not treat articulation constants as evidence of perceptual or classifier separation.
- **Memorization**: Cannot rule out memorization of commercial melodies from Lakh MIDI pretraining.
- **UNVERIFIED items**: See docs/VERIFIED_NOTES.md for items that could not be verified without a Colab GPU run.

---

## 8. Licensing

- MAESTRO v3.0.0: CC BY-NC-SA 4.0. Any redistributed MIDI segments or adapters trained on it carry the same terms.
- anticipation library and music-medium-800k weights: Apache 2.0.
- Lakh MIDI (pretraining data for AMT): contains scraped commercial songs; memorization cannot be ruled out.

---

## 9. Future Work

- Jazz-to-rock style conversion (cross-genre, not just historical period)
- Layer B: AMT with key-matched MAESTRO excerpt as history
- Listening test with human evaluators
- Larger Baroque corpus (e.g. JSB Chorales, OpenScore)
- Longer LoRA training with more data

---

## References

- Thickstun et al. (2023). Anticipatory Music Transformer. arXiv:2306.08620.
- Cifka et al. (2020). Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data. ISMIR 2020.
- Wu & Yang (2023). MuseMorphose: Full-Song and Fine-Grained Piano Music Style Transfer with One Transformer VAE. TASLP 2023.
- Wu et al. (2023). Compose & Embellish. ICASSP 2023.
- Hawthorne et al. (2019). Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset. ICLR 2019.
