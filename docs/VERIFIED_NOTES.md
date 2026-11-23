# VERIFIED NOTES — anticipation library

Source: https://github.com/jthickstun/anticipation
These notes will be updated once the library is cloned and its source read on Colab.

## Items to verify (all UNVERIFIED until run on Colab)

### Melody/accompaniment separation
- Generated-file metrics use the saved input melody and one-to-one onset matching
  within 25 ms.
- Real MAESTRO windows have no separate melody reference track, so classifier
  training and real-vs-generated distance/memorization comparisons remove the
  skyline melody and retain the remaining notes as accompaniment.
- **UNVERIFIED**: the skyline heuristic's quality on the full MAESTRO windows;
  inspect its output on representative excerpts before treating the metrics as
  musically definitive.

### CONTROL_OFFSET
- **UNVERIFIED**: The integer value of `CONTROL_OFFSET` in `anticipation/config.py`.
- Expected: a large integer added to pitch/duration tokens to mark them as control tokens.
- Action: `grep -n CONTROL_OFFSET anticipation/config.py` on Colab after install.

### Time units in ops.clip / ops.translate
- **UNVERIFIED**: Whether `ops.clip` and `ops.translate` accept seconds or 10 ms bins.
- The `seconds=True` keyword may exist; check `anticipation/ops.py`.
- Action: read the function signatures and docstrings.

### generate() return value
- **UNVERIFIED**: Whether `generate()` returns the input tokens prepended to the output,
  or only the newly generated tokens.
- Critical for evaluation: must use only the generated window, not the prompt.
- Action: read `anticipation/sample.py`, look at the return slice.

### midi_to_events and sustain pedal (CC 64)
- **UNVERIFIED**: Whether `midi_to_events` in `anticipation/convert.py` extends note
  durations based on sustain pedal (CC 64) or ignores it.
- If ignored, Romantic-style legato may be under-represented in training data.

### Vocab size
- **UNVERIFIED**: Exact vocab size in the model config for `stanford-crfm/music-medium-800k`.
- Expected ~3000–4000 tokens (time bins + duration bins + 128 pitches + offsets).
- Action: `model.config.vocab_size` after loading.

### Token grammar
- **VERIFIED (from README)**: AMT encodes events as strict 3-token groups:
  (arrival_time, duration, pitch). Extra tokens break the grammar.
- Therefore: do NOT add new vocabulary tokens, do NOT prepend style tokens.

### LoRA target modules
- **UNVERIFIED**: Whether GPT-2's Conv1D layers are named `c_attn`, `c_proj`, `c_fc`
  in the loaded model. Verify with `[n for n, _ in model.named_modules()]`.
- `fan_in_fan_out=True` is required for Conv1D (weight is transposed vs nn.Linear).

### fp16 loading
- **UNVERIFIED**: Whether `GPT2LMHeadModel.from_pretrained("stanford-crfm/music-medium-800k", torch_dtype=torch.float16)` succeeds or whether the "architectures" field in the config causes AutoModelForCausalLM to fail first.
- Fallback: load with GPT2LMHeadModel directly.

### transformers 4.57.6 + Colab torch compatibility
- **UNVERIFIED**: Whether pinned transformers==4.57.6 installs cleanly alongside
  Colab's preinstalled CUDA torch without downgrading torch.
- The smoke test notebook will reveal this.

## Verified facts (from public documentation)

- MAESTRO v3.0.0 is CC BY-NC-SA 4.0. Any redistributed MIDI segments or adapters
  trained on it carry the same license terms.
- anticipation code and music-medium-800k weights are Apache 2.0.
- The model has ~360M parameters (GPT-2 "medium" scale).
- AMT was pretrained on **Lakh MIDI** (NOT MAESTRO). MAESTRO is used only for
  fine-tuning in this project. This distinction matters for memorization analysis.
- Lakh MIDI contains scraped commercial songs, so memorization of commercial
  melodies cannot be ruled out.
