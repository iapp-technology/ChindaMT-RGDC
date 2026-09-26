# Local patches to vendored submodules

Both submodules need a local fix. A plain `git clone --recursive` fetches upstream **without**
them, so apply the patches after cloning.

## Applying

```bash
git submodule update --init --recursive
git -C alpaca_eval apply ../patches/alpaca_eval.patch
git -C training/LlamaFactory apply ../../patches/llamafactory.patch
```

Verify with `git -C <submodule> diff --stat`: alpaca_eval two files (12 insertions, 5 deletions),
LlamaFactory one file (9 insertions).

## alpaca_eval.patch

| File | Fix |
|---|---|
| `src/alpaca_eval/annotators/pairwise_evaluator.py` | Pandas dtype handling when the annotation cache is empty or partially filled |
| `src/alpaca_eval/decoders/openai.py` | Retry on transient server errors such as HTTP 504 from the judge endpoint, which otherwise abort a run mid-suite |

## llamafactory.patch (upstream `436d26b`)

| File | Fix |
|---|---|
| `src/llamafactory/data/collator.py` | Supply all-zeros `token_type_ids` for gemma3 text-only batches; transformers 5.2 `create_causal_mask_mapping` requires it whenever training, and zeros (all text) is exact for a text-only dataset |

Needed only to fine-tune Gemma-3, as in the cross-family run of the paper. Pair it with
`template: gemma` (not `gemma3`): the text formats are identical, but the gemma3 mm plugin's
fake-image injection breaks packed text batches (pads batch 0 past the RoPE cache length).

## Regenerating

If a submodule working tree changes, refresh its patch:

```bash
git -C alpaca_eval diff > patches/alpaca_eval.patch
git -C training/LlamaFactory diff -- src > patches/llamafactory.patch
```
