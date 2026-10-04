# ComfyUI-qwen_img_2_1_enhancer

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)

Reference-image strength and prompt phrase weighting for **Qwen Image 2.1** in ComfyUI.

Adjust the influence of individual reference images and emphasize selected parts of a prompt. The two nodes can be used independently or together.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/capitan01R/ComfyUI-qwen_img_2_1_enhancer.git
```

Restart ComfyUI after installing or updating.

Requires ComfyUI with native **Qwen Image 2.1** support and a working model, text encoder, and VAE setup. No additional Python packages or ComfyUI core-file edits are required.

## Included Nodes

| Node | Outputs | Purpose |
|---|---|---|
| **Qwen Image 2.1 Reference Strength** | `MODEL` | Controls attention priority for a selected reference image. |
| **Qwen Image 2.1 Edit — Phrase Weights** | `positive`, `negative`, `latent`, `phrase_report` | Adds `(phrase:weight)` syntax to positive and negative prompts. |

## Reference Strength

Place **Qwen Image 2.1 Reference Strength** after your model and LoRA loaders, then connect its MODEL output to the sampler.

```text
Load Diffusion Model -> LoRA loaders (optional) -> Reference Strength -> sampler
```

Connect your reference images and VAE to **Text Encode Qwen Image 2.1** or the included **Phrase Weights** encoder.

| Input | Default | Description |
|---|---:|---|
| `model` | — | Qwen Image 2.1 MODEL, including any applied LoRAs. |
| `reference_index` | `1` | Reference to adjust: `1` is the first connected image, `2` is the second, and so on. |
| `strength` | `1.2` | Attention-odds multiplier for the selected reference. |

| Strength | Behavior |
|---:|---|
| `1.0` | Native reference weighting. |
| Above `1.0` | Increases reference priority. |
| Between `0.0` and `1.0` | Reduces reference priority. |
| `0.0` | Blocks direct attention to the selected reference's latent tokens. Information about it can remain elsewhere in the conditioning. |

Reference selection follows the encoder's connected-image order; empty inputs are skipped. The adjustment applies to the whole reference image.

Chain nodes to adjust multiple references. Repeating an index replaces its earlier strength. Setting that index's strength to `1.0` removes its adjustment.

## Phrase Weights

Use **Qwen Image 2.1 Edit — Phrase Weights** in place of the native **Text Encode Qwen Image 2.1** node.

Connect your Qwen Image 2.1 CLIP, reference images, and VAE. Send the `positive`, `negative`, and `latent` outputs to the sampler. The model follows its normal path to the sampler; this encoder does not require a MODEL connection.

### Syntax

Wrap a word or phrase in parentheses followed by a colon and a nonnegative weight:

```text
Add (warm lighting:1.3) and (soft shadows:1.1).
```

| Weight | Behavior |
|---:|---|
| `1.0` | Native phrase weighting. |
| Above `1.0` | Increases phrase priority. |
| Between `0.0` and `1.0` | Reduces phrase priority. |
| `0.0` | Strongly suppresses attention to the selected phrase tokens. |

Positive and negative prompts have independent weights. Only the marked occurrence of a phrase receives the adjustment. Use complete words or phrases in separate, non-nested sections.

### Controls and outputs

| Input/output | Description |
|---|---|
| `clip` | Native Qwen Image 2.1 text encoder, loaded with CLIP type `qwen_image`. |
| `prompt` | Positive prompt with optional weighted phrases. |
| `negative_prompt` | Negative prompt with its own optional weighted phrases. |
| `vae` | Native Qwen Image 2.1 VAE. Required when combining with Reference Strength. |
| `resolution` | Reference resizing target. Default `1024`; `0` keeps references near their original dimensions, rounded to multiples of 32. |
| `images` | Reference images; additional sockets appear as needed. |
| `positive`, `negative` | Conditioning outputs for the sampler. |
| `latent` | Empty latent sized from the first reference after resizing. |
| `phrase_report` | Optional JSON text report showing the clean prompts, weights, and selected token rows. |

## Using Both Nodes

Connect Reference Strength to the sampler's MODEL input and Phrase Weights to its conditioning and latent inputs:

```text
MODEL -> Reference Strength -------------------------> sampler MODEL

CLIP + VAE + images -> Phrase Weights
                           |------------------------> sampler positive
                           |------------------------> sampler negative
                           `------------------------> sampler latent_image
```

Adjust one control at a time while keeping other settings fixed to compare its effect.

## How It Works

Both nodes adjust attention scores inside the native Qwen Image 2.1 transformer. For a positive weight `w`, the selected scores receive `log(w)` before softmax, changing their relative attention priority while retaining normalization.

Reference Strength selects reference-image keys. Phrase Weights selects the text keys belonging to marked phrases and carries their weights with the conditioning. Text embeddings and reference pixels are not rescaled or composited into the output. Native prefix caching is preserved.

Weights control attention priority, so a value of `2.0` does not imply twice the visible effect. Large values can dominate other instructions or references and reduce editing flexibility.

## Compatibility

- Requires the native **Qwen Image 2.1** implementation.
- Attention backends must support additive masks.
- Phrase mapping requires whole tokenizer pieces and does not support textual-inversion embeddings.
- Changes to conditioning token rows after phrase encoding can invalidate the mapping.

## License

[MIT](LICENSE).
