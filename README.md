# ComfyUI-qwen_img_2_1_enhancer

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)

Reference-image strength, masked reference control, and prompt phrase weighting for **Qwen Image 2.1** in ComfyUI.

Adjust the influence of whole reference images or mask-selected regions, and emphasize selected parts of a prompt. Reference controls and phrase weighting can be used independently or together.

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
| **Qwen Image 2.1 Reference Strength Masked** | `MODEL` | Applies reference strength to mask-selected tokens, with optional exclusion of unselected tokens. |
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

## Reference Strength Masked

Use **Qwen Image 2.1 Reference Strength Masked** in the MODEL path after your model and LoRA loaders. Connect a mask painted on the reference image you want to control. Keep the reference images and VAE connected to the native Qwen Image 2.1 encoder or the included Phrase Weights encoder.

1. Set `reference_index` to the reference's position: `1` for the first connected image, `2` for the second, and so on. Empty encoder inputs are skipped.
2. Connect that reference's mask. White marks the source region to select; gray values contribute partial coverage.
3. Match `mask_resize_method` to the image's resize path, then choose the selection threshold and strength.
4. Send the node's MODEL output to the sampler.

| Input | Default | Description |
|---|---:|---|
| `model` | — | Native Qwen Image 2.1 MODEL. |
| `mask` | — | Mask in the selected reference's coordinate system. The first batch item is used. |
| `reference_index` | `1` | Reference to control, using the encoder's connected-image order. |
| `strength` | `4.0` | Attention-odds multiplier for selected tokens: `1` is native, above `1` increases priority, below `1` reduces it, and `0` blocks their direct attention from generated-image queries. |
| `mask_resize_method` | `nearest-exact` | Mask interpolation before token selection. Use `nearest-exact` for an external nearest-exact image resize, or `lanczos` when the native encoder resizes the original image. |
| `mask_threshold` | `1.0` | Minimum average mask coverage per token. `1` requires full coverage; lower values include more boundary tokens. `0` selects every token, including fully black regions. |
| `mask_behavior` | `focus_only` | Choose whether unselected reference tokens remain available or are excluded from diffusion attention. |

**Mask behaviors**

- `focus_only`: applies strength only to selected tokens. Unselected tokens retain native weighting.
- `zero_unmasked_tokens`: also blocks unselected reference keys in every diffusion attention block, including reference-prefix processing.

The mask is resized to the selected reference's actual encoded pixel dimensions, then averaged over each **16 × 16 pixel** token footprint. Different source and encoded resolutions are supported. Cropping, padding, rotation, or a sequence of image resizes must be matched in the mask path; the node cannot infer those transformations.

Chain masked nodes to control multiple references with separate masks and settings. Repeating an index replaces the earlier masked-node rule for that reference. `strength = 1` with `focus_only` removes that masked-node adjustment.

This mask selects **source reference tokens**, not output pixels to preserve. Excluding tokens from diffusion attention does not erase information already mixed into retained tokens or text conditioning by the encoders.

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
| `vae` | Native Qwen Image 2.1 VAE. Required when using either reference-strength node. |
| `resolution` | Reference resizing target. Default `1024`; `0` keeps references near their original dimensions, rounded to multiples of 32. |
| `images` | Reference images; additional sockets appear as needed. |
| `positive`, `negative` | Conditioning outputs for the sampler. |
| `latent` | Empty latent sized from the first reference after resizing. |
| `phrase_report` | Optional JSON text report showing the clean prompts, weights, and selected token rows. |

## Combining the Nodes

Connect Reference Strength or Reference Strength Masked to the sampler's MODEL input and Phrase Weights to its conditioning and latent inputs:

```text
MODEL -> Reference Strength (or Masked) -------------> sampler MODEL

CLIP + VAE + images -> Phrase Weights
                           |------------------------> sampler positive
                           |------------------------> sampler negative
                           `------------------------> sampler latent_image
```

Adjust one control at a time while keeping other settings fixed to compare its effect.

## How It Works

These nodes adjust attention scores inside the native Qwen Image 2.1 transformer. For a positive weight `w`, the selected scores receive `log(w)` before softmax, changing their relative attention priority while retaining normalization.

Reference Strength selects all keys belonging to one reference image. Reference Strength Masked selects keys within that reference using mask coverage and can exclude the remaining reference keys. Phrase Weights selects the text keys belonging to marked phrases and carries their weights with the conditioning. Text embeddings and reference pixels are not rescaled or composited into the output. Native prefix caching is preserved.

Weights control attention priority, so a value of `2.0` does not imply twice the visible effect. Large values can dominate other instructions or references and reduce editing flexibility.

## Compatibility

- Requires the native **Qwen Image 2.1** implementation.
- Attention backends must support additive masks.
- Phrase mapping requires whole tokenizer pieces and does not support textual-inversion embeddings.
- Changes to conditioning token rows after phrase encoding can invalidate the mapping.

## License

[MIT](LICENSE).
