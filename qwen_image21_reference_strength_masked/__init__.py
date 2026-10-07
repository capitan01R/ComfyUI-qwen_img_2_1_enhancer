import math

import torch
import torch.nn.functional as F

import comfy.utils
from comfy_api.latest import ComfyExtension, io
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
from comfy.patcher_extension import WrappersMP


WRAPPER_KEY = "qwen_image21_reference_strength_masked"


def select_tokens(mask, height, width, threshold, device, resize_method="nearest-exact"):
    # Qwen 2.1 has one latent token per 16x16 encoded-image pixels.
    pixel_height, pixel_width = height * 16, width * 16
    if mask.shape[-2:] != (pixel_height, pixel_width):
        mask = comfy.utils.common_upscale(mask, pixel_width, pixel_height, resize_method, "disabled")
        # Native Lanczos returns [B,H,W] for a single-channel input.
        mask = mask.reshape(1, 1, pixel_height, pixel_width)
    coverage = F.avg_pool2d(mask, kernel_size=16, stride=16).flatten()
    return (coverage >= threshold).to(device=device)


class ReferenceAttention:
    # Original Qwen Reference Strength operation: additive log weights on target -> reference attention.
    def __init__(self, target_length, key_length, start, selected, strength, behavior, attention_modules, previous):
        self.target_length = target_length
        self.key_length = key_length
        self.start = start
        self.selected = selected
        self.strength = strength
        self.block_unselected = behavior == "zero_unmasked_tokens"
        self.attention_modules = attention_modules
        self.previous = previous
        self.target_bias = None
        self.source_bias = None
        self.applied_blocks = set()

    def __call__(self, function, q, k, v, heads, mask=None, **kwargs):
        block_index = kwargs["transformer_options"]["block_index"]
        target = k.shape[1] == self.key_length
        if target or (self.block_unselected and k.shape[1] > self.start):
            if self.target_bias is None or self.target_bias.dtype != q.dtype or self.target_bias.device != q.device:
                self.source_bias = torch.zeros((1, 1, 1, self.key_length), device=q.device, dtype=q.dtype)
                keep = self.selected.to(device=q.device)
                end = self.start + keep.numel()
                if self.block_unselected:
                    self.source_bias[..., self.start:end].masked_fill_(~keep, -math.inf)
                self.target_bias = self.source_bias.clone()
                self.target_bias[..., self.start:end].masked_fill_(keep, math.log(self.strength) if self.strength else -math.inf)
            bias = self.target_bias if target else self.source_bias[..., :k.shape[1]]
            if mask is None:
                mask = bias
            else:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                if mask.dtype == torch.bool:
                    mask = bias.masked_fill(~mask, -math.inf)
                else:
                    mask = mask + bias.to(dtype=mask.dtype)
        if target:
            if q.ndim != 3 or q.shape[1] != self.target_length:
                raise ValueError("Qwen Image 2.1 target rows changed; reference tokens cannot be mapped.")
            self.applied_blocks.add(block_index)

        selected = self.attention_modules[block_index].function
        if selected is not None:
            function = selected
        if self.previous is not None:
            return self.previous(function, q, k, v, heads, mask=mask, **kwargs)
        return function(q, k, v, heads, mask=mask, **kwargs)


class ReferenceStrengthMaskedWrapper:
    def __init__(self, reference_index, strength, mask_threshold, mask_behavior, mask, mask_resize_method):
        self.reference_index = reference_index
        self.strength = strength
        self.mask_threshold = mask_threshold
        self.mask_behavior = mask_behavior
        self.mask = mask
        self.mask_resize_method = mask_resize_method

    def __call__(self, executor, x, timestep, context, ref_latents=None, image_slots=None, transformer_options=None, **kwargs):
        options = dict(transformer_options or {})
        refs = list(ref_latents or [])
        branches = options.get("cond_or_uncond", [])
        if not refs and branches and all(branch == 1 for branch in branches):
            return executor(x, timestep, context, ref_latents, image_slots, options, **kwargs)
        index = self.reference_index - 1
        if index >= len(refs):
            raise ValueError(f"Reference Strength Masked selects <image{self.reference_index}>, but this conditioning branch contains {len(refs)} VAE reference latents. Use the native Qwen Image 2.1 encoder with its VAE connected.")
        if image_slots is None or len(image_slots) != len(refs):
            raise ValueError("Reference Strength Masked needs the native Qwen Image 2.1 image_slots metadata.")
        slots = list(image_slots)
        if slots != sorted(slots) or slots[0] < 0 or slots[-1] > context.shape[1]:
            raise ValueError("Reference Strength Masked received invalid reference insertion positions.")

        lengths = [ref.shape[-2] * ref.shape[-1] for ref in refs]
        height, width = refs[index].shape[-2:]
        selected = select_tokens(self.mask, height, width, self.mask_threshold, x.device, self.mask_resize_method)
        target_length = x.shape[-2] * x.shape[-1]
        attention = ReferenceAttention(
            target_length, context.shape[1] + sum(lengths) + target_length,
            slots[index] + sum(lengths[:index]), selected, self.strength, self.mask_behavior,
            [block.attn.comfy_attention for block in executor.class_obj.transformer_blocks],
            options.get("optimized_attention_override"),
        )
        options["optimized_attention_override"] = attention
        output = executor(x, timestep, context, ref_latents, image_slots, options, **kwargs)
        if len(attention.applied_blocks) != len(executor.class_obj.transformer_blocks):
            raise ValueError("Another patch bypassed Qwen Image 2.1 target attention; Reference Strength Masked did not reach every block.")
        return output


class QwenImage21ReferenceStrengthMasked(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21ReferenceStrengthMasked",
            display_name="Qwen Image 2.1 Reference Strength Masked",
            category="model/patch/qwen image",
            is_experimental=True,
            description="Apply reference strength to source tokens selected by a mask. Resize the mask to the encoded reference dimensions before token selection; optionally exclude unselected reference tokens as attention sources throughout the diffusion transformer.",
            inputs=[
                io.Model.Input("model"),
                io.Mask.Input("mask", tooltip="Paint on the selected reference photo, with the same proportions. Uses the first mask batch item, matching the native image encoder. Coverage is averaged over each reference token's footprint."),
                io.Int.Input("reference_index", default=1, min=1, max=16,
                             tooltip="1 = <image1>, 2 = <image2>, in nonempty encoder-input order. Chain nodes for independent reference settings; repeating an index replaces this node's earlier rule."),
                io.Float.Input("strength", default=4.0, min=0.0, max=64.0, step=0.05,
                               tooltip="Original attention multiplier on selected tokens: 1 native, above 1 stronger, below 1 weaker, 0 blocks their direct target attention. No reference-budget normalization."),
                io.Combo.Input("mask_resize_method", options=["nearest-exact", "lanczos"], default="nearest-exact",
                               tooltip="Resize the source mask to the actual encoded reference pixel size before pooling 16x16 tokens. Match external nearest-exact resizing with nearest-exact; match native encoder resizing from the original with lanczos. This does not reproduce cropping or a sequence of multiple resizes."),
                io.Float.Input("mask_threshold", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Select tokens with average mask coverage >= this value. 1 requires full coverage; lower values admit boundary tokens. 0 selects every token, including black regions."),
                io.Combo.Input("mask_behavior", options=["focus_only", "zero_unmasked_tokens"], default="focus_only",
                               tooltip="focus_only: only selected tokens receive the strength change; other tokens remain native. zero_unmasked_tokens: also exclude unselected reference keys in every diffusion attention block, including prefix processing. This controls source tokens, not output pixels; already encoded information is not removed."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mask, reference_index=1, strength=4.0, mask_threshold=1.0, mask_behavior="focus_only", mask_resize_method="nearest-exact"):
        if not isinstance(model.get_model_object("diffusion_model"), QwenImage21Transformer2DModel):
            raise ValueError("Reference Strength Masked requires the native Qwen Image 2.1 model.")
        if reference_index < 1:
            raise ValueError("Reference indices start at 1: <image1>, <image2>, and so on.")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("Reference strength must be finite and nonnegative.")
        if not math.isfinite(mask_threshold) or not 0 <= mask_threshold <= 1:
            raise ValueError("Mask threshold must be between 0 and 1.")
        if mask_behavior not in ("focus_only", "zero_unmasked_tokens"):
            raise ValueError("Choose focus_only or zero_unmasked_tokens.")
        if mask_resize_method not in ("nearest-exact", "lanczos"):
            raise ValueError("Choose nearest-exact or lanczos to match the image resize path.")
        if mask.ndim not in (2, 3) or mask.numel() == 0:
            raise ValueError("Connect a nonempty MASK with shape [H,W] or [B,H,W].")
        mask = mask.reshape(-1, 1, *mask.shape[-2:])[:1].to(dtype=torch.float32, copy=True)
        if not ((mask >= 0) & (mask <= 1)).all():
            raise ValueError("Mask values must be finite and between 0 and 1.")
        key = f"{WRAPPER_KEY}:{reference_index}"
        neutral = strength == 1 and mask_behavior == "focus_only"
        if neutral and not model.get_wrappers(WrappersMP.DIFFUSION_MODEL, key):
            return io.NodeOutput(model)
        patched = model.clone()
        patched.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, key)
        if not neutral:
            patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, ReferenceStrengthMaskedWrapper(reference_index, strength, mask_threshold, mask_behavior, mask, mask_resize_method))
        return io.NodeOutput(patched)


class QwenReferenceStrengthMaskedExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenImage21ReferenceStrengthMasked]


async def comfy_entrypoint():
    return QwenReferenceStrengthMaskedExtension()
