import math

import torch

from comfy_api.latest import ComfyExtension, io
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
from comfy.patcher_extension import WrappersMP


WRAPPER_KEY = "qwen_image21_reference_strength"


class ReferenceAttention:
    def __init__(self, target_length, key_length, start, end, log_strength, attention_modules, previous):
        self.target_length = target_length
        self.key_length = key_length
        self.start = start
        self.end = end
        self.log_strength = log_strength
        self.attention_modules = attention_modules
        self.previous = previous
        self.bias = None
        self.applied_blocks = set()

    def __call__(self, function, q, k, v, heads, mask=None, **kwargs):
        options = kwargs["transformer_options"]
        block_index = options["block_index"]
        if k.shape[1] == self.key_length:
            if q.ndim != 3 or q.shape[1] != self.target_length:
                raise ValueError("Qwen Image 2.1 target rows changed; reference strength cannot be mapped.")
            if self.bias is None or self.bias.dtype != q.dtype or self.bias.device != q.device:
                self.bias = q.new_zeros((1, 1, 1, self.key_length))
                self.bias[..., self.start:self.end] = self.log_strength
            if mask is None:
                mask = self.bias
            else:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                elif mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                if mask.dtype == torch.bool:
                    mask = self.bias.masked_fill(~mask, float("-inf"))
                else:
                    mask = mask + self.bias.to(dtype=mask.dtype)
            self.applied_blocks.add(block_index)

        # Native wrap_attn drops preferred_attention before calling overrides.
        # Resolve the same Qwen block's selected callable without changing it.
        selected = self.attention_modules[block_index].function
        if selected is not None:
            function = selected
        if self.previous is not None:
            return self.previous(function, q, k, v, heads, mask=mask, **kwargs)
        return function(q, k, v, heads, mask=mask, **kwargs)


class ReferenceDiffusionWrapper:
    def __init__(self, reference_index, strength):
        self.reference_index = reference_index
        self.log_strength = math.log(strength) if strength else float("-inf")

    def __call__(self, executor, x, timestep, context, ref_latents=None, image_slots=None, transformer_options=None, **kwargs):
        model = executor.class_obj
        if not isinstance(model, QwenImage21Transformer2DModel):
            raise ValueError("Reference Strength requires the native Qwen Image 2.1 model.")
        refs = list(ref_latents or [])
        index = self.reference_index - 1
        if index >= len(refs):
            raise ValueError(f"Reference {self.reference_index} was selected, but conditioning contains {len(refs)} reference latents. Connect the references and VAE to the Qwen Image 2.1 encoder.")

        text_length = context.shape[1]
        slots = (list(image_slots or []) + [text_length] * len(refs))[:len(refs)]
        lengths = [ref.shape[-2] * ref.shape[-1] for ref in refs]
        start = slots[index] + sum(lengths[:index])
        target_length = x.shape[-2] * x.shape[-1]
        key_length = text_length + sum(lengths) + target_length
        options = dict(transformer_options or {})
        attention = ReferenceAttention(
            target_length, key_length, start, start + lengths[index], self.log_strength,
            [block.attn.comfy_attention for block in model.transformer_blocks],
            options.get("optimized_attention_override"),
        )
        options["optimized_attention_override"] = attention
        output = executor(x, timestep, context, ref_latents, image_slots, options, **kwargs)
        if len(attention.applied_blocks) != len(model.transformer_blocks):
            raise ValueError("Another patch bypassed Qwen Image 2.1 target attention; reference strength was not applied to every block.")
        return output


class QwenImage21ReferenceStrength(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21ReferenceStrength",
            display_name="Qwen Image 2.1 Reference Strength",
            category="model/patch/qwen image",
            is_experimental=True,
            description="Control a selected reference's attention weight during Qwen Image 2.1 editing. Changes target-to-reference scores while preserving normalized attention, image values, and native prefix caching. This is not a hard preservation lock.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("reference_index", default=1, min=1, max=16,
                             tooltip="1-based order of connected reference images in the encoder. Use another node for a different reference."),
                io.Float.Input("strength", default=1.2, min=0.0, max=8.0, step=0.05,
                               tooltip="1 is native. Above 1 increases this reference's attention odds; below 1 reduces them. 0 blocks direct attention to its latent tokens, but not reference information already present in text. High values can suppress requested edits."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, reference_index=1, strength=1.2):
        if not isinstance(model.get_model_object("diffusion_model"), QwenImage21Transformer2DModel):
            raise ValueError("Reference Strength requires the native Qwen Image 2.1 model.")
        if reference_index < 1:
            raise ValueError("Reference indices start at 1.")
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("Reference strength must be finite and nonnegative.")
        key = f"{WRAPPER_KEY}:{reference_index}"
        if strength == 1 and not model.get_wrappers(WrappersMP.DIFFUSION_MODEL, key):
            return io.NodeOutput(model)
        patched = model.clone()
        patched.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, key)
        if strength != 1:
            patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, key, ReferenceDiffusionWrapper(reference_index, strength))
        return io.NodeOutput(patched)


class QwenReferenceExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenImage21ReferenceStrength]


async def comfy_entrypoint():
    return QwenReferenceExtension()
