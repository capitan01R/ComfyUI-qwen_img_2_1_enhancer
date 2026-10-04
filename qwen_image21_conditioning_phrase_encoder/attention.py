import torch

import comfy.hooks
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
from comfy.patcher_extension import WrappersMP


HOOK_ID = "qwen_image21_conditioning_phrase_encoder"


class TargetAttentionBias:
    """One diffusion call owns this object and its small, broadcastable score bias."""

    def __init__(self, target_length, key_length, biases, previous=None):
        self.target_length = target_length
        self.key_length = key_length
        self.biases = biases
        self.previous = previous
        self.bias = None

    def __call__(self, function, q, k, v, heads, mask=None, **kwargs):
        if k.shape[1] == self.key_length:
            if q.ndim != 3 or q.shape[1] != self.target_length:
                raise ValueError("Qwen Image 2.1's target attention rows changed; phrase weights cannot be mapped exactly.")
            if self.bias is None or self.bias.device != q.device or self.bias.dtype != q.dtype:
                self.bias = q.new_zeros((1, 1, 1, self.key_length))
                indices, values = zip(*self.biases)
                self.bias[..., list(indices)] = self.bias.new_tensor(values)
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
        if self.previous is not None:
            return self.previous(function, q, k, v, heads, mask=mask, **kwargs)
        return function(q, k, v, heads, mask=mask, **kwargs)


class PhraseDiffusionWrapper:
    def __init__(self, token_count, biases):
        self.token_count = token_count
        self.biases = tuple(biases)

    def __call__(self, executor, x, timestep, context, ref_latents=None, image_slots=None, transformer_options=None, **kwargs):
        if not isinstance(executor.class_obj, QwenImage21Transformer2DModel):
            raise ValueError("Qwen Image 2.1 phrase conditioning must be sampled with a Qwen Image 2.1 model.")
        if context.shape[1] != self.token_count:
            raise ValueError("Conditioning token rows changed after phrase encoding; encode the combined prompt in this node.")
        refs = list(ref_latents or [])
        slots = (list(image_slots or []) + [self.token_count] * len(refs))[:len(refs)]
        ref_lengths = [r.shape[-2] * r.shape[-1] for r in refs]
        target_length = x.shape[-2] * x.shape[-1]
        key_length = self.token_count + sum(ref_lengths) + target_length
        biases = tuple((row + sum(n for slot, n in zip(slots, ref_lengths) if slot <= row), value) for row, value in self.biases)
        options = dict(transformer_options or {})
        attention = TargetAttentionBias(target_length, key_length, biases, options.get("optimized_attention_override"))
        options["optimized_attention_override"] = attention
        # Prefix queries never see the bias, so the native prefix cache remains valid.
        return executor(x, timestep, context, ref_latents, image_slots, options, **kwargs)


def attach_phrase_hook(conditioning, token_count, biases):
    if not biases:
        return conditioning
    hook = comfy.hooks.TransformerOptionsHook(
        {"wrappers": {WrappersMP.DIFFUSION_MODEL: {HOOK_ID: [PhraseDiffusionWrapper(token_count, biases)]}}},
        hook_scope=comfy.hooks.EnumHookScope.HookedOnly,
    )
    hook.hook_id = HOOK_ID
    group = comfy.hooks.HookGroup()
    group.add(hook)
    output = []
    for tensor, metadata in conditioning:
        updated = metadata.copy()
        existing = updated.get("hooks")
        if existing is None:
            updated.pop("hooks", None)
        elif any(h.hook_id == HOOK_ID for h in existing.hooks):
            existing = existing.clone()
            for old in tuple(existing.hooks):
                if old.hook_id == HOOK_ID:
                    existing.remove(old)
            if len(existing):
                updated["hooks"] = existing
            else:
                updated.pop("hooks", None)
        output.append([tensor, updated])
    return comfy.hooks.set_hooks_for_conditioning(output, group)
