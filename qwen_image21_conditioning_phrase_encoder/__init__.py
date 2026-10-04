import json

from comfy_api.latest import ComfyExtension, io
from comfy_extras.nodes_qwen import TextEncodeQwenImage21

from .attention import attach_phrase_hook
from .phrases import ZERO_WEIGHT_BIAS, build_layout, finish_layout, make_offset_tokenizer, parse_phrases


class PhraseClip:
    """Delegate the native edit encoder's two CLIP calls without changing its image path."""

    def __init__(self, clip, prompts):
        self.clip = clip
        self.prompts = iter(prompts)
        self.tokenizer = None
        self.layout = None
        self.reports = []

    def tokenize(self, text, **kwargs):
        clean_text, spans = next(self.prompts)
        tokens = self.clip.tokenize(text, **kwargs)
        self.layout = None
        if spans:
            if self.tokenizer is None:
                self.tokenizer = make_offset_tokenizer()
            self.layout = build_layout(clean_text, spans, tokens, len(kwargs.get("images", ())), self.tokenizer)
        self.clean_text = clean_text
        return tokens

    def encode_from_tokens_scheduled(self, tokens):
        conditioning = self.clip.encode_from_tokens_scheduled(tokens)
        if self.layout is None:
            self.reports.append({"active": False, "clean_text": self.clean_text, "weighted_spans": []})
            return conditioning
        output, reports = [], []
        for tensor, metadata in conditioning:
            biases, report = finish_layout(self.layout, tensor, metadata)
            output.extend(attach_phrase_hook([[tensor, metadata]], tensor.shape[1], biases))
            reports.append(report)
        self.reports.append({"active": any(r["active"] for r in reports), "entries": reports})
        return output


class QwenImage21ConditioningPhraseEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = TextEncodeQwenImage21.define_schema()
        schema.node_id = "QwenImage21ConditioningPhraseEncoder"
        schema.display_name = "Qwen Image 2.1 Edit — Phrase Weights"
        schema.description = (
            "Native Qwen Image 2.1 image-edit encoding with conditioning-scoped (phrase:weight) syntax. "
            "Weights multiply target-image attention odds for the selected text keys; 1 is neutral. "
            "Reference-image encoding, text embeddings, and native prefix caching are preserved."
        )
        for field in schema.inputs:
            if field.id in ("prompt", "negative_prompt"):
                field.dynamic_prompts = False
                field.tooltip = "Use (phrase:weight), e.g. Change the jacket to (dark green:1.2). 1 is neutral; 0 uses a -80 score bias."
        schema.outputs.append(io.String.Output(display_name="phrase_report"))
        return schema

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, vae=None, resolution=1024, images=None):
        positive_text, negative_text = parse_phrases(prompt), parse_phrases(negative_prompt)
        adapter = PhraseClip(clip, (positive_text, negative_text))
        native = TextEncodeQwenImage21.execute(adapter, positive_text[0], negative_text[0], vae, resolution, images)
        positive, negative, latent = native.result
        report = {
            "scope": "this conditioning; target-image queries to phrase text keys; all native DiT blocks",
            "formula": "attention score += log(weight)",
            "zero_weight_bias": ZERO_WEIGHT_BIAS,
            "native_conditioning_preserved": True,
            "positive": adapter.reports[0],
            "negative": adapter.reports[1],
        }
        return io.NodeOutput(positive, negative, latent, json.dumps(report, ensure_ascii=False, indent=2))


class QwenPhraseExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenImage21ConditioningPhraseEncoder]


async def comfy_entrypoint():
    return QwenPhraseExtension()
