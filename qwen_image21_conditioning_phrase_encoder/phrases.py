import math
import numbers
import re
from decimal import Decimal
from pathlib import Path

from transformers import Qwen2TokenizerFast

from comfy.text_encoders import qwen_image21


TOKEN_KEY = "qwen3vl_8b"
ZERO_WEIGHT_BIAS = -80.0
WEIGHT_PATTERN = re.compile(
    r"\((?P<phrase>[^()]+):(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)"
)


def parse_phrases(text):
    parts, spans = [], []
    cursor = length = 0
    for match in WEIGHT_PATTERN.finditer(text):
        raw_phrase = match.group("phrase")
        phrase = raw_phrase.strip()
        if not phrase:
            raise ValueError("A weighted phrase cannot be empty.")
        decimal_weight = Decimal(match.group("weight"))
        weight = float(decimal_weight)
        if decimal_weight < 0 or not math.isfinite(weight) or (weight == 0 and decimal_weight != 0):
            raise ValueError(f"Weight for {phrase!r} must be a finite, representable number of zero or greater.")
        preceding = text[cursor:match.start()]
        parts.extend((preceding, raw_phrase))
        length += len(preceding)
        spans.append({
            "phrase": phrase,
            "weight": weight,
            "start": length + len(raw_phrase) - len(raw_phrase.lstrip()),
            "end": length + len(raw_phrase.rstrip()),
            "source": match.group(0),
        })
        length += len(raw_phrase)
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts), spans


def make_offset_tokenizer():
    path = Path(qwen_image21.__file__).parent / "qwen25_tokenizer"
    return Qwen2TokenizerFast.from_pretrained(path, local_files_only=True)


def build_layout(clean_text, spans, tokens, image_count, tokenizer):
    if TOKEN_KEY not in tokens or len(tokens[TOKEN_KEY]) != 1 or "keep_vision" not in tokens:
        raise ValueError("Connect Qwen Image 2.1's qwen3vl_8b CLIP, loaded with type qwen_image.")

    if clean_text.startswith("<|im_start|>"):
        rendered, offset = clean_text, 0
    else:
        refs = " ".join(f"<image{i + 1}>{qwen_image21.VISION_BLOCK}" for i in range(image_count))
        template = qwen_image21.T2I_TEMPLATE.replace("{}", refs + "{}", 1)
        offset = template.index("{}")
        rendered = template.format(clean_text or " ")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    full_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]

    actual_ids, image_indices = [], []
    for index, item in enumerate(tokens[TOKEN_KEY][0]):
        token = item[0]
        if isinstance(token, numbers.Integral):
            actual_ids.append(int(token))
        elif isinstance(token, dict) and token.get("type") == "image":
            actual_ids.append(151655)
            image_indices.append(index)
        else:
            raise ValueError("Phrase rows cannot be mapped through textual-inversion embeddings.")
    padding = len(actual_ids) - len(full_ids)
    if padding > 0 and actual_ids[len(full_ids):] == [151643] * padding:
        full_ids += [151643] * padding
        offsets += [(0, 0)] * padding
    if full_ids != actual_ids:
        raise ValueError("The connected CLIP's tokens differ from the native Qwen Image 2.1 phrase layout.")

    im_starts = [i for i, token in enumerate(full_ids) if token == 151644]
    trim = im_starts[1] if len(im_starts) > 1 else 0
    owners, mapped_spans = {}, []
    for span_index, span in enumerate(spans):
        selected = []
        for index, (start, end) in enumerate(offsets):
            start, end = start - offset, end - offset
            if start >= span["end"] or end <= span["start"]:
                continue
            if index < trim or index in image_indices:
                raise ValueError("Weight phrases in the retained edit instruction, not in system or image markers.")
            piece = tokenizer.decode([full_ids[index]], clean_up_tokenization_spaces=False)
            outside = clean_text[max(0, start):max(0, min(end, span["start"]))]
            outside += clean_text[max(0, max(start, span["end"])):min(len(clean_text), end)]
            if outside.strip():
                raise ValueError(f"Qwen token {piece!r} crosses {span['source']!r}. Include the whole token in the marked phrase.")
            if index in owners:
                raise ValueError(f"Two weighted phrases share Qwen token {piece!r}.")
            owners[index] = span_index
            selected.append(index)
        if not selected:
            raise ValueError(f"Phrase {span['phrase']!r} does not contain a Qwen token.")
        mapped_spans.append({
            **span,
            "raw_indices": selected,
            "pieces": [tokenizer.decode([full_ids[i]], clean_up_tokenization_spaces=False) for i in selected],
            "token_ids": [full_ids[i] for i in selected],
        })
    return {
        "clean_text": clean_text,
        "full_token_count": len(full_ids),
        "trim": trim,
        "image_indices": tuple(image_indices),
        "keep_vision": tokens["keep_vision"],
        "weighted_spans": mapped_spans,
    }


def finish_layout(layout, tensor, metadata):
    if tensor.ndim != 3 or tensor.shape[-1] != 4096:
        raise ValueError("Phrase weighting requires Qwen Image 2.1's native 4096-wide conditioning.")
    count = tensor.shape[1]
    trim = layout["trim"]
    images = layout["image_indices"]
    retained_images = [i for i in images if i >= trim]
    plain_count = layout["full_token_count"] - trim
    if not layout["keep_vision"]:
        expected_count = plain_count - len(retained_images)
        expected_slots = [sum(j not in images for j in range(trim, i)) for i in images]
        if count != expected_count or list(metadata.get("image_slots", ())) != expected_slots:
            raise ValueError("The encoded text rows or reference-image slots differ from the mapped Qwen Image 2.1 layout.")
    elif (not retained_images and count != plain_count) or count < plain_count:
        raise ValueError("The encoded text length differs from the mapped Qwen Image 2.1 layout.")

    biases, spans = [], []
    for span in layout["weighted_spans"]:
        rows = []
        for index in span["raw_indices"]:
            if not layout["keep_vision"]:
                row = index - trim - sum(trim <= i < index for i in images)
            elif not retained_images or index < retained_images[0]:
                row = index - trim
            elif index > retained_images[-1]:
                # All native references precede the edit text; locate its rows from the unchanged suffix.
                row = count - (layout["full_token_count"] - index)
            else:
                raise ValueError("Without a VAE, place weighted text after all image markers so expanded vision rows can be mapped exactly.")
            rows.append(row)
            if span["weight"] != 1.0:
                bias = math.log(span["weight"]) if span["weight"] > 0 else ZERO_WEIGHT_BIAS
                biases.append((row, bias))
        spans.append({key: value for key, value in span.items() if key != "raw_indices"} | {"row_indices": rows})
    return tuple(biases), {
        "active": bool(biases),
        "clean_text": layout["clean_text"],
        "token_count": count,
        "image_slots": list(metadata.get("image_slots", ())),
        "vision_rows_retained": layout["keep_vision"] and bool(images),
        "weighted_spans": spans,
    }
