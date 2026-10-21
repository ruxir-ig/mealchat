"""Vision path: a separate multimodal model sees photos; the text agent never does.

Pipeline: resize/encode locally (cheap, cuts upload + token cost) -> one
structured call to the vision model -> a parsed VisionResult that the text
agent consumes as context. If the model can't identify the food or the parse
fails, we surface uncertainty and ASK the user — never silently guess.
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from PIL import Image, ImageOps

MAX_DIM = 1024  # longest side sent to the vision model
JPEG_QUALITY = 85

_VISION_PROMPT = """You are a food-photo analyst for a calorie-tracking chat app.
Look at this photo of food. Respond with ONLY a JSON object, no markdown, no prose:
{
  "description": "<one short sentence describing the food>",
  "items": [{"food": "<name>", "quantity": <number, for the WHOLE visible portion>, "unit": "<piece|cup|bowl|plate|box|serving>", "confidence": "<high|medium|low>"}],
  "single_serving": <true if this looks like one person's own plate/meal; false for a shared or family spread>,
  "uncertain": <true only if you cannot reliably identify the food at all>,
  "question": "<if uncertain: the single best question to ask the user, else null>"
}
Rules: quantity covers the whole visible portion. Merge duplicates (four rotis on one plate -> one item, quantity 4). Name every distinct food — if actual food is visible but its exact type is unclear, use the best generic name (e.g. "mixed vegetable curry", "dal") rather than hedging or merging dishes. But if the image does not clearly show real food (an object, a screenshot, a solid color, a drawing, an empty plate), you MUST set uncertain=true and give the question — never invent a food for a non-food image."""


@dataclass
class VisionResult:
    description: str = ""
    items: list[dict] = field(default_factory=list)
    uncertain: bool = False
    question: str | None = None
    raw: str | None = None
    meta: dict = field(default_factory=dict)  # original/resized dims, bytes

    def prompt_block(self) -> str:
        """Rendered into the user turn for the TEXT agent (clearly delimited)."""
        lines = ["[Photo analysis by the vision model — not the user's own words]"]
        lines.append(f"Description: {self.description or 'unavailable'}")
        single = self.meta.get("single_serving")
        if single is not None:
            lines.append("Looks like one person's own meal." if single else "Looks like a SHARED/family spread.")
        if self.items:
            for it in self.items:
                conf = it.get("confidence", "?")
                lines.append(
                    f"- {it.get('food', '?')} x{it.get('quantity', '?')} "
                    f"{it.get('unit', '')} (confidence: {conf})"
                )
        else:
            lines.append("- no items identified")
        if self.uncertain:
            lines.append(
                f"UNCERTAIN — do NOT log from this photo; ask the user: "
                f"{self.question or 'can you tell me what this is and how much of it you ate?'}"
            )
        else:
            lines.append("Confident enough to log if the user doesn't say otherwise.")
        return "\n".join(lines)


def prepare_image(path: str | Path, max_dim: int = MAX_DIM) -> tuple[bytes, dict]:
    """EXIF-orient, downscale to max_dim on the longest side, re-encode JPEG.

    Returns (jpeg_bytes, meta) — meta records the size reduction for the
    latency report.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    orig_w, orig_h = img.size
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=JPEG_QUALITY)
    data = buf.getvalue()
    meta = {
        "original": f"{orig_w}x{orig_h}",
        "sent": f"{img.size[0]}x{img.size[1]}",
        "jpeg_kb": round(len(data) / 1024, 1),
    }
    return data, meta


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def analyze_photo(
    llm: BaseChatModel, image_path: str | Path, max_dim: int = MAX_DIM
) -> VisionResult:
    """One structured vision call. Any doubt -> VisionResult(uncertain=True)."""
    jpeg, meta = prepare_image(image_path, max_dim)
    b64 = base64.b64encode(jpeg).decode()
    result = VisionResult(meta=meta)
    try:
        msg = llm.invoke(
            [
                (
                    "user",
                    [
                        {"type": "text", "text": _VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                )
            ]
        )
        raw = msg.content if isinstance(msg.content, str) else "".join(
            getattr(p, "text", "") or "" for p in msg.content
        )
        result.raw = raw
        data = _parse_json(raw)
        if not data:
            return VisionResult(
                uncertain=True,
                question="what's in this photo and roughly how much of it did you eat?",
                raw=raw,
                meta=meta,
            )
        items = []
        for it in data.get("items", []) or []:
            if isinstance(it, dict) and it.get("food"):
                items.append(
                    {
                        "food": str(it.get("food")),
                        "quantity": float(it.get("quantity", 1) or 1),
                        "unit": str(it.get("unit", "serving")),
                        "confidence": str(it.get("confidence", "medium")).lower(),
                    }
                )
        result.description = str(data.get("description", ""))
        result.items = items
        result.uncertain = bool(data.get("uncertain", False)) or not items
        result.question = data.get("question") or None
        if "single_serving" in data:
            result.meta["single_serving"] = bool(data["single_serving"])
    except Exception as e:  # noqa: BLE001 — any failure degrades to "ask the user", never a crash
        result.uncertain = True
        result.question = "I couldn't analyze that photo — what did you eat?"
        result.description = f"(vision call failed: {e})"
    if result.uncertain and not result.question:
        result.question = "what's in this photo and roughly how much of it did you eat?"
    return result
