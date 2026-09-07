"""인물 사진에서 옷 영역을 회색으로 지워 clothing-agnostic 입력을 만든다.

try-on LoRA(xocialize/tryon-FLUX.2-klein-4B-lora)의 1번 입력 형식. 같은 저자가 학습에 쓴
segformer_b2_clothes로 상의·치마·바지·원피스 픽셀을 찾아 회색으로 채운다.

    uv run --extra torch python experiments/08-klein-tryon/agnostic.py outputs/tryon/avatar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

MODEL = "mattmdjaga/segformer_b2_clothes"
CLOTHES_LABELS = {4: "upper-clothes", 5: "skirt", 6: "pants", 7: "dress"}
GREY = (128, 128, 128)


def clothes_mask(image: Image.Image, device: str) -> np.ndarray:
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    labels = upsampled.argmax(dim=1)[0].cpu().numpy()
    return np.isin(labels, list(CLOTHES_LABELS))


def grey_out(image: Image.Image, mask: np.ndarray, dilate_px: int) -> Image.Image:
    if dilate_px > 0:
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.array(mask_image.filter(ImageFilter.MaxFilter(dilate_px * 2 + 1))) > 0
    pixels = np.array(image)
    pixels[mask] = GREY
    return Image.fromarray(pixels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--out", help="기본값: <입력>_agnostic.png")
    parser.add_argument("--dilate", type=int, default=8, help="마스크 팽창 픽셀 (옷 경계 여유)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    image = Image.open(args.image).convert("RGB")
    mask = clothes_mask(image, device)
    result = grey_out(image, mask, args.dilate)
    out = Path(args.out or Path(args.image).with_name(f"{Path(args.image).stem}_agnostic.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(out)
    print(f"device={device} masked={mask.mean() * 100:.1f}% of pixels → {out}")


if __name__ == "__main__":
    main()
