"""
Generates visually distinct sample test images for test_cases.py.
Run from the assignment-03b directory: python test_images/create_samples.py
Requires: Pillow  (pip install Pillow)
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = Path(__file__).parent


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ------------------------------------------------------------------
# 1. whiteboard_sample.jpg
#    Off-white/grey background with marker-style text:
#    irregular line thickness simulated by drawing each line twice
#    with a 1-2 px offset and slightly varying opacity.
# ------------------------------------------------------------------
def make_whiteboard(path: Path) -> None:
    rng = random.Random(42)
    W, H = 900, 560
    img = Image.new("RGB", (W, H), color=(228, 226, 222))
    draw = ImageDraw.Draw(img)

    # Faint ruled lines to suggest a whiteboard surface
    for y in range(60, H, 48):
        draw.line([(0, y), (W, y)], fill=(210, 208, 205), width=1)

    # Slight vertical grey shading near edges (shadow)
    for x in range(20):
        alpha = int(8 * (1 - x / 20))
        draw.line([(x, 0), (x, H)], fill=(200 - alpha, 198 - alpha, 194 - alpha))
        draw.line([(W - x, 0), (W - x, H)], fill=(200 - alpha, 198 - alpha, 194 - alpha))

    title_font = _font(36)
    body_font = _font(27)

    lines = [
        ("Meeting Notes  –  Sprint Review", 32,  title_font, (28, 32, 38)),
        ("Attendees: Alan, Sarah, Tom",      96,  body_font,  (35, 38, 44)),
        ("Action: Deploy by Friday",         148, body_font,  (32, 36, 42)),
        ("  Owner: Tom",                     185, body_font,  (32, 36, 42)),
        ("Action: Send report to client",    233, body_font,  (32, 36, 42)),
        ("  Owner: Alan",                    270, body_font,  (32, 36, 42)),
        ("Blocker: staging env down",        322, body_font,  (140, 30, 30)),
        ("Decision: use v2 API",             374, body_font,  (32, 36, 42)),
        ("Next meeting: Monday 9am",         426, body_font,  (32, 36, 42)),
    ]

    for text, base_y, font, colour in lines:
        # Draw twice with slight jitter to mimic uneven marker stroke
        for dx, dy in [(0, 0), (rng.randint(0, 1), rng.randint(0, 1))]:
            draw.text((68 + dx, base_y + dy), text, font=font, fill=colour)

    # Very light Gaussian blur to soften the too-perfect edges
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    img.save(path, "JPEG", quality=88)
    print(f"  Created: {path}")


# ------------------------------------------------------------------
# 2. handwritten_sample.jpg
#    Cream background. Each text line is rendered on a transparent
#    layer, rotated by a small random angle, then composited —
#    simulating handwritten notes with an uneven baseline.
# ------------------------------------------------------------------
def make_handwritten(path: Path) -> None:
    rng = random.Random(7)
    W, H = 900, 580
    base = Image.new("RGB", (W, H), color=(253, 249, 224))
    base_draw = ImageDraw.Draw(base)

    # Ruled lines
    for y in range(72, H, 54):
        base_draw.line([(48, y), (W - 48, y)], fill=(210, 195, 155), width=1)

    # Left margin line
    base_draw.line([(90, 0), (90, H)], fill=(200, 160, 140), width=2)

    entries = [
        (28,  "Sprint Review Notes",               48,  (55, 30, 10)),
        (26,  "Attendees: Alan, Sarah, Tom",        80,  (50, 32, 12)),
        (24,  "- Deploy by Friday  (Tom)",          130, (42, 26, 10)),
        (24,  "- Send report to client  (Alan)",    178, (42, 26, 10)),
        (23,  "- Fix login bug  (Sarah)",           226, (42, 26, 10)),
        (25,  "Blocker: staging env down",          278, (130, 22, 22)),
        (24,  "Decision: use v2 API",               330, (42, 26, 10)),
        (23,  "Next meeting: Monday 9am",           380, (42, 26, 10)),
        (21,  "Q: confirm budget sign-off?",        428, (60, 50, 10)),
    ]

    for font_size, text, y_pos, colour in entries:
        font = _font(font_size)
        angle = rng.uniform(-2.5, 2.5)
        x_jitter = rng.randint(-4, 4)

        # Render text on a small RGBA patch, rotate it, paste onto base
        bbox_w, bbox_h = 720, font_size + 20
        patch = Image.new("RGBA", (bbox_w, bbox_h), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(patch)
        pdraw.text((4, 4), text, font=font, fill=colour + (255,))

        rotated = patch.rotate(angle, expand=True, resample=Image.BICUBIC)
        paste_x = 100 + x_jitter
        paste_y = y_pos - 4
        base.paste(rotated, (paste_x, paste_y), rotated)

    # Subtle noise to break the digital-clean look
    noise = Image.new("RGB", (W, H))
    npix = noise.load()
    for px in range(W):
        for py in range(H):
            v = rng.randint(-6, 6)
            r, g, b = base.getpixel((px, py))
            npix[px, py] = (
                max(0, min(255, r + v)),
                max(0, min(255, g + v)),
                max(0, min(255, b + v)),
            )
    base = noise

    base.save(path, "JPEG", quality=85)
    print(f"  Created: {path}")


# ------------------------------------------------------------------
# 3. blurry_sample.jpg  — unchanged, works correctly
# ------------------------------------------------------------------
def make_blurry(path: Path) -> None:
    img = Image.new("RGB", (800, 500), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    font = _font(30)
    draw.text((100, 200), "some text here", font=font, fill=(100, 100, 100))
    draw.text((200, 280), "more content", font=font, fill=(120, 120, 120))
    for _ in range(8):
        img = img.filter(ImageFilter.GaussianBlur(radius=6))
    img.save(path, "JPEG", quality=60)
    print(f"  Created: {path}")


if __name__ == "__main__":
    print("Generating test images...")
    make_whiteboard(OUT_DIR / "whiteboard_sample.jpg")
    make_handwritten(OUT_DIR / "handwritten_sample.jpg")
    make_blurry(OUT_DIR / "blurry_sample.jpg")
    print("Done.")
