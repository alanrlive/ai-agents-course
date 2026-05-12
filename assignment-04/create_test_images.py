"""
Generates the three sample test images used by test_cases.py.
Run once: python create_test_images.py
Requires: Pillow  (pip install Pillow)
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = Path("test_images")
OUT_DIR.mkdir(exist_ok=True)


def _default_font(size: int) -> ImageFont.ImageFont:
    """Return a truetype font if available, otherwise the built-in bitmap font."""
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


# ------------------------------------------------------------------
# 1. whiteboard_sample.jpg
# ------------------------------------------------------------------
def make_whiteboard(path: Path) -> None:
    img = Image.new("RGB", (800, 500), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    title_font = _default_font(32)
    body_font = _default_font(24)

    lines = [
        ("Meeting Notes - Sprint Review", 40, title_font, (20, 20, 20)),
        ("", 90, body_font, (40, 40, 40)),
        ("Attendees: Alan, Sarah, Tom", 110, body_font, (40, 40, 40)),
        ("", 150, body_font, (40, 40, 40)),
        ("Action: Deploy by Friday  -  Owner: Tom", 170, body_font, (30, 30, 30)),
        ("Action: Send report to client  -  Owner: Alan", 210, body_font, (30, 30, 30)),
        ("", 250, body_font, (40, 40, 40)),
        ("Next meeting: Monday 9am", 270, body_font, (40, 40, 40)),
    ]

    # faint grid lines to mimic a whiteboard surface
    for y in range(0, 500, 40):
        draw.line([(0, y), (800, y)], fill=(220, 220, 220), width=1)

    for text, y, font, colour in lines:
        if text:
            draw.text((60, y), text, font=font, fill=colour)

    img.save(path, "JPEG", quality=90)
    print(f"  Created: {path}")


# ------------------------------------------------------------------
# 2. handwritten_sample.jpg
# ------------------------------------------------------------------
def make_handwritten(path: Path) -> None:
    img = Image.new("RGB", (800, 520), color=(255, 252, 235))
    draw = ImageDraw.Draw(img)

    font = _default_font(26)

    # Slightly offset lines to simulate uneven handwriting
    entries = [
        (58,  42,  "Sprint Review Notes",              (45, 30, 10)),
        (62,  88,  "Attendees: Alan, Sarah, Tom",       (50, 35, 15)),
        (55, 138,  "- Deploy by Friday  (Tom)",         (40, 25, 10)),
        (60, 182,  "- Send report to client  (Alan)",   (40, 25, 10)),
        (57, 228,  "- Fix login bug  (Sarah)",          (40, 25, 10)),
        (63, 278,  "Blocker: staging env down",         (80, 20, 20)),
        (56, 328,  "Decision: use v2 API",              (40, 25, 10)),
        (61, 378,  "Next meeting: Monday 9am",          (40, 25, 10)),
    ]

    # ruled lines
    for y in range(70, 520, 50):
        draw.line([(40, y), (760, y)], fill=(200, 185, 150), width=1)

    for x, y, text, colour in entries:
        draw.text((x, y), text, font=font, fill=colour)

    img.save(path, "JPEG", quality=88)
    print(f"  Created: {path}")


# ------------------------------------------------------------------
# 3. blurry_sample.jpg  — heavily blurred, no readable text
# ------------------------------------------------------------------
def make_blurry(path: Path) -> None:
    img = Image.new("RGB", (800, 500), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)

    font = _default_font(30)
    # Write some text that will be completely obliterated by blur
    draw.text((100, 200), "some text here", font=font, fill=(100, 100, 100))
    draw.text((200, 280), "more content", font=font, fill=(120, 120, 120))

    # Apply heavy Gaussian blur multiple times to destroy all legibility
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
