# Usage: uv run python make_video_from_png.py --input-dir ~/results/npz_save_CoffeeServeMug
# to be used with PNG generated from interactive_rollout.py
import argparse
import glob
import os
import re
import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True, help="Directory containing *_step*.png files")
parser.add_argument("--output", default=None, help="Output video path (default: <input-dir>/rollout.mp4)")
parser.add_argument("--fps", type=int, default=10)
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_PATH = args.output or f"{INPUT_DIR}/rollout.mp4"
FPS = args.fps

pngs = sorted(glob.glob(f"{INPUT_DIR}/*_step*.png"))
pattern = re.compile(r"(.+)_step(\d+)\.png$")
frames = []
for p in pngs:
    m = pattern.search(os.path.basename(p))
    if m:
        frames.append((m.group(1), int(m.group(2)), p))
frames.sort(key=lambda x: (x[0], x[1]))

if not frames:
    raise SystemExit("No matching PNGs found")

sample = Image.open(frames[0][2])
h, w = sample.size[1], sample.size[0]

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except OSError:
    font = ImageFont.load_default()

rendered = []
for prefix, step, path in frames:
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    label = str(step)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = 8, 8
    draw.rectangle([x - 2, y - 2, x + tw + 4, y + th + 4], fill=(0, 0, 0))
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    rendered.append(np.array(img))

iio.imwrite(
    OUTPUT_PATH,
    rendered,
    fps=FPS,
    codec="libx264",
    plugin="pyav",
)
print(f"Wrote {len(rendered)} frames to {OUTPUT_PATH}")
