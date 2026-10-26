"""iPad の実演動画と、同じ試行の Panda での再生（MuJoCo）を左右に並べた動画を作る。

    python tools/compare_video.py H01 trial_013 --video-dir ../physicalAI/本番01

差し込み区間の開始 pre 秒前から、解放の post 秒後までを切り出す。
シミュレーション動画（data/sim/<s>/<t>_side_right.mp4）がなければ holder_to_panda.py で作る。
出力: data/sim/<session>/<trial>_compare.mp4
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from privacy import LabelBlur  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
SIM_HUMAN_START_S = 3.5  # holder_to_panda.py の settle 0.5 s ＋ approach 3 s
FPS = 30


def label(frame: np.ndarray, text: str, size: int = 28) -> np.ndarray:
    im = Image.fromarray(frame)
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, size)
    x0, y0, x1, y1 = d.textbbox((12, 10), text, font=font)
    d.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0))
    d.text((12, 10), text, font=font, fill=(255, 255, 255))
    return np.asarray(im)


def read_clip(path: Path, start_s: float, n: int, height: int) -> list[np.ndarray]:
    r = imageio.get_reader(path, "ffmpeg", input_params=["-ss", f"{max(start_s, 0):.3f}"],
                           output_params=["-r", str(FPS), "-vf", f"scale=-2:{height}"])
    out = []
    for i, f in enumerate(r):
        if i >= n:
            break
        out.append(f)
    r.close()
    while len(out) < n:
        out.append(out[-1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("trial")
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--pre", type=float, default=1.0)
    ap.add_argument("--post", type=float, default=3.0)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config" / "sessions.json").read_text())["use"][args.session]
    video = args.video_dir / cfg["ipad_video"][args.trial]
    offset = cfg["video_offset_s"][args.trial]
    seg = json.loads((ROOT / "data" / "processed" / args.session / f"{args.trial}_holder.json").read_text())
    sim_dir = ROOT / "data" / "sim" / args.session
    sim_video = sim_dir / f"{args.trial}_side_right.mp4"
    if not sim_video.exists():
        subprocess.run([sys.executable, str(ROOT / "retarget" / "holder_to_panda.py"),
                        args.session, args.trial], check=True, capture_output=True, cwd=ROOT)

    n = int((args.pre + seg["duration_s"] + args.post) * FPS)
    blur = LabelBlur(args.video_dir)
    human = [blur(f) for f in read_clip(video, seg["t_start_s"] - offset - args.pre, n, args.height)]
    sim = read_clip(sim_video, SIM_HUMAN_START_S - args.pre, n, args.height)
    out = sim_dir / f"{args.trial}_compare.mp4"
    w = imageio.get_writer(out, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    for h, s in zip(human, sim):
        w.append_data(np.concatenate([label(h, "人の実演（iPad）"), label(s, "Panda で再生（MuJoCo）")], axis=1))
    w.close()
    print(out)


if __name__ == "__main__":
    main()
