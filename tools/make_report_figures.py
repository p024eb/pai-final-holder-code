"""レポート用の図を docs/figures/ に作る。

    MUJOCO_GL=egl python tools/make_report_figures.py --ipad-dir ../physicalAI/本番01 --run act_v7
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from privacy import LabelBlur  # noqa: E402
from trajectory_panel import TrajectoryPanel  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
OUT = ROOT / "docs" / "figures"


def read_frame(path: Path, t: float) -> np.ndarray:
    r = imageio.get_reader(path, "ffmpeg", input_params=["-ss", f"{max(t, 0):.3f}"])
    f = r.get_next_data()
    r.close()
    return f


def fig_pipeline() -> None:
    steps = [("収録", "Vision Pro\n手首・指の関節\n約 150 Hz"),
             ("軌跡への変換", "つまみ・着座・\n解放を検出\n受け部基準の軌跡"),
             ("ロボットで再生", "逆運動学（mink）\nMuJoCo の Panda\nで着座を確認"),
             ("データの水増し", "装置・開始位置\n速さ・揺らぎ\n683 エピソード"),
             ("模倣学習", "ACT\n画像2枚＋関節角\n→ 1 秒分の行動")]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, len(steps) * 2)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    for i, (h, b) in enumerate(steps):
        x = i * 2 + 0.05
        ax.add_patch(FancyBboxPatch((x, 0.05), 1.75, 2.5, boxstyle="round,pad=0.02", fc="#e8f0fb", ec="#2b5fa8"))
        ax.text(x + 0.875, 2.2, h, ha="center", va="center", fontsize=12.5, weight="bold")
        ax.text(x + 0.875, 1.05, b, ha="center", va="center", fontsize=11, linespacing=1.7)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.0, 1.3), xytext=(x + 1.8, 1.3),
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#2b5fa8"))
    fig.savefig(OUT / "fig1_pipeline.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_scene(ipad_dir: Path, session: str, trial: str) -> None:
    cfg = json.loads((ROOT / "config" / "sessions.json").read_text())["use"][session]
    seg = json.loads((ROOT / "data" / "processed" / session / f"{trial}_holder.json").read_text())
    t = seg["t_start_s"] + seg["duration_s"] * 0.7 - cfg["video_offset_s"][trial]
    human = LabelBlur(ipad_dir)(read_frame(ipad_dir / cfg["ipad_video"][trial], t))
    sim = read_frame(ROOT / "data" / "sim" / session / f"{trial}_side_right.mp4", 3.5 + seg["duration_s"] * 0.7)
    over = read_frame(ROOT / "data" / "sim" / session / f"{trial}_overview.mp4", 3.5 + seg["duration_s"] * 0.7)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), gridspec_kw={"width_ratios": [0.6, 1.78, 1.78]})
    for a, im, title in zip(ax, (human, sim, over),
                            ("(a) 人の実演（iPad）", "(b) Panda での再生（横から）", "(c) シーン全体")):
        a.imshow(im)
        a.set_title(title, fontsize=10)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_scene.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_trajectory(session: str, trial: str) -> None:
    p = TrajectoryPanel(session, trial, width=900, height=680)
    p.fig.savefig(OUT / "fig3_trajectory.png", dpi=200)


def fig_observations(data: str) -> None:
    files = sorted((ROOT / "data" / data).glob("episode_*.hdf5"))
    with h5py.File(files[0]) as f:
        n = len(f["action"])
        idx = [0, n // 3, int(n * 0.55)]
        rows = [np.concatenate([f["observations/images/front_top"][i], f["observations/images/wrist"][i]], 1)
                for i in idx]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.2))
    for a, im, lab in zip(ax, rows, ("開始", "差し込み中", "着座の直前")):
        a.imshow(im)
        a.set_title(f"{lab}（左: 手前上方のカメラ、右: 手先カメラ）", fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_observations.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_progress() -> None:
    vers = [("v1", 0.23, "側面カメラ"), ("v2", 0.07, "1秒保持"), ("v3", 0.54, "静止区間を除く"),
            ("v4", 0.66, "揺らぎ(DART)"), ("v5", 0.72, "解放ラベルを状態で"), ("v7", 0.80, "IK の枝をそろえる")]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    xs = np.arange(len(vers))
    bars = ax.bar(xs, [v[1] * 100 for v in vers], color=["#9db7dc"] * 5 + ["#2b5fa8"])
    for b, v in zip(bars, vers):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v[1] * 100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(xs, [f"{v[0]}\n{v[2]}" for v in vers], fontsize=8)
    ax.set_ylabel("成功率 [%]")
    ax.set_ylim(0, 100)
    ax.set_title("ACT の改良と成功率（v1, v2 は 30 回、v3 以降は同じ 50 回で評価）", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_progress.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_rollout(run: str, tag: str, ep: int) -> None:
    path = ROOT / "runs" / run / "videos" / f"ep_{ep:02d}{tag}.mp4"
    ev = {e["episode"]: e for e in json.loads((ROOT / "runs" / run / f"eval{tag}.json").read_text())["episodes"]}
    rel = ev[ep]["release_s"]
    ts = [0.2, rel * 0.4, rel * 0.75, rel + 0.6]
    fig, ax = plt.subplots(1, 4, figsize=(10, 1.9))
    for a, t, lab in zip(ax, ts, ("開始", "運搬", "差し込み", "解放後")):
        im = read_frame(path, t)
        a.imshow(im[:, : im.shape[1] // 2])
        a.set_title(f"{lab}（{t:.1f} s）", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_rollout.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipad-dir", type=Path, required=True)
    ap.add_argument("--run", default="act_v7")
    ap.add_argument("--data", default="act_v7")
    ap.add_argument("--session", default="H01")
    ap.add_argument("--trial", default="trial_013")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_scene(args.ipad_dir, args.session, args.trial)
    fig_trajectory(args.session, args.trial)
    fig_observations(args.data)
    fig_progress()
    fig_rollout(args.run, "_n50", 0)
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
