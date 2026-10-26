"""収録セッションの品質確認：受信間隔、マーカー追跡率、手首軌跡、机叩きの検出を図にする。

    python inspect_session.py data/raw/DRY01
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def unique_samples(t: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """直前と同じ手首姿勢の再送を除いたインデックス。"""
    changed = np.ones(len(t), dtype=bool)
    changed[1:] = np.abs(np.diff(wrist, axis=0)).sum(axis=1) > 0
    return np.where(changed)[0]


def frozen_intervals(t: np.ndarray, idx: np.ndarray, min_s: float = 0.3) -> list[tuple[float, float]]:
    """手首が更新されない区間（手が視界外・追跡喪失の候補）。"""
    tu = t[idx]
    gaps = np.where(np.diff(tu) > min_s)[0]
    return [(float(tu[g]), float(tu[g + 1])) for g in gaps]


def detect_sync_bursts(t: np.ndarray, z: np.ndarray, idx: np.ndarray) -> list[dict]:
    """持ち上げ→下ろしの山が3回以上連続する区間を同期合図として返す。"""
    tu, zu = t[idx], z[idx]
    grid = np.arange(tu[0], tu[-1], 0.01)
    zg = np.interp(grid, tu, zu)
    zs = np.convolve(zg, np.ones(5) / 5, mode="same")
    peaks, props = find_peaks(zs, prominence=0.03, distance=25, width=10)
    pt = grid[peaks]
    bursts, cur = [], [0] if len(pt) else []
    for k in range(1, len(pt)):
        if pt[k] - pt[cur[-1]] <= 2.5:
            cur.append(k)
        else:
            bursts.append(cur)
            cur = [k]
    if cur:
        bursts.append(cur)
    out = []
    for b in bursts:
        if len(b) >= 3:
            out.append({"start_s": round(float(pt[b[0]]), 2), "end_s": round(float(pt[b[-1]]), 2),
                        "n_peaks": len(b),
                        "lift_cm": round(float(np.median(props["prominences"][b]) * 100), 1)})
    return out


def inspect_trial(npz_path: Path, out_dir: Path) -> dict:
    d = np.load(npz_path)
    t = d["t_mono"] - d["t_mono"][0]
    wrist = d["right_wrist"][:, :3, 3]
    dt = np.diff(t)
    idx = unique_samples(t, wrist)
    frozen = frozen_intervals(t, idx)
    info = {
        "trial": npz_path.stem,
        "samples": int(len(t)),
        "unique_samples": int(len(idx)),
        "unique_hz": round(float(len(idx) / t[-1]), 1),
        "duration_s": round(float(t[-1]), 2),
        "gaps_over_50ms": int((dt > 0.05).sum()),
        "wrist_nan_ratio": round(float(np.isnan(wrist).any(axis=1).mean()), 3),
        "wrist_frozen_s": round(sum(b - a for a, b in frozen), 1),
        "wrist_frozen_intervals": [(round(a, 1), round(b, 1)) for a, b in frozen if b - a > 1.0],
    }

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for k, name in enumerate("xyz"):
        axes[0].plot(t, wrist[:, k], label=f"wrist {name}")
    axes[0].set_ylabel("right wrist [m]\nred=sync, gray=frozen")
    axes[0].legend(loc="upper right")

    bursts = detect_sync_bursts(t, wrist[:, 2], idx)
    for b in bursts:
        axes[0].axvspan(b["start_s"], b["end_s"], color="tab:red", alpha=0.15)
    for a, b in frozen:
        axes[0].axvspan(a, b, color="gray", alpha=0.25)
    info["sync_bursts"] = bursts

    marker_keys = sorted(k for k in d.files if k.endswith("_tracked"))
    for key in marker_keys:
        mid = key[len("marker"):-len("_tracked")]
        tracked = d[key]
        pose = d[f"marker{mid}_pose"]
        info[f"marker{mid}_tracked_ratio"] = round(float(tracked.mean()), 3)
        info[f"marker{mid}_fixed_ratio"] = round(float(d[f"marker{mid}_fixed"].mean()), 3)
        axes[1].plot(t, tracked.astype(float) + 0.02 * int(mid), label=f"marker {mid} tracked")
        p = pose[:, :3, 3]
        valid = tracked & ~np.isnan(p).any(axis=1)
        if valid.any():
            ref = np.nanmedian(p[valid], axis=0)
            dev = np.linalg.norm(p - ref, axis=1) * 1000
            dev[~valid] = np.nan
            axes[2].plot(t, dev, label=f"marker {mid}")
            info[f"marker{mid}_jitter_mm_p95"] = round(float(np.nanpercentile(dev, 95)), 1)
            info[f"marker{mid}_median_xyz_m"] = [round(float(v), 4) for v in ref]
    axes[1].set_ylabel("tracked")
    axes[1].legend(loc="upper right")
    axes[2].set_ylabel("marker deviation\nfrom median [mm]")
    axes[2].legend(loc="upper right")

    axes[3].plot(t[1:], dt * 1000, lw=0.6)
    axes[3].set_ylabel("receive interval [ms]")
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylim(0, max(60, float(np.percentile(dt * 1000, 99.9)) * 1.1))
    fig.suptitle(npz_path.parent.name + " / " + npz_path.stem)
    fig.tight_layout()
    png = out_dir / f"{npz_path.stem}.png"
    fig.savefig(png, dpi=90)
    plt.close(fig)
    info["figure"] = str(png)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    out_dir = args.session_dir.parent.parent / "inspect" / args.session_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for npz in sorted(args.session_dir.glob("trial_*.npz")):
        print(json.dumps(inspect_trial(npz, out_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
