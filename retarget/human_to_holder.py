"""人の装着デモ（Vision Proの手首姿勢）から、受け部基準のホルダー軌跡 T_FO(t) を作る。

    python retarget/human_to_holder.py p03 trial_001

仮定（metadataにも保存する）:
- 把持中（区間内）はホルダーが手首と剛体的に動く。T_O(t) = T_W(t) T_W(t_rel)^-1 T_O,seat
- 解放時刻 t_rel にホルダーは着座しており、受け部座標系 F の原点・姿勢と一致する。
  これは終端を使ったオフライン登録で、学習時の入力には使わない。
- 装置の奥行方向は、差し込み区間の手首の水平移動方向。上方向は世界Z。
- 指先のつまみ点はホルダー手前辺の中央から grip_inset だけ内側。
座標: F は x=差し込み方向（奥）、y=左、z=上。単位 m。
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from inspect_session import detect_sync_bursts  # noqa: E402

THUMB_TIP, INDEX_TIP = 4, 9


def load_dims() -> dict:
    return json.loads((ROOT / "config" / "receiver_dimensions.json").read_text())


def unique_idx(wrist: np.ndarray) -> np.ndarray:
    keep = np.ones(len(wrist), dtype=bool)
    keep[1:] = np.abs(np.diff(wrist.reshape(len(wrist), -1), axis=0)).sum(axis=1) > 0
    return np.where(keep)[0]


def resample_poses(t: np.ndarray, T: np.ndarray, grid: np.ndarray) -> np.ndarray:
    slerp = Slerp(t, Rotation.from_matrix(T[:, :3, :3]))
    out = np.tile(np.eye(4), (len(grid), 1, 1))
    out[:, :3, :3] = slerp(grid).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(grid, t, T[:, k, 3])
    return out


def lowpass_poses(T: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    b, a = butter(2, cutoff / (fs / 2))
    out = T.copy()
    out[:, :3, 3] = filtfilt(b, a, T[:, :3, 3], axis=0)
    r_ref = Rotation.from_matrix(T[len(T) // 2, :3, :3])
    rv = (r_ref.inv() * Rotation.from_matrix(T[:, :3, :3])).as_rotvec()
    rv = filtfilt(b, a, rv, axis=0)
    out[:, :3, :3] = (r_ref * Rotation.from_rotvec(rv)).as_matrix()
    return out


def first_sustained(t, cond, t_from, hold_s):
    idx = np.where(t >= t_from)[0]
    run = 0
    dt = np.median(np.diff(t))
    for i in idx:
        run = run + 1 if cond[i] else 0
        if run * dt >= hold_s:
            return t[i - run + 1]
    return None


def runs(mask):
    """True が続く区間の (開始, 終了の次) の一覧。"""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def fill_short_gaps(mask, max_gap):
    out = mask.copy()
    for s, e in runs(~mask):
        if 0 < s and e < len(mask) and e - s < max_gap:
            out[s:e] = True
    return out


def marker_center(raw):
    """固定マーカー（受け部の近くに貼ったもの）の位置。見えていなければ None。"""
    for k in sorted(raw.files):
        if k.startswith("marker") and k.endswith("_pose"):
            tracked = raw[k.replace("_pose", "_tracked")].astype(bool)
            if tracked.any():
                return np.median(raw[k][tracked, :3, 3], axis=0)
    return None


def detect_release(grid, pinch_pt, holding, grip, in_win, fs, marker, near_xy=0.35, still_v=0.05):
    """受け部付近でつまんだまま静止した最後の区間の後、指が開くか手が動き出した時刻。

    スライド配置での指の開閉や、解放後に力を抜いた指が閉じる動きを拾わないよう、
    静止・把持・受け部付近（マーカーから水平 near_xy 以内かつマーカーより 0.1 m 以上下でない）を同時に要求する。
    """
    v = np.linalg.norm(np.gradient(pinch_pt, 1 / fs, axis=0), axis=1)
    v = np.convolve(v, np.ones(int(0.5 * fs)) / int(0.5 * fs), mode="same")
    still = v < still_v
    near = np.ones(len(grid), bool)
    if marker is not None:
        near = (np.linalg.norm(pinch_pt[:, :2] - marker[:2], axis=1) < near_xy) & \
               (pinch_pt[:, 2] > marker[2] - 0.1)
    cand = [(s, e) for s, e in runs(still & grip & in_win & near) if (e - s) / fs >= 0.8]
    if not cand:
        return None
    s, e = cand[-1]
    lim = min(len(grid), e + int(2.0 * fs))
    opened = [i for i in range(s, lim - int(0.3 * fs)) if (~holding[i:i + int(0.3 * fs)]).all()]
    moving = [i for i in range(e, lim) if v[i] > 0.15]
    ends = [x[0] for x in (opened, moving) if x]
    return grid[min(ends) if ends else e] - 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("trial")
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--cutoff-hz", type=float, default=4.0)
    parser.add_argument("--grip-inset-m", type=float, default=0.010)
    parser.add_argument("--hold-pinch-m", type=float, default=0.019,
                        help="これ未満のつまみ距離を把持中とみなす")
    parser.add_argument("--loose-pinch-m", type=float, default=0.035,
                        help="解放が見つからないとき、把持中とみなす距離をここまでゆるめて探し直す")
    parser.add_argument("--approach-m", type=float, default=0.30,
                        help="着座点からこの距離以内に入ってからを差し込み区間とする")
    parser.add_argument("--lateral-m", type=float, default=0.04,
                        help="つまみ点の左右ずれがこれ以内に揃ってからを差し込み区間とする")
    parser.add_argument("--start", type=float, help="区間開始[s]を手動指定")
    parser.add_argument("--release", type=float, help="解放時刻[s]を手動指定")
    args = parser.parse_args()

    dims = load_dims()
    holder_len = dims["holder_outer"]["width"] / 1000  # 127 mm: 差し込み方向
    raw = np.load(ROOT / "data" / "raw" / args.session / f"{args.trial}.npz")
    t_all = raw["t_mono"] - raw["t_mono"][0]
    keep = unique_idx(raw["right_wrist"])
    t_u = t_all[keep]
    W_u = raw["right_wrist"][keep]
    J_u = raw["right_joints_in_wrist"][keep]

    grid = np.arange(t_u[0], t_u[-1], 1 / args.fs)
    W = lowpass_poses(resample_poses(t_u, W_u, grid), args.fs, args.cutoff_hz)
    tips = []
    for k in (THUMB_TIP, INDEX_TIP):
        Tk = W_u @ J_u[:, k]
        tips.append(np.stack([np.interp(grid, t_u, Tk[:, j, 3]) for j in range(3)], axis=1))
    b, a = butter(2, args.cutoff_hz / (args.fs / 2))
    thumb, index = (filtfilt(b, a, p, axis=0) for p in tips)
    pinch_pt = (thumb + index) / 2
    pinch = np.linalg.norm(thumb - index, axis=1)

    holding = pinch < args.hold_pinch_m
    speed = np.linalg.norm(np.gradient(W[:, :3, 3], 1 / args.fs, axis=0), axis=1)
    bursts = detect_sync_bursts(t_all, raw["right_wrist"][:, 2, 3], unique_idx(raw["right_wrist"]))
    if len(bursts) < 2:
        raise SystemExit(f"開始・終了の同期合図が2つ見つからない: {bursts}")
    # 作業は最初の2回の合図の間（終了の合図をやり直すと3つ目が入る）
    win_from, win_to = bursts[0]["end_s"] + 0.3, bursts[1]["start_s"] - 0.3
    # 0.3秒未満の短い開き（追跡のゆらぎ）は把持中とみなす
    grip = fill_short_gaps(holding, int(0.3 * args.fs))
    in_win = (grid > win_from) & (grid < win_to)
    if args.release is not None:
        t_rel = args.release
    else:
        t_rel = detect_release(grid, pinch_pt, holding, grip, in_win, args.fs, marker_center(raw))
        if t_rel is None:
            # 着座後に指をゆるめて押さえる試行では、つまみ距離が 2〜3 cm になる
            loose = pinch < args.loose_pinch_m
            t_rel = detect_release(grid, pinch_pt, loose, fill_short_gaps(loose, int(0.3 * args.fs)),
                                   in_win, args.fs, marker_center(raw), still_v=0.08)
        if t_rel is None:
            raise SystemExit("合図の間に、受け部付近でつまんだまま静止→解放する区間が見つからない")
    if not win_from < t_rel < win_to:
        raise SystemExit(f"解放時刻 {t_rel:.2f}s が合図の間 {win_from:.2f}-{win_to:.2f}s にない")
    i_rel = int(np.searchsorted(grid, t_rel))
    if args.start is not None:
        t_start = args.start
    else:
        # つまみ点が着座点から approach_m より遠かった最後の時刻（運搬の途中）から
        dist = np.linalg.norm(pinch_pt - pinch_pt[i_rel], axis=1)
        far = np.where(in_win & (np.arange(len(grid)) < i_rel) & (dist > args.approach_m))[0]
        if not len(far):
            raise SystemExit("解放前に着座点から離れていた時刻がない（区間の開始を決められない）")
        t_start = grid[far[-1] + 1]
    i0 = int(np.searchsorted(grid, t_start))

    def seat_frame(i_from):
        # 装置の奥行方向：着座点の手前 0.15 m 以内に入ってから解放までの手首の水平移動
        # （それ以前は作業台から開口部の前まで運ぶ横移動を含む）
        d_seat = np.linalg.norm(pinch_pt[:i_rel] - pinch_pt[i_rel], axis=1)
        i_dir = max(i_from, int(np.where(d_seat > 0.15)[0][-1]) + 1 if (d_seat > 0.15).any() else i_from)
        disp = W[i_rel, :3, 3] - W[i_dir, :3, 3]
        ex = np.array([disp[0], disp[1], 0.0])
        ex /= np.linalg.norm(ex)
        ez = np.array([0.0, 0.0, 1.0])
        O = np.eye(4)
        O[:3, :3] = np.stack([ex, np.cross(ez, ex), ez], axis=1)
        O[:3, 3] = pinch_pt[i_rel] + ex * (holder_len / 2 - args.grip_inset_m)
        return O

    O_seat = seat_frame(i0)
    if args.start is None:
        # 開口部の正面に揃った（つまみ点の左右ずれが lateral_m 以内になった）時点から。
        # それ以前の作業台からの運搬は装置の側面形状が未計測のため、ロボット側の接近動作で置き換える
        Oi = np.linalg.inv(O_seat)
        y_F = pinch_pt @ Oi[1, :3] + Oi[1, 3]
        off = np.where((np.arange(len(grid)) >= i0) & (np.arange(len(grid)) < i_rel)
                       & (np.abs(y_F) > args.lateral_m))[0]
        if len(off):
            i0 = int(off[-1]) + 1
            O_seat = seat_frame(i0)
        # 持ち直し（つまみが開いている）が直後にあれば、その後から
        t_grip = first_sustained(grid, holding, grid[i0], 0.2)
        if t_grip is not None and t_grip - grid[i0] < 0.6:
            i0 = int(np.searchsorted(grid, t_grip))
        t_start = grid[i0]

    seg = slice(i0, i_rel + 1)
    W_rel_inv = np.linalg.inv(W[i_rel])
    O = W[seg] @ W_rel_inv @ O_seat
    T_FO = np.linalg.inv(O_seat) @ O
    t_seg = grid[seg]
    rpy = Rotation.from_matrix(T_FO[:, :3, :3]).as_euler("xyz", degrees=True)

    out_dir = ROOT / "data" / "processed" / args.session
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{args.trial}_holder"
    meta = {
        "session": args.session, "trial": args.trial,
        "t_start_s": round(float(t_start), 3), "t_release_s": round(float(t_rel), 3),
        "duration_s": round(float(t_seg[-1] - t_seg[0]), 3),
        "fs_hz": args.fs, "lowpass_hz": args.cutoff_hz, "grip_inset_m": args.grip_inset_m,
        "hold_pinch_m": args.hold_pinch_m,
        "insertion_dir_world": O_seat[:3, 0].round(4).tolist(),
        "seat_pose_world": O_seat.round(5).tolist(),
        "pinch_during_segment_cm": [round(float(pinch[seg].min() * 100), 2),
                                     round(float(pinch[seg].max() * 100), 2)],
        "start_offset_in_F_m": T_FO[0, :3, 3].round(4).tolist(),
        "travel_in_F_m": {"x": round(float(np.ptp(T_FO[:, 0, 3])), 4),
                          "y": round(float(np.ptp(T_FO[:, 1, 3])), 4),
                          "z": round(float(np.ptp(T_FO[:, 2, 3])), 4)},
        "angle_range_deg": {k: [round(float(rpy[:, j].min()), 1), round(float(rpy[:, j].max()), 1)]
                            for j, k in enumerate(["roll_x", "pitch_y", "yaw_z"])},
        "assumptions": __doc__.split("仮定（metadataにも保存する）:")[1].split("座標:")[0].strip(),
    }
    O_seat_inv = np.linalg.inv(O_seat)
    pinch_pt_F = pinch_pt[seg] @ O_seat_inv[:3, :3].T + O_seat_inv[:3, 3]
    np.savez_compressed(f"{stem}.npz", t_src=t_seg, T_FO=T_FO, pinch=pinch[seg],
                        pinch_pt_F=pinch_pt_F)
    Path(f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ts = t_seg - t_seg[0]
    for j, name in enumerate(["x (insertion)", "y (left)", "z (up)"]):
        axes[0].plot(ts, T_FO[:, j, 3] * 1000, label=name)
    axes[0].set_ylabel("holder in F [mm]")
    axes[0].legend()
    for j, name in enumerate(["roll about x", "pitch about y", "yaw about z"]):
        axes[1].plot(ts, rpy[:, j], label=name)
    axes[1].set_ylabel("holder angle [deg]")
    axes[1].legend()
    axes[2].plot(ts, pinch[seg] * 100)
    axes[2].axhline(args.hold_pinch_m * 100, color="k", ls="--", lw=0.8)
    axes[2].set_ylabel("pinch [cm]")
    axes[2].set_xlabel(f"time from {t_start:.2f} s [s]")
    fig.suptitle(f"{args.session}/{args.trial}: holder relative to receiver (seat = 0)")
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=90)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("seat_pose_world", "assumptions")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
