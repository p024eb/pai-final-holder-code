"""Vision Proの手・頭・ArUcoマーカーを試行ごとにファイルへ保存する収録スクリプト。

    python record_demo.py <VisionProのIP> --session S01            # Local（IP）
    python record_demo.py --fake --session TEST                   # 実機なしの練習

Enterで試行開始、もう一度Enterで終了。終了後に成否（s/f/x）とメモを入力する。
avp_streamは送信側の時刻を渡さないため、新しい受信サンプルを検出した時点の
PC時刻（wall / monotonic）を記録する。同じサンプルを重複して保存しない。
"""

import argparse
import csv
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

N_JOINTS = 27
MARKER_IDS = range(20)


class FakeStreamer:
    """実機なしで手順を練習するための90 Hzの疑似データ。"""

    def __init__(self) -> None:
        self.latest = None
        self._markers = {}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        t0 = time.time()
        while True:
            t = time.time() - t0
            wrist = np.eye(4)[None].copy()
            wrist[0, :3, 3] = [0.3 + 0.05 * np.sin(t), 0.2, 1.0 + 0.03 * np.cos(t)]
            self.latest = {
                "head": np.eye(4)[None],
                "right_wrist": wrist,
                "left_wrist": np.eye(4)[None],
                "right_arm": np.tile(np.eye(4), (N_JOINTS, 1, 1)),
                "left_arm": np.tile(np.eye(4), (N_JOINTS, 1, 1)),
                "right_pinch_distance": 0.05,
                "left_pinch_distance": 0.08,
            }
            pose = np.eye(4)
            pose[:3, 3] = [0.5, 0.0, 0.8]
            self._markers = {0: {"pose": pose, "is_tracked": True, "is_fixed": False}}
            time.sleep(1 / 90)

    def get_markers(self):
        return dict(self._markers)


def open_streamer(target: str | None, fake: bool):
    if fake:
        return FakeStreamer()
    from avp_stream import VisionProStreamer

    return VisionProStreamer(ip=target, record=False)


def as_joints(raw, key: str) -> np.ndarray:
    """手首基準の関節姿勢。旧アプリの25関節は前腕2関節をNaNで埋める。"""
    out = np.full((N_JOINTS, 4, 4), np.nan)
    value = raw.get(key)
    if value is not None:
        arr = np.asarray(value, dtype=np.float64).reshape(-1, 4, 4)
        out[: min(len(arr), N_JOINTS)] = arr[:N_JOINTS]
    return out


def as_pose(raw, key: str) -> np.ndarray:
    value = raw.get(key)
    if value is None:
        return np.full((4, 4), np.nan)
    return np.asarray(value, dtype=np.float64).reshape(4, 4)


class TrialRecorder:
    def __init__(self, streamer) -> None:
        self.s = streamer
        self.rows = []
        self.stop = threading.Event()
        self.max_gap = 0.0

    def run(self) -> None:
        prev = self.s.latest
        last_new = time.perf_counter()
        last_print = 0.0
        while not self.stop.is_set():
            raw = self.s.latest
            now = time.perf_counter()
            if raw is None or raw is prev:
                if now - last_print > 1.0:
                    self._status(now - last_new)
                    last_print = now
                time.sleep(0.001)
                continue
            prev = raw
            self.max_gap = max(self.max_gap, now - last_new) if self.rows else 0.0
            last_new = now
            self.rows.append((time.time(), now, raw, self.s.get_markers()))
            if now - last_print > 1.0:
                self._status(0.0)
                last_print = now

    def _status(self, stall: float) -> None:
        n = len(self.rows)
        if n == 0:
            msg = "受信待ち…"
        else:
            dur = self.rows[-1][1] - self.rows[0][1]
            hz = (n - 1) / dur if dur > 0 else 0.0
            markers = self.rows[-1][3]
            tracked = [i for i, m in markers.items() if m.get("is_tracked")]
            x, y, z = as_pose(self.rows[-1][2], "right_wrist")[:3, 3]
            marker_txt = ",".join(map(str, tracked)) if tracked else "-"
            msg = (f"{dur:5.1f}s {hz:4.0f}Hz 手首({x:+.2f},{y:+.2f},{z:+.2f}) "
                   f"marker:{marker_txt}")
        if stall > 0.5:
            msg += f" ⚠{stall:.1f}s新着なし"
        print("\r" + msg.ljust(56), end="", flush=True)

    def arrays(self) -> dict[str, np.ndarray]:
        n = len(self.rows)
        def stack_pose(key):
            return np.stack([as_pose(r[2], key) for r in self.rows]) if n else np.zeros((0, 4, 4))

        def stack_joints(key):
            return np.stack([as_joints(r[2], key) for r in self.rows]) \
                if n else np.zeros((0, N_JOINTS, 4, 4))

        out = {
            "t_wall": np.array([r[0] for r in self.rows]),
            "t_mono": np.array([r[1] for r in self.rows]),
            "head": stack_pose("head"),
            "right_wrist": stack_pose("right_wrist"),
            "left_wrist": stack_pose("left_wrist"),
            "right_joints_in_wrist": stack_joints("right_arm"),
            "left_joints_in_wrist": stack_joints("left_arm"),
            "right_pinch": np.array([r[2].get("right_pinch_distance", np.nan) for r in self.rows]),
            "left_pinch": np.array([r[2].get("left_pinch_distance", np.nan) for r in self.rows]),
        }
        seen = sorted({i for r in self.rows for i in r[3]})
        for marker_id in seen:
            pose = np.full((n, 4, 4), np.nan)
            tracked = np.zeros(n, dtype=bool)
            fixed = np.zeros(n, dtype=bool)
            for k, r in enumerate(self.rows):
                info = r[3].get(marker_id)
                if info is not None:
                    pose[k] = info["pose"]
                    tracked[k] = bool(info.get("is_tracked"))
                    fixed[k] = bool(info.get("is_fixed"))
            out[f"marker{marker_id}_pose"] = pose
            out[f"marker{marker_id}_tracked"] = tracked
            out[f"marker{marker_id}_fixed"] = fixed
        return out


def lib_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", help="Vision ProのIP（Local）またはルームコード")
    parser.add_argument("--session", required=True, help="例: S01。配置・装着をやり直したら変える")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "raw")
    parser.add_argument("--marker-size-cm", type=float, default=10)
    parser.add_argument("--prediction-offset-ms", type=float, default=0)
    parser.add_argument("--fake", action="store_true")
    args = parser.parse_args()
    if not args.fake and not args.target:
        parser.error("IPまたはルームコードを指定する（練習は --fake）")

    session_dir = args.out / args.session
    session_dir.mkdir(parents=True, exist_ok=True)
    session_meta = session_dir / "session.json"
    if not session_meta.exists():
        session_meta.write_text(json.dumps({
            "session": args.session,
            "created": datetime.now().isoformat(timespec="seconds"),
            "target": "fake" if args.fake else args.target,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "avp_stream": lib_version("avp_stream"),
            "numpy": np.__version__,
            "marker_size_cm": args.marker_size_cm,
            "prediction_offset_ms": args.prediction_offset_ms,
            "axis_convention": "head・wrist・markerはavp_streamの既定（Z上向き、origin=avp）。"
                               "*_joints_in_wristは手首基準の相対姿勢",
        }, ensure_ascii=False, indent=2))

    print("Vision Proへ接続中…")
    streamer = open_streamer(args.target, args.fake)
    t_wait = time.time()
    while streamer.latest is None:
        if time.time() - t_wait > 60:
            sys.exit("60秒待っても手のデータが届かない。IP・アプリ画面・Wi-Fiを確認する")
        time.sleep(0.05)
    print("受信OK")

    log_path = session_dir / "trials.csv"
    new_log = not log_path.exists()
    existing = sorted(session_dir.glob("trial_*.npz"))
    trial_no = len(existing) + 1

    quit_after = False
    while not quit_after:
        try:
            cmd = input(f"\n[試行 {trial_no:03d}] Enter=開始 / q=終了: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break
        if cmd == "q":
            break
        rec = TrialRecorder(streamer)
        th = threading.Thread(target=rec.run, daemon=True)
        start_iso = datetime.now().isoformat(timespec="milliseconds")
        th.start()
        print("● 収録中（Enterで停止）")
        print("  iPad録画開始→机3回→2秒静止→装着→2秒静止→机3回→Enter")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            quit_after = True
        rec.stop.set()
        th.join()
        print()

        if quit_after:
            result, note = "u", "Ctrl+Cで中断（結果未入力）"
        else:
            result = ""
            while result not in ("s", "f", "x"):
                try:
                    result = input("結果 s=成功 / f=失敗 / x=無効 / q=保存して終了: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    result = "q"
                if result == "q":
                    result, quit_after = "u", True
                    break
            try:
                note = "" if quit_after else input("メモ（なければEnter）: ").strip()
            except (KeyboardInterrupt, EOFError):
                note, quit_after = "", True

        arrays = rec.arrays()
        name = f"trial_{trial_no:03d}"
        np.savez_compressed(session_dir / f"{name}.npz", **arrays)
        n = len(arrays["t_mono"])
        dur = float(arrays["t_mono"][-1] - arrays["t_mono"][0]) if n > 1 else 0.0
        info = {
            "trial": name, "session": args.session, "start": start_iso,
            "samples": n, "duration_s": round(dur, 3),
            "mean_hz": round((n - 1) / dur, 1) if dur > 0 else 0.0,
            "max_gap_s": round(rec.max_gap, 3), "result": result, "note": note,
            "markers_seen": sorted(int(k[6:-5]) for k in arrays if k.endswith("_pose")),
        }
        (session_dir / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        with log_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(info))
            if new_log:
                w.writeheader()
                new_log = False
            w.writerow({**info, "markers_seen": " ".join(map(str, info["markers_seen"]))})
        print(f"保存: {session_dir / name}.npz  {n}件 {dur:.1f}秒 "
              f"平均{info['mean_hz']}Hz 最大欠落{info['max_gap_s']}秒 marker={info['markers_seen']}")
        trial_no += 1

    print("終了")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    # avp_streamの受信スレッドが残っても確実に終了させる
    os._exit(0)
