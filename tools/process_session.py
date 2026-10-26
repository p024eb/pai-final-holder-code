"""1セッション分の成功試行をまとめて処理し、一覧表を作る。

    python tools/process_session.py H01 --video-dir ../physicalAI/本番01 [--video]

成功（result=s）の試行ごとに
  1. retarget/human_to_holder.py（差し込み区間の切り出し、受け部基準の軌跡）
  2. retarget/holder_to_panda.py（Panda への変換と MuJoCo での再生・評価）
を実行し、iPad 動画を撮影開始時刻で対応づける。
出力: data/processed/<session>/summary.csv, summary.json
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
JST = timezone(timedelta(hours=9))


def video_starts(video_dir: Path) -> dict[str, datetime]:
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = {}
    for f in sorted(video_dir.glob("*.MOV")) + sorted(video_dir.glob("*.mov")):
        info = subprocess.run([ff, "-i", str(f)], capture_output=True, text=True).stderr
        for line in info.splitlines():
            if "creation_time" in line:
                ts = line.split(":", 1)[1].strip().replace("Z", "+00:00")
                out[f.name] = datetime.fromisoformat(ts)
                break
    return out


def match_video(t_trial: datetime, videos: dict[str, datetime], max_lag_s: float = 30.0):
    """収録開始の後 max_lag_s 秒以内に撮影を始めた動画のうち最も近いもの。"""
    best = None
    for name, tv in videos.items():
        lag = (tv - t_trial).total_seconds()
        if -2.0 <= lag <= max_lag_s and (best is None or lag < best[1]):
            best = (name, lag)
    return best


def run_json(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        return {"error": (p.stderr.strip().splitlines() or ["失敗"])[-1]}
    return json.loads(p.stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--video-dir", type=Path)
    ap.add_argument("--video", action="store_true", help="シミュレーション動画も書き出す（遅い）")
    args = ap.parse_args()

    raw_dir = ROOT / "data" / "raw" / args.session
    with open(raw_dir / "trials.csv", encoding="utf-8") as f:
        trials = [r for r in csv.DictReader(f)]
    videos = video_starts(args.video_dir) if args.video_dir else {}

    rows = []
    for r in trials:
        if r["result"] != "s":
            continue
        trial = r["trial"]
        t_trial = datetime.fromisoformat(r["start"]).replace(tzinfo=JST)
        mv = match_video(t_trial, videos) if videos else None
        h = run_json([sys.executable, "retarget/human_to_holder.py", args.session, trial])
        row = {"trial": trial, "duration_s": r["duration_s"],
               "ipad_video": mv[0] if mv else "", "video_offset_s": round(mv[1], 2) if mv else ""}
        if "error" in h:
            row.update({"stage": "human_to_holder", "error": h["error"]})
            rows.append(row)
            print(trial, "切り出し失敗:", h["error"])
            continue
        row.update({"t_start_s": h["t_start_s"], "t_release_s": h["t_release_s"],
                    "segment_s": h["duration_s"]})
        cmd = [sys.executable, "retarget/holder_to_panda.py", args.session, trial]
        if not args.video:
            cmd.append("--no-video")
        m = run_json(cmd)
        if "error" in m:
            row.update({"stage": "holder_to_panda", "error": m["error"]})
        else:
            load = m.get("grasp_load", {}).get("human", {})
            row.update({"success": m["success"], "final_pos_mm": m["final_error"]["pos_mm"],
                        "final_rot_deg": m["final_error"]["rot_deg"],
                        "optics_contact": m["optics_contact"],
                        "ik_pos_err_mm_max": round(m["ik_pos_err_mm"]["max"], 2),
                        "grasp_force_N_p95": load.get("force_N_p95", ""),
                        "grasp_force_N_max": load.get("force_N_max", "")})
        rows.append(row)
        print(trial, {k: row.get(k) for k in ("segment_s", "success", "final_pos_mm",
                                               "grasp_force_N_max", "ipad_video", "error")})

    out_dir = ROOT / "data" / "processed" / args.session
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row}, key=lambda k: list(rows[0]).index(k)
                  if rows and k in rows[0] else 99)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n_ok = sum(1 for row in rows if row.get("success") is True)
    summary = {"session": args.session, "n_success_trials": len(rows), "n_sim_success": n_ok,
               "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n成功試行 {len(rows)} 本のうち、シミュレーションで着座 {n_ok} 本")


if __name__ == "__main__":
    main()
