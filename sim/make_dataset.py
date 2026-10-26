"""人の差し込み軌跡を水増しして MuJoCo で再生し、ACT の学習データ（hdf5）を作る。

    MUJOCO_GL=egl python sim/make_dataset.py --per-demo 10 --workers 40

元の軌跡: config/sessions.json の use（human_to_holder.py の出力がある試行）。
1 エピソードごとに次をランダムに変える。
  装置の位置（前後・左右 ±receiver_m）、ホルダー開始位置のずらし（±start_m、着座点に近づくほど小さく 0 へ）、
  再生の速さ（time_scale）。
着座に成功し、IK の誤差が小さいエピソードだけを保存する。
出力: data/<out>/episode_XXXX.hdf5, data/<out>/index.json
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "retarget"))
from env import CAMERAS, HolderEnv  # noqa: E402
from holder_to_panda import IK, make_targets, plan_phases  # noqa: E402

OUT = ROOT / "data" / "act"
# 逆運動学で着座姿勢から離れすぎないための重み。小さいと冗長な自由度に流され、q1・q5 が数 rad 回る
POSTURE_COST = 1e-3


def base_demos() -> list[tuple[str, str]]:
    cfg = json.loads((ROOT / "config" / "sessions.json").read_text())
    out = []
    for session, v in cfg["use"].items():
        for trial in v["trials"]:
            if (ROOT / "data" / "processed" / session / f"{trial}_holder.npz").exists():
                out.append((session, trial))
    return out


def sample_params(rng, receiver_m, start_m) -> dict:
    return {"receiver_dx": float(rng.uniform(-receiver_m, receiver_m)),
            "receiver_dy": float(rng.uniform(-receiver_m, receiver_m)),
            "start_offset": [float(rng.uniform(-start_m, start_m)),
                             float(rng.uniform(-start_m, start_m)),
                             float(rng.uniform(-start_m / 2, start_m / 2))],
            "time_scale": float(rng.uniform(0.85, 1.15))}


_SEAT_REF = None
SEAT_REF_PATH = ROOT / "config" / "seat_reference.json"


def seat_reference() -> np.ndarray:
    """装置が標準の位置にあるときの着座姿勢（全エピソードで同じ解の枝を使うための基準）。"""
    global _SEAT_REF
    if _SEAT_REF is None:
        if not SEAT_REF_PATH.exists():
            raise SystemExit("config/seat_reference.json がない。python sim/make_dataset.py --find-seat-reference で作る")
        _SEAT_REF = np.array(json.loads(SEAT_REF_PATH.read_text())["q"])
    return _SEAT_REF


def find_seat_reference(n_seeds: int = 48) -> None:
    """着座姿勢の候補を多数の初期値から求め、関節の動きが最も小さい候補を基準にする。

    候補ごとに、長い元の軌跡を装置の位置を前後左右に振った配置で逆向きに解き、
    接近〜差し込みの間の関節の動いた幅の最大を比べる。
    """
    global _SEAT_REF
    env = HolderEnv(cameras=())
    T_seat = env.geom.F_world @ np.linalg.inv(env.T_hO)
    ik = IK(env.model.key("home").qpos[:9].copy(), -0.20)
    lo, hi = env.model.jnt_range[:7, 0], env.model.jnt_range[:7, 1]
    rng = np.random.default_rng(0)
    cands = []
    for _ in range(n_seeds):
        q0 = ik.cfg.q.copy()
        q0[:7] = rng.uniform(lo + 0.1, hi - 0.1)
        ik.cfg.update(q0)
        q, dp, da = ik.solve(T_seat, iters=300)
        if dp < 5e-4 and da < 0.2 and not any(np.linalg.norm(q[:7] - c[:7]) < 0.3 for c in cands):
            cands.append(q.copy())
    demos = base_demos()
    lens = [len(np.load(ROOT / "data" / "processed" / s / f"{t}_holder.npz")["t_src"]) for s, t in demos]
    s, t = demos[int(np.argmax(lens))]
    probes = [{"receiver_dx": dx, "receiver_dy": dy, "start_offset": [-0.03, dy and -dy, 0.015],
               "time_scale": 1.0} for dx, dy in ((0, 0), (0.03, 0.03), (-0.03, -0.03), (0.03, -0.03))]
    best = None
    for q in cands:
        _SEAT_REF = q
        worst = 0.0
        for p in probes:
            e = HolderEnv(p["receiver_dx"], p["receiver_dy"], cameras=())
            plan = plan_episode(e, s, t, p)
            qs = np.concatenate([plan["phases"][0][1], plan["phases"][1][1]])
            worst = max(worst, float((qs.max(0) - qs.min(0)).max()), 1e3 * float(np.max(plan["ik_dp"])) / 10)
        print("候補", np.round(q[:7], 2), "関節の動いた幅の最大", round(worst, 2), flush=True)
        if best is None or worst < best[0]:
            best = (worst, q)
    SEAT_REF_PATH.write_text(json.dumps({"q": best[1][:9].tolist(), "max_joint_range_rad": best[0],
                                         "probe_demo": f"{s}/{t}", "n_candidates": len(cands)}, indent=1))
    print("基準:", np.round(best[1][:7], 2), "最大の動いた幅", round(best[0], 2))


def plan_episode(env: HolderEnv, session: str, trial: str, p: dict, posture_cost: float = POSTURE_COST) -> dict:
    src = np.load(ROOT / "data" / "processed" / session / f"{trial}_holder.npz")
    t_src = src["t_src"] - src["t_src"][0]
    fs = 1.0 / np.median(np.diff(t_src))
    T_FO, _, _ = make_targets(src, env.geom, "level", 0.010, fs, True, p["start_offset"])
    plan = plan_phases(env.model, T_FO, env.geom.F_world, env.T_hO, t_src, p["time_scale"],
                       hold_steps=10, q_seat_ref=seat_reference(),
                       posture_cost=posture_cost)
    return trim_static(plan)


def trim_static(plan: dict, skip_start=40, seat_tol_rad=0.002) -> dict:
    """同じ観測に違う行動が対応する静止区間を除く。

    - 開始前の静止（settle）と、接近の最初の skip_start ステップ（最小躍度で速度がほぼ 0）を除き、
      動き出した状態から始める。残すと方策が開始姿勢で止まったまま動き出さない。
    - 人の区間の末尾で、着座姿勢から seat_tol_rad 以内に入った後の静止を除き、0.1 秒で離す。
      残すと「止まっている」観測から「保持」と「解放」のどちらも出るため、解放の判断ができない。
    """
    phases = []
    for name, qs, grip, weld_on in plan["phases"]:
        if name == "settle":
            continue
        if name == "approach":
            qs = qs[skip_start:]
        if name == "human":
            dev = np.abs(qs - qs[-1]).max(axis=1)
            moving = np.where(dev > seat_tol_rad)[0]
            qs = qs[:moving[-1] + 2] if len(moving) else qs[:1]
        phases.append((name, qs, grip, weld_on))
    q_start = plan["q_pre"].copy()
    q_start[:len(phases[0][1][0])] = phases[0][1][0]
    return {**plan, "phases": phases, "q_start": q_start}


def scripted_actions(plan: dict) -> tuple[np.ndarray, np.ndarray]:
    """100 Hz の計画を 50 Hz の行動列にする（関節目標 7 ＋ 解放フラグ）。"""
    acts, names = [], []
    for name, qs, _, weld_on in plan["phases"]:
        for q in qs[::2]:
            acts.append(np.concatenate([q, [0.0 if weld_on else 1.0]]))
            names.append(name)
    return np.array(acts, np.float32), np.array(names)


def action_noise(rng, phase: np.ndarray, max_rad: float, fade_s: float = 0.3) -> np.ndarray:
    """実行する行動に加える、なめらかな関節の揺らぎ（0.2〜1 Hz の正弦波の和）。

    ラベルは計画どおりの行動のままにし、ずれた状態から戻る行動を学ばせる（DART と同じ考え方）。
    人の区間の最後 fade_s 秒で 0 にし、着座・解放以降は加えない。
    """
    n = len(phase)
    t = np.arange(n) * 0.02
    amp = rng.uniform(0, max_rad)
    noise = np.zeros((n, 7))
    for _ in range(3):
        f = rng.uniform(0.2, 1.0, 7)
        ph = rng.uniform(0, 2 * np.pi, 7)
        noise += np.sin(2 * np.pi * f[None] * t[:, None] + ph[None]) / 3
    w = np.isin(phase, ["approach", "human"]).astype(float)
    end = np.where(phase == "human")[0]
    if len(end):
        k = int(fade_s / 0.02)
        w[max(end[-1] - k, 0):end[-1] + 1] = np.linspace(1, 0, min(k, end[-1]) + 1)
    return amp * noise * w[:, None]


def run_one(job) -> dict:
    idx, session, trial, seed, receiver_m, start_m, noise_rad = job
    rng = np.random.default_rng(seed)
    p = sample_params(rng, receiver_m, start_m)
    env = HolderEnv(p["receiver_dx"], p["receiver_dy"])
    plan = plan_episode(env, session, trial, p)
    info = {"index": idx, "session": session, "trial": trial, "seed": seed, **p,
            "ik_pos_err_mm_max": float(np.max(plan["ik_dp"]) * 1000)}
    actions, phase = scripted_actions(plan)
    noise = action_noise(rng, phase, noise_rad)
    info["noise_rad"] = float(np.abs(noise).max()) if len(noise) else 0.0
    obs = env.reset(plan["q_start"])
    qpos, seat_err, imgs = [], [], {c: [] for c in CAMERAS}
    for a, e in zip(actions, noise):
        qpos.append(obs["qpos"])
        seat_err.append(env.seat_error()[0])
        for c in CAMERAS:
            imgs[c].append(obs["images"][c])
        obs = env.step(np.concatenate([a[:7] + e, a[7:]]))
    dp, da = env.seat_error()
    info.update({"steps": len(actions), "seat_err_mm": dp, "seat_err_deg": da,
                 "optics_contact": env.optics_contact, "success": env.success()})
    if not info["success"] or info["ik_pos_err_mm_max"] > 2.0:
        info["saved"] = False
        return info
    path = OUT / f"episode_{idx:04d}.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("observations/qpos", data=np.array(qpos))
        for c in CAMERAS:
            f.create_dataset(f"observations/images/{c}", data=np.array(imgs[c]),
                             chunks=(1, *imgs[c][0].shape), compression="gzip", compression_opts=4)
        f.create_dataset("action", data=actions)
        f.create_dataset("seat_err_mm", data=np.array(seat_err, np.float32))
        f.create_dataset("phase", data=phase.astype("S10"))
        f.attrs["meta"] = json.dumps(info)
    info["saved"] = True
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-demo", type=int, default=10)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--receiver-m", type=float, default=0.03)
    ap.add_argument("--start-m", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-rad", type=float, default=0.015, help="実行時の関節の揺らぎの最大振幅")
    ap.add_argument("--out", default="act", help="data/ の下の出力先")
    ap.add_argument("--find-seat-reference", action="store_true",
                    help="着座姿勢の基準解を探して config/seat_reference.json に保存して終わる")
    args = ap.parse_args()
    if args.find_seat_reference:
        find_seat_reference()
        return

    global OUT
    OUT = ROOT / "data" / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    demos = base_demos()
    jobs = [(i * args.per_demo + k, s, t, args.seed * 100000 + i * args.per_demo + k,
             args.receiver_m, args.start_m, args.noise_rad)
            for i, (s, t) in enumerate(demos) for k in range(args.per_demo)]
    print(f"元の軌跡 {len(demos)} 本 × {args.per_demo} = {len(jobs)} エピソード")
    with Pool(args.workers) as pool:
        infos = []
        for info in pool.imap_unordered(run_one, jobs):
            infos.append(info)
            if len(infos) % 20 == 0:
                print(f"{len(infos)}/{len(jobs)} 保存 {sum(i['saved'] for i in infos)}", flush=True)
    infos.sort(key=lambda i: i["index"])
    (OUT / "index.json").write_text(json.dumps({"args": vars(args), "demos": demos,
                                                "episodes": infos}, indent=1, ensure_ascii=False))
    ok = [i for i in infos if i["saved"]]
    print(f"保存 {len(ok)}/{len(infos)}、失敗した例:",
          [(i["session"], i["trial"], round(i["seat_err_mm"], 1), i["optics_contact"],
            round(i["ik_pos_err_mm_max"], 1)) for i in infos if not i["saved"]][:10])


if __name__ == "__main__":
    main()
