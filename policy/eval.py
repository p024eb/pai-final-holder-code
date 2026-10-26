"""学習した ACT を MuJoCo で動かし、差し込みの成功率を測る。

    MUJOCO_GL=egl python policy/eval.py --run act_v1 [--n 50] [--videos 4]

各エピソードは学習に使っていない乱数で、装置の位置・ホルダーの開始位置を決める（範囲は学習と同じ）。
開始姿勢（ホルダーを持った腕の姿勢）は、元の軌跡の1本を同じ手順で計画して得る。
方策は 50 Hz で行動列（chunk）を出し、時間的アンサンブルでつなぐ。
解放してから 1 秒後に、着座のずれ 5 mm 以内・3 度以内・光学系に非接触なら成功。
出力: runs/<run>/eval.json, runs/<run>/videos/ep_XX.mp4
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "policy"))
sys.path.insert(0, str(ROOT / "sim"))
from act import ACT  # noqa: E402
from env import HolderEnv  # noqa: E402
from make_dataset import base_demos, plan_episode, sample_params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="act_v1")
    ap.add_argument("--ckpt", default="policy_last.pt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--after-release", type=int, default=50, help="解放後に続けるステップ数（1 秒）")
    ap.add_argument("--ensemble-m", type=float, default=0.01)
    ap.add_argument("--receiver-m", type=float, default=0.03)
    ap.add_argument("--start-m", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--videos", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    run = ROOT / "runs" / args.run
    ck = torch.load(run / args.ckpt, map_location="cuda", weights_only=False)
    cams = tuple(ck["cameras"])
    chunk = ck["args"]["chunk"]
    model = ACT(n_cams=len(cams), chunk=chunk).cuda().eval()
    model.load_state_dict(ck["model"])
    st = {k: torch.tensor(v, device="cuda") for k, v in ck["stats"].items()}

    demos = base_demos()
    rng = np.random.default_rng(args.seed)
    envs, infos = [], []
    for k in range(args.n):
        # 学習データと同じく、腕が届かない（IK の誤差 2 mm 超）配置は引き直す
        while True:
            s, t = demos[int(rng.integers(len(demos)))]
            p = sample_params(rng, args.receiver_m, args.start_m)
            env = HolderEnv(p["receiver_dx"], p["receiver_dy"], cameras=cams)
            plan = plan_episode(env, s, t, p)
            if np.max(plan["ik_dp"]) * 1000 <= 2.0:
                break
        env.obs = env.reset(plan["q_start"])
        envs.append(env)
        infos.append({"episode": k, "start_from": f"{s}/{t}", **p})

    vid_dir = run / "videos"
    vid_dir.mkdir(exist_ok=True)
    vids = {}
    for k in range(min(args.videos, args.n)):
        r = mujoco.Renderer(envs[k].model, 360, 480)
        w = imageio.get_writer(vid_dir / f"ep_{k:02d}{args.tag}.mp4", fps=50, codec="libx264",
                               quality=7, macro_block_size=8)
        vids[k] = (r, w)

    n = len(envs)
    buf = torch.zeros(n, args.max_steps + chunk, args.max_steps, 8, device="cuda")
    filled = torch.zeros(n, args.max_steps + chunk, args.max_steps, dtype=torch.bool, device="cuda")
    release_step = [None] * n
    flag_max = np.zeros(n)
    flag_now_max = np.zeros(n)
    active = list(range(n))
    for t in range(args.max_steps):
        if not active:
            break
        img = torch.stack([torch.from_numpy(np.stack([envs[i].obs["images"][c] for c in cams]))
                           for i in active]).cuda().permute(0, 1, 4, 2, 3).float() / 255.0
        q = torch.stack([torch.from_numpy(envs[i].obs["qpos"]) for i in active]).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred, _, _ = model(img, (q - st["qpos_mean"]) / st["qpos_std"])
        pred = pred.float() * st["action_std"] + st["action_mean"]
        idx = torch.tensor(active, device="cuda")
        buf[idx, t:t + chunk, t] = pred
        filled[idx, t:t + chunk, t] = True
        for j, i in enumerate(active):
            m = filled[i, t, :t + 1]
            a_all = buf[i, t, :t + 1][m]
            w = torch.exp(-args.ensemble_m * torch.arange(len(a_all), device="cuda"))
            a = ((a_all * w[:, None]).sum(0) / w.sum()).cpu().numpy()
            flag_max[i] = max(flag_max[i], a[7])
            flag_now_max[i] = max(flag_now_max[i], float(pred[j, 0, 7]))
            envs[i].obs = envs[i].step(a)
            if envs[i].released and release_step[i] is None:
                release_step[i] = t
            if i in vids:
                r, wr = vids[i]
                r.update_scene(envs[i].data, camera="side_right")
                wr.append_data(np.concatenate([r.render(), np.asarray(
                    _resize(envs[i].obs["images"]["wrist"], 360, 480))], axis=1))
        active = [i for i in active
                  if release_step[i] is None or t - release_step[i] < args.after_release]
    for r, w in vids.values():
        w.close()

    for i, env in enumerate(envs):
        dp, da = env.seat_error()
        rel = env.data.xpos[env.model.body("holder").id] - env.data.site_xpos[env.seat]
        infos[i].update({"release_flag_max": round(float(flag_max[i]), 3),
                         "release_flag_latest_max": round(float(flag_now_max[i]), 3),
                         "holder_in_F_mm": (rel * 1000).round(1).tolist(),"released": env.released,
                         "release_s": None if release_step[i] is None else release_step[i] * 0.02,
                         "seat_err_mm": round(dp, 2), "seat_err_deg": round(da, 2),
                         "optics_contact": env.optics_contact, "success": env.success()})
    rate = float(np.mean([i["success"] for i in infos]))
    summary = {"run": args.run, "ckpt": args.ckpt, "n": n, "success_rate": rate,
               "released_rate": float(np.mean([i["released"] for i in infos])),
               "seat_err_mm_median": float(np.median([i["seat_err_mm"] for i in infos])),
               "args": vars(args), "episodes": infos}
    (run / f"eval{args.tag}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("episodes", "args")},
                     ensure_ascii=False))
    print("失敗:", [(i["episode"], i["released"], i["seat_err_mm"], i["optics_contact"])
                  for i in infos if not i["success"]])


def _resize(img, h, w):
    return np.asarray(torch.nn.functional.interpolate(
        torch.from_numpy(img).permute(2, 0, 1)[None].float(), size=(h, w), mode="bilinear")[0]
        .permute(1, 2, 0).clamp(0, 255).byte())


if __name__ == "__main__":
    main()
