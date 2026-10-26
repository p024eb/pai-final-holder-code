"""ACT の学習。

    python policy/train.py --name act_v1 [--steps 30000]

data/<data>/episode_*.hdf5 を使う。--val-demos の元の軌跡から作ったエピソードは検証用に外す
（未知の人の動きへの汎化を見るため）。出力: runs/<name>/{policy_last.pt, policy_best.pt, log.json}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "policy"))
from act import ACT, act_loss  # noqa: E402

CAMERAS = ("front_top", "wrist")
DEFAULT_VAL = ["H01/trial_008", "H01/trial_016", "p04/trial_009"]


def list_episodes(data_dir: Path, val_demos: list[str]):
    index = json.loads((data_dir / "index.json").read_text())
    train, val = [], []
    for e in index["episodes"]:
        if not e["saved"]:
            continue
        path = data_dir / f"episode_{e['index']:04d}.hdf5"
        (val if f"{e['session']}/{e['trial']}" in val_demos else train).append(path)
    return train, val


def compute_stats(paths) -> dict:
    qs, acts = [], []
    for p in paths:
        with h5py.File(p) as f:
            qs.append(f["observations/qpos"][()])
            acts.append(f["action"][()])
    qs, acts = np.concatenate(qs), np.concatenate(acts)
    stats = {"qpos_mean": qs.mean(0), "qpos_std": np.clip(qs.std(0), 1e-2, None),
             "action_mean": acts.mean(0), "action_std": np.clip(acts.std(0), 1e-2, None)}
    # 解放フラグは 0/1 のまま扱う（推論時に 0.5 でしきい値処理）
    stats["action_mean"][7], stats["action_std"][7] = 0.0, 1.0
    return {k: v.astype(np.float32) for k, v in stats.items()}


class EpisodeDataset(Dataset):
    def __init__(self, paths, stats, chunk, samples_per_epoch, release_tol_mm=0.0):
        self.paths, self.stats, self.chunk, self.n = paths, stats, chunk, samples_per_epoch
        self.release_tol_mm = release_tol_mm
        self.lengths = []
        for p in paths:
            with h5py.File(p) as f:
                self.lengths.append(len(f["action"]))
        self.files = {}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        e = int(rng.integers(len(self.paths)))
        if e not in self.files:
            self.files[e] = h5py.File(self.paths[e], "r")
        f, T = self.files[e], self.lengths[e]
        t = int(rng.integers(T))
        qpos = f["observations/qpos"][t]
        imgs = np.stack([f[f"observations/images/{c}"][t] for c in CAMERAS])
        act = f["action"][t:t + self.chunk].copy()
        if self.release_tol_mm > 0:
            # 着座位置から release_tol_mm 以内に入った時刻からは「離す」とする（時刻でなく状態で決まるラベル）
            # 運搬の途中で一瞬近づく場合を除くため、元の解放の 0.5 秒前以降に限る
            rel = int(np.argmax(f["action"][:, 7] > 0.5))
            near = np.where(f["seat_err_mm"][()] < self.release_tol_mm)[0]
            near = near[near >= rel - 25]
            if len(near):
                act[np.arange(t, t + len(act)) >= near[0], 7] = 1.0
        n = len(act)
        is_pad = np.zeros(self.chunk, bool)
        if n < self.chunk:
            act = np.concatenate([act, np.repeat(act[-1:], self.chunk - n, 0)])
            is_pad[n:] = True
        s = self.stats
        return (torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0,
                torch.from_numpy((qpos - s["qpos_mean"]) / s["qpos_std"]),
                torch.from_numpy((act - s["action_mean"]) / s["action_std"]),
                torch.from_numpy(is_pad))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="act_v1")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--kl-weight", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-demos", nargs="*", default=DEFAULT_VAL)
    ap.add_argument("--data", default="act", help="data/ の下の学習データ")
    ap.add_argument("--release-tol-mm", type=float, default=0.0,
                    help=">0 なら、着座位置からこの距離以内に入った時刻以降の解放フラグを 1 にする")
    args = ap.parse_args()

    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    train_paths, val_paths = list_episodes(ROOT / "data" / args.data, args.val_demos)
    stats = compute_stats(train_paths)
    print(f"学習 {len(train_paths)} エピソード、検証 {len(val_paths)} エピソード")
    mk = lambda paths, n: DataLoader(EpisodeDataset(paths, stats, args.chunk, n, args.release_tol_mm),
                                     batch_size=args.batch,
                                     num_workers=args.workers, pin_memory=True, persistent_workers=True)
    train_dl = mk(train_paths, args.steps * args.batch)
    val_dl = mk(val_paths, 20 * args.batch) if val_paths else None

    dev = "cuda"
    model = ACT(n_cams=len(CAMERAS), chunk=args.chunk).to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                             {"params": bb, "lr": args.lr_backbone}], weight_decay=1e-4)

    def save(name, step, val_l1=None):
        torch.save({"model": model.state_dict(), "stats": stats, "args": vars(args), "step": step,
                    "val_l1": val_l1, "cameras": CAMERAS}, out / name)

    log, best, t0 = [], float("inf"), time.time()
    model.train()
    for step, (img, q, a, pad) in enumerate(train_dl, 1):
        img, q, a, pad = (x.to(dev, non_blocking=True) for x in (img, q, a, pad))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, mu, logvar = model(img, q, a, pad)
        loss, parts = act_loss(pred.float(), a, pad, mu.float(), logvar.float(), args.kl_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            rec = {"step": step, "loss": loss.item(), **parts, "min": round((time.time() - t0) / 60, 1)}
            if val_dl is not None and step % 2000 == 0:
                model.eval()
                l1s = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for img, q, a, pad in val_dl:
                        img, q, a, pad = (x.to(dev) for x in (img, q, a, pad))
                        pred, _, _ = model(img, q)
                        l1s.append(((pred.float() - a).abs().mean(-1) * ~pad).sum().item()
                                   / (~pad).sum().item())
                rec["val_l1"] = float(np.mean(l1s))
                if rec["val_l1"] < best:
                    best = rec["val_l1"]
                    save("policy_best.pt", step, best)
                model.train()
            log.append(rec)
            print(json.dumps(rec), flush=True)
            (out / "log.json").write_text(json.dumps(log))
        if step % 5000 == 0:
            save("policy_last.pt", step)
    save("policy_last.pt", args.steps)


if __name__ == "__main__":
    main()
