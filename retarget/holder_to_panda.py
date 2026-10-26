"""受け部基準のホルダー軌跡を Panda の手先目標に変換し、IK→MuJoCo で再生・評価する。

    python retarget/holder_to_panda.py p03 trial_001 [--time-scale 1.0] [--no-video]

入力: data/processed/<session>/<trial>_holder.npz（human_to_holder.py の出力）
出力: data/sim/<session>/<trial>_{side_right,overview}.mp4, _metrics.json, _joints.npz, _sim.png

把持: ホルダー長辺の後端から grip_inset だけ内側を、手先 z 軸＝差し込み方向、
指の開閉方向（手先 y）＝ホルダー法線でつまむ。把持中は hand–holder の weld で固定し、
解放時に weld を無効化して指を開く。
"""

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
from scene import ILLUM_R, TCP_IN_HAND, build_model, load_geom, panda_only_model  # noqa: E402

N_ARM = 7
GRIP_CLOSED, GRIP_OPEN = 13.0, 255.0  # 指の間隔 ≈4 mm / 80 mm
FPS = 30


def holder_in_hand(holder_len: float, grip_inset: float, flip: int = 1) -> np.ndarray:
    """ホルダー x＝手先 z、ホルダー法線（z）＝手先 flip*y。後端から grip_inset をパドル先端で保持。"""
    T = np.eye(4)
    T[:3, :3] = np.array([[0, flip, 0], [0, 0, flip], [1, 0, 0]], float)
    T[:3, 3] = TCP_IN_HAND + np.array([0, 0, holder_len / 2 - grip_inset])
    return T


def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def pose_err(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    dp = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    da = float(np.degrees(Rot.from_matrix(T_a[:3, :3].T @ T_b[:3, :3]).magnitude()))
    return dp, da


class IK:
    def __init__(self, q_init: np.ndarray, base_z: float = 0.0, posture_cost: float = 1e-3):
        self.model = panda_only_model(base_z)
        self.cfg = mink.Configuration(self.model)
        self.cfg.update(q_init)
        self.task = mink.FrameTask("hand", "body", position_cost=1.0, orientation_cost=0.5,
                                   lm_damping=1e-3)
        self.posture = mink.PostureTask(self.model, cost=posture_cost)
        self.posture.set_target(q_init)
        self.limits = [mink.ConfigurationLimit(self.model)]

    def solve(self, T_target: np.ndarray, iters: int = 30, dt: float = 0.02):
        self.task.set_target(mink.SE3.from_matrix(T_target))
        for _ in range(iters):
            v = mink.solve_ik(self.cfg, [self.task, self.posture], dt, "daqp",
                              damping=1e-4, limits=self.limits)
            self.cfg.integrate_inplace(v, dt)
            dp, da = pose_err(self.cfg.get_transform_frame_to_world("hand", "body").as_matrix(),
                              T_target)
            if dp < 2e-4 and da < 0.05:
                break
        return self.cfg.q.copy(), dp, da

    def solve_global(self, T_target: np.ndarray, n_seeds: int = 64, seed: int = 0):
        """可動域内の一様乱数を初期値に多数回解き、誤差が最小で中立姿勢に近い解を採る。"""
        rng = np.random.default_rng(seed)
        lo, hi = self.model.jnt_range[:N_ARM, 0], self.model.jnt_range[:N_ARM, 1]
        mid = (lo + hi) / 2
        q_base = self.cfg.q.copy()
        best = None
        for _ in range(n_seeds):
            q0 = q_base.copy()
            q0[:N_ARM] = rng.uniform(lo + 0.1, hi - 0.1)
            self.cfg.update(q0)
            q, dp, da = self.solve(T_target, iters=300)
            margin = np.min(np.minimum(q[:N_ARM] - lo, hi - q[:N_ARM]))
            score = dp * 1000 + da + 0.1 * np.linalg.norm(q[:N_ARM] - mid) - 2 * min(margin, 0.3)
            if best is None or score < best[0]:
                best = (score, q, dp, da)
        self.cfg.update(best[1])
        self.posture.set_target(best[1])
        return best[1], best[2], best[3]


def _level_pose(T: np.ndarray) -> np.ndarray:
    """同じ x, y とヨーのまま、水平で着座高さ（z=0）にした姿勢。"""
    yaw = Rot.from_matrix(T[:3, :3]).as_euler("ZYX")[0]
    L = np.eye(4)
    L[:3, :3] = Rot.from_euler("z", yaw).as_matrix()
    L[:2, 3] = T[:2, 3]
    return L


def _blend(T: np.ndarray, L: np.ndarray, a: float) -> np.ndarray:
    out = T.copy()
    r = Rot.from_matrix(np.stack([T[:3, :3], L[:3, :3]]))
    out[:3, :3] = (r[0] * Rot.from_rotvec(a * (r[0].inv() * r[1]).as_rotvec())).as_matrix()
    out[2, 3] = (1 - a) * T[2, 3] + a * L[2, 3]
    return out


def clearance_correction(T_FO: np.ndarray, geom, fs: float,
                         margin: float = 0.003) -> tuple[np.ndarray, np.ndarray]:
    """上部照明にもデッキ・支持段にも当たらないよう、各時刻で人の姿勢を
    「水平・着座高さ」の姿勢へ最小限だけ寄せる（寄せ率 a∈[0,1]）。a は平滑化して滑らかにつなぐ。
    """
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    xs = np.linspace(-geom.holder_len / 2, geom.holder_len / 2, 13)
    ys = np.linspace(-geom.holder_wid / 2, geom.holder_wid / 2, 9)
    P = np.array([[x, y, s * geom.holder_th / 2, 1.0] for x in xs for y in ys for s in (-1, 1)]).T
    illum_bottom = geom.holder_th / 2 + geom.illum_gap
    support_top = -geom.holder_th / 2
    rim = geom.holder_len / 2 + 0.001 - geom.recv_depth - geom.rim_to_stage

    def ok(T):
        Q = (T @ P)[:3]
        under = np.hypot(Q[0], Q[1]) < ILLUM_R + margin
        if (Q[2, under] > illum_bottom - margin).any():
            return False
        over_support = Q[0] > rim
        return not (Q[2, over_support] < support_top - 0.0005).any()

    a = np.zeros(len(T_FO))
    for i, T in enumerate(T_FO):
        if ok(T):
            continue
        L = _level_pose(T)
        lo, hi = 0.0, 1.0
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if ok(_blend(T, L, mid)) else (mid, hi)
        a[i] = hi
    a = maximum_filter1d(a, int(0.4 * fs) | 1)
    a = np.clip(np.maximum(gaussian_filter1d(a, 0.1 * fs), a), 0, 1)
    out = np.array([_blend(T, _level_pose(T), ai) if ai > 0 else T for T, ai in zip(T_FO, a)])
    return out, a


def holder_targets(src, orientation: str, holder_len: float, grip_inset: float) -> np.ndarray:
    """受け部座標でのホルダー目標姿勢の列。

    Vision Pro で観測できるのは手の姿勢だけで、手の中でのホルダーの回転・持ち替えは見えない。
    level / level_yaw では人のデータからつまみ点の位置と時間だけを使い、
    ホルダーは水平（受け部に入る姿勢）とし、中心はつまみ点から差し込み方向へ (L/2 - inset) 先に置く。
    """
    T_FO = src["T_FO"]
    if orientation == "human" or "pinch_pt_F" not in src.files:
        return T_FO.copy()
    p = src["pinch_pt_F"]
    out = np.tile(np.eye(4), (len(p), 1, 1))
    for i in range(len(p)):
        yaw = Rot.from_matrix(T_FO[i, :3, :3]).as_euler("ZYX")[0] if orientation == "level_yaw" else 0.0
        R = Rot.from_euler("z", yaw).as_matrix()
        out[i, :3, :3] = R
        out[i, :3, 3] = p[i] + R[:, 0] * (holder_len / 2 - grip_inset)
    return out


def project_into_receiver(T_FO: np.ndarray, geom, fs: float) -> np.ndarray:
    """人の押し込みすぎ（奥の当て・左右の壁・支持面を越える分）をその手前で止める。

    人は当てに押し付けて止まるが、位置制御のロボットがそのまま追うと当てを強く押すため、
    接触で止まる柔らかい制御の代わりに目標側を拘束内へ射影する。水平姿勢（level*）での使用を想定。
    """
    from scipy.ndimage import gaussian_filter1d
    L, W = geom.holder_len, geom.holder_wid
    stage_front = L / 2 + 0.001 - geom.recv_depth
    rim = stage_front - geom.rim_to_stage
    y_play = (geom.recv_wid - W) / 2 - 0.001

    def clamp(P):
        P = P.copy()
        P[:, 0] = np.minimum(P[:, 0], 0.0)
        in_pocket = P[:, 0] + L / 2 > stage_front
        P[in_pocket, 1] = np.clip(P[in_pocket, 1], -y_play, y_play)
        over_support = P[:, 0] + L / 2 > rim
        P[over_support, 2] = np.maximum(P[over_support, 2], 0.0)
        return P

    P = clamp(T_FO[:, :3, 3])
    P = clamp(gaussian_filter1d(P, 0.08 * fs, axis=0, mode="nearest"))
    out = T_FO.copy()
    out[:, :3, 3] = P
    return out


def body_T(model, data, name) -> np.ndarray:
    b = model.body(name).id
    T = np.eye(4)
    T[:3, :3] = data.xmat[b].reshape(3, 3)
    T[:3, 3] = data.xpos[b]
    return T


def set_weld_from_current(model, data, eq_id: int) -> None:
    """現在の hand と holder の相対姿勢を weld の relpose に設定する。"""
    T1 = body_T(model, data, "hand")
    T2 = body_T(model, data, "holder")
    rel = np.linalg.inv(T1) @ T2
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, rel[:3, :3].flatten())
    model.eq_data[eq_id, 0:3] = 0
    model.eq_data[eq_id, 3:6] = rel[:3, 3]
    model.eq_data[eq_id, 6:10] = q
    model.eq_data[eq_id, 10] = 1


def classify_contacts(model, data, holder_gids: set, device_gids: set, robot_bodies: set):
    hits = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if c.dist > 0:
            continue
        if g1 in holder_gids or g2 in holder_gids:
            other = g2 if g1 in holder_gids else g1
            if other in device_gids:
                hits.append(("holder", model.geom(other).name, -c.dist))
        elif (g1 in device_gids and model.geom_bodyid[g2] in robot_bodies) or \
             (g2 in device_gids and model.geom_bodyid[g1] in robot_bodies):
            dev = g1 if g1 in device_gids else g2
            rob = g2 if dev == g1 else g1
            hits.append(("robot:" + model.body(model.geom_bodyid[rob]).name,
                         model.geom(dev).name, -c.dist))
    return hits


def make_targets(src, geom, orientation: str, grip_inset: float, fs: float,
                 clearance_fix: bool = True, start_offset=None):
    """受け部座標でのホルダー目標列（補正済み, 補正前, 補正の寄せ率）。

    start_offset: 受け部座標での開始位置のずらし [m]。着座点からの距離に比例して効かせ、着座点では 0。
    """
    T_raw = holder_targets(src, orientation, geom.holder_len, grip_inset)
    if start_offset is not None:
        d = np.linalg.norm(T_raw[:, :3, 3] - T_raw[-1, :3, 3], axis=1)
        w = np.clip(d / max(d[0], 1e-6), 0, 1)
        T_raw[:, :3, 3] += w[:, None] * np.asarray(start_offset)[None]
    if orientation != "human":
        T_raw = project_into_receiver(T_raw, geom, fs)
    if clearance_fix:
        T, blend = clearance_correction(T_raw, geom, fs)
    else:
        T, blend = T_raw.copy(), np.zeros(len(T_raw))
    return T, T_raw, blend


def plan_phases(model, T_FO, F, T_hO, t_src, time_scale=1.0, approach_s=3.0, retreat_m=0.12,
                flip=1, base_z=-0.20, dt_ctrl=0.01, hold_steps=30, q_seat_ref=None,
                posture_cost=1e-3):
    """ホルダー目標列から、100 Hz の関節目標をフェーズごとに作る。

    q_seat_ref: 着座姿勢の基準解。与えるとまずそこから局所的に解き、配置が変わっても同じ解の枝を使う。
    """
    T_Oh = np.linalg.inv(T_hO)
    home = model.key("home").qpos[:9].copy()
    ik = IK(home, base_z, posture_cost)
    T_hand_targets = np.array([F @ T @ T_Oh for T in T_FO])
    # 開始姿勢: 最初の目標から手前に 0.10 m・上に 0.06 m 離れた位置（同じ向き）
    T_pre = T_hand_targets[0].copy()
    T_pre[:3, 3] += np.array([-0.10, 0.0, 0.06])
    # 着座姿勢（全試行で共通、確実に解ける）を大域的に解き、時間を逆向きにたどって解く。
    # 区間の最初から順に解くと、運搬中の姿勢で選んだ解の枝が着座付近で関節限界に張り付き、解が飛ぶ
    seated = False
    if q_seat_ref is not None:
        ik.cfg.update(np.concatenate([q_seat_ref[:N_ARM], home[N_ARM:]]))
        q, dp, da = ik.solve(T_hand_targets[-1], iters=300)
        if dp < 1e-3 and da < 0.5:
            ik.posture.set_target(q)
            seated = True
    if not seated:
        ik.solve_global(T_hand_targets[-1])
    q_human, ik_dp, ik_da = [], [], []
    for T in T_hand_targets[::-1]:
        q, dp, da = ik.solve(T)
        q_human.append(q[:N_ARM])
        ik_dp.append(dp)
        ik_da.append(da)
    q_human, ik_dp, ik_da = (np.array(x[::-1]) for x in (q_human, ik_dp, ik_da))
    n_app = int(approach_s / dt_ctrl)
    q_app = []
    for s in min_jerk(np.linspace(1, 0, n_app)):
        T = T_hand_targets[0].copy()
        T[:3, 3] = T_pre[:3, 3] + (T_hand_targets[0][:3, 3] - T_pre[:3, 3]) * s
        q_app.append(ik.solve(T)[0][:N_ARM])
    q_app = np.array(q_app[::-1])
    q_pre = np.concatenate([q_app[0], home[N_ARM:]])
    ik.cfg.update(np.concatenate([q_human[-1], home[N_ARM:]]))

    # 退避: パドルをホルダー上面から 3 mm 浮かせ、手先 -z 方向へ引く
    T_last = T_hand_targets[-1]
    up = flip * T_last[:3, 1]
    retreat = []
    for s in np.linspace(0, 1, 181):
        T = T_last.copy()
        lift = 0.003 * min_jerk(min(s / 0.15, 1.0))
        back = retreat_m * min_jerk(max((s - 0.15) / 0.85, 0.0))
        T[:3, 3] = T_last[:3, 3] + up * lift - T_last[:3, 2] * back
        retreat.append(ik.solve(T)[0][:N_ARM])
    retreat = np.array(retreat)

    t_h = (t_src - t_src[0]) * time_scale
    n_h = int(round(t_h[-1] / dt_ctrl)) + 1
    q_h = np.stack([np.interp(np.arange(n_h) * dt_ctrl, t_h, q_human[:, j]) for j in range(N_ARM)], 1)
    phases = [("settle", np.repeat(q_pre[None, :N_ARM], 50, 0), GRIP_CLOSED, True),
              ("approach", q_app, GRIP_CLOSED, True),
              ("human", q_h, GRIP_CLOSED, True),
              ("hold", np.repeat(q_h[-1:], hold_steps, 0), GRIP_CLOSED, True),
              ("release", np.repeat(q_h[-1:], 50, 0), GRIP_OPEN, False),
              ("retreat", retreat, GRIP_OPEN, False),
              ("rest", np.repeat(retreat[-1:], 100, 0), GRIP_OPEN, False)]
    return {"phases": phases, "q_pre": q_pre, "ik_dp": ik_dp, "ik_da": ik_da}


def start_holding(model, data, q_pre, T_hO) -> int:
    """開始姿勢の手先にホルダーを持たせ、把持の weld を張る。weld の id を返す。"""
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[:N_ARM] = q_pre[:N_ARM]
    data.qpos[7:9] = 0.002
    mujoco.mj_forward(model, data)
    T_holder0 = body_T(model, data, "hand") @ T_hO
    hq = model.joint(model.body("holder").jntadr[0]).qposadr[0]
    data.qpos[hq:hq + 3] = T_holder0[:3, 3]
    mujoco.mju_mat2Quat(data.qpos[hq + 3:hq + 7], T_holder0[:3, :3].flatten())
    mujoco.mj_forward(model, data)
    eq = model.equality("grasp").id
    set_weld_from_current(model, data, eq)
    data.eq_active[eq] = 1
    return eq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("trial")
    ap.add_argument("--time-scale", type=float, default=1.0, help="人の区間を何倍の時間で再生するか")
    ap.add_argument("--approach-s", type=float, default=3.0)
    ap.add_argument("--receiver-x", type=float, default=0.85)
    ap.add_argument("--base-z", type=float, default=-0.20,
                    help="机上面に対する Panda 基部の高さ。0 だと作業高さが肩の高さと重なり IK が特異姿勢付近で解けない")
    ap.add_argument("--flip", type=int, default=1, choices=[1, -1])
    ap.add_argument("--grip-inset-m", type=float, default=0.010,
                    help="ホルダー手前端からパドル先端（吸着点）までの距離。手前の帯は幅約15 mm")
    ap.add_argument("--retreat-m", type=float, default=0.12)
    ap.add_argument("--no-clearance-fix", dest="clearance_fix", action="store_false",
                    help="照明・支持面に当たる時刻だけ人の姿勢を水平・着座高さへ最小限寄せる補正（既定）を切る")
    ap.add_argument("--orientation", choices=["level", "level_yaw", "human"], default="level",
                    help="ホルダー姿勢の決め方。level: 水平・差し込み方向に固定、level_yaw: 水平で人のヨーのみ、"
                         "human: 手首の姿勢をそのまま（手の中での持ち替えがないと仮定）")
    ap.add_argument("--tag", default="", help="出力ファイル名の接尾辞")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    src = np.load(ROOT / "data" / "processed" / args.session / f"{args.trial}_holder.npz")
    t_src = src["t_src"] - src["t_src"][0]
    fs = 1.0 / np.median(np.diff(t_src))

    geom = load_geom(args.receiver_x)
    T_FO, T_FO_raw, blend = make_targets(src, geom, args.orientation, args.grip_inset_m, fs,
                                         args.clearance_fix)
    corr_mm = np.linalg.norm(T_FO[:, :3, 3] - T_FO_raw[:, :3, 3], axis=1) * 1000
    corr_deg = np.degrees([Rot.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()
                           for a, b in zip(T_FO_raw, T_FO)])
    model, _ = build_model(geom, args.flip, args.base_z)
    data = mujoco.MjData(model)
    F = geom.F_world
    T_hO = holder_in_hand(geom.holder_len, args.grip_inset_m, args.flip)
    plan = plan_phases(model, T_FO, F, T_hO, t_src, args.time_scale, args.approach_s,
                       args.retreat_m, args.flip, args.base_z)
    phases, ik_dp, ik_da = plan["phases"], plan["ik_dp"], plan["ik_da"]
    dt_ctrl = 0.01
    eq = start_holding(model, data, plan["q_pre"], T_hO)

    holder_body = model.body("holder").id
    holder_gids = {g for g in range(model.ngeom) if model.geom_bodyid[g] == holder_body}
    dev_body = model.body("device").id
    device_gids = {g for g in range(model.ngeom) if model.geom_bodyid[g] == dev_body}
    robot_bodies = {model.body(n).id for n in
                    ["link1", "link2", "link3", "link4", "link5", "link6", "link7",
                     "hand", "left_finger", "right_finger"]}
    seat = model.site("seat_target").id

    renderers = {}
    writers = {}
    out_dir = ROOT / "data" / "sim" / args.session
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.trial + (f"_{args.tag}" if args.tag else "")
    if not args.no_video:
        for cam in ("side_right", "overview"):
            renderers[cam] = mujoco.Renderer(model, 720, 1280)
            writers[cam] = imageio.get_writer(out_dir / f"{stem}_{cam}.mp4", fps=FPS,
                                              codec="libx264", quality=7, macro_block_size=8)

    substeps = int(round(dt_ctrl / model.opt.timestep))
    log = {k: [] for k in ("t", "phase", "q_des", "q", "holder_T", "track_mm", "track_deg")}
    contacts = {}
    weld_force = {p[0]: [] for p in phases}
    t_sim, frame_t = 0.0, 0.0
    i_h = 0
    for name, qs, grip, weld_on in phases:
        if not weld_on and data.eq_active[eq]:
            data.eq_active[eq] = 0
        for k, q_des in enumerate(qs):
            data.ctrl[:N_ARM] = q_des
            data.ctrl[7] = grip
            for _ in range(substeps):
                # 実機の Franka と同じく重力（とコリオリ）を補償し、位置制御の垂れ下がりをなくす
                data.qfrc_applied[:N_ARM] = data.qfrc_bias[:N_ARM]
                mujoco.mj_step(model, data)
                if weld_on:
                    rows = np.where((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                                    & (data.efc_id == eq))[0]
                    if len(rows) >= 6:
                        f = data.efc_force[rows]
                        weld_force[name].append((np.linalg.norm(f[:3]), np.linalg.norm(f[3:6])))
                for who, dev, depth in classify_contacts(model, data, holder_gids, device_gids,
                                                         robot_bodies):
                    key = f"{name}|{who}|{dev}"
                    c = contacts.setdefault(key, {"steps": 0, "max_depth_mm": 0.0, "t0": t_sim})
                    c["steps"] += 1
                    c["max_depth_mm"] = max(c["max_depth_mm"], depth * 1000)
            t_sim += dt_ctrl
            T_hold = body_T(model, data, "holder")
            log["t"].append(t_sim)
            log["phase"].append(name)
            log["q_des"].append(q_des.copy())
            log["q"].append(data.qpos[:N_ARM].copy())
            log["holder_T"].append(T_hold)
            if name == "human":
                dp, da = pose_err(T_hold, F @ T_FO[min(int(round(k * dt_ctrl / args.time_scale * 100)),
                                                        len(T_FO) - 1)])
                log["track_mm"].append(dp * 1000)
                log["track_deg"].append(da)
                i_h += 1
            while renderers and frame_t <= t_sim:
                for cam, r in renderers.items():
                    r.update_scene(data, camera=cam)
                    writers[cam].append_data(r.render())
                frame_t += 1 / FPS
    for w in writers.values():
        w.close()

    # --- 評価 ---
    T_final = log["holder_T"][-1]
    T_seat = np.eye(4)
    T_seat[:3, 3] = data.site_xpos[seat]
    rel = np.linalg.inv(F) @ T_final
    eul = Rot.from_matrix(rel[:3, :3]).as_euler("xyz", degrees=True)
    final_dp, final_da = pose_err(T_final, T_seat)
    tol_pos_mm, tol_deg = 5.0, 3.0
    holder_hits = {k: v for k, v in contacts.items() if "|holder|" in k}
    robot_hits = {k: v for k, v in contacts.items() if "|robot:" in k}
    lens_hit = any("objective" in k or "illuminator" in k for k in contacts)
    metrics = {
        "session": args.session, "trial": args.trial, "time_scale": args.time_scale,
        "receiver_x_m": args.receiver_x, "base_z_m": args.base_z, "grip_inset_m": args.grip_inset_m,
        "clearance_fix": {"blend_max": round(float(blend.max()), 3),
                          "active_s": round(float((blend > 1e-3).sum() / fs), 2),
                          "pos_change_mm_max": round(float(corr_mm.max()), 1),
                          "rot_change_deg_max": round(float(corr_deg.max()), 1)},
        "ik_pos_err_mm": {"max": float(np.max(ik_dp) * 1000), "mean": float(np.mean(ik_dp) * 1000)},
        "ik_rot_err_deg": {"max": float(np.max(ik_da)), "mean": float(np.mean(ik_da))},
        "holder_tracking_during_human": {"pos_mm_max": float(np.max(log["track_mm"])),
                                         "pos_mm_mean": float(np.mean(log["track_mm"])),
                                         "rot_deg_max": float(np.max(log["track_deg"]))},
        "final_holder_in_F": {"pos_mm": (rel[:3, 3] * 1000).round(2).tolist(),
                              "rpy_deg": eul.round(2).tolist()},
        "final_error": {"pos_mm": round(final_dp * 1000, 2), "rot_deg": round(final_da, 2)},
        "success": bool(final_dp * 1000 < tol_pos_mm and final_da < tol_deg and not lens_hit),
        "success_rule": f"解放後1秒でホルダー中心が着座位置から{tol_pos_mm} mm以内・{tol_deg}度以内、かつ対物レンズ・照明に非接触",
        "optics_contact": lens_hit,
        "grasp_load": {p: {"force_N_max": round(float(np.max(np.array(v)[:, 0])), 2),
                           "force_N_p95": round(float(np.percentile(np.array(v)[:, 0], 95)), 2),
                           "torque_Nm_max": round(float(np.max(np.array(v)[:, 1])), 3),
                           "torque_Nm_p95": round(float(np.percentile(np.array(v)[:, 1], 95)), 3)}
                       for p, v in weld_force.items() if v},
        "holder_contacts": holder_hits,
        "robot_contacts": robot_hits,
    }
    (out_dir / f"{stem}_metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    np.savez_compressed(out_dir / f"{stem}_joints.npz", t=np.array(log["t"]),
                        q_des=np.array(log["q_des"]), q=np.array(log["q"]),
                        holder_T=np.array(log["holder_T"]), phase=np.array(log["phase"]))

    t = np.array(log["t"])
    HT = np.array(log["holder_T"])
    relF = np.einsum("ij,njk->nik", np.linalg.inv(F), HT)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for j, lab in enumerate("xyz"):
        ax[0].plot(t, relF[:, j, 3] * 1000, label=lab)
    ax[0].set_ylabel("holder in F [mm]")
    ax[0].set_ylim(-400, 200)
    ax[0].legend()
    qd, qq = np.array(log["q_des"]), np.array(log["q"])
    ax[1].plot(t, np.degrees(qd - qq))
    ax[1].set_ylabel("joint tracking err [deg]")
    ax[2].plot(t, np.linalg.norm(np.diff(np.vstack([qd[:1], qd]), axis=0), axis=1) / dt_ctrl)
    ax[2].set_ylabel("|dq_des/dt| [rad/s]")
    ax[2].set_xlabel("sim time [s]")
    ph = np.array(log["phase"])
    for a in ax:
        for p in ("approach", "human", "release", "retreat"):
            idx = np.where(ph == p)[0]
            if len(idx):
                a.axvline(t[idx[0]], color="gray", lw=0.6, ls=":")
    ax[0].set_title(f"{args.session}/{args.trial}: Panda replay (success={metrics['success']})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_sim.png", dpi=100)
    print(json.dumps(metrics, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
