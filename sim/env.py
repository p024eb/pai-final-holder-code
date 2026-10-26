"""ACT の学習データ作成と方策の評価に共通の MuJoCo 環境（50 Hz）。

行動 a (8): Panda 7 関節の目標角 [rad] と把持の解除（0=保持, 1=解放）。解放は一度だけで、戻せない。
観測: qpos (8) = 7 関節角＋指の開き [m]、images = {カメラ名: H×W×3 uint8}。
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "retarget"))
from holder_to_panda import (GRIP_CLOSED, GRIP_OPEN, N_ARM, body_T, holder_in_hand,  # noqa: E402
                             start_holding)
from scene import build_model, load_geom  # noqa: E402

RECEIVER_X = 0.85
CAMERAS = ("front_top", "wrist")
CTRL_DT = 0.02


class HolderEnv:
    def __init__(self, receiver_dx=0.0, receiver_dy=0.0, cameras=CAMERAS, height=180, width=240,
                 base_z=-0.20, flip=1, grip_inset=0.010):
        self.geom = load_geom(RECEIVER_X + receiver_dx, receiver_dy)
        self.model, _ = build_model(self.geom, flip, base_z)
        self.data = mujoco.MjData(self.model)
        self.T_hO = holder_in_hand(self.geom.holder_len, grip_inset, flip)
        self.cameras = cameras
        self.renderer = mujoco.Renderer(self.model, height, width) if cameras else None
        self.substeps = int(round(CTRL_DT / self.model.opt.timestep))
        m = self.model
        self.seat = m.site("seat_target").id
        holder = m.body("holder").id
        self.holder_gids = {g for g in range(m.ngeom) if m.geom_bodyid[g] == holder}
        self.optics_gids = {g for g in range(m.ngeom)
                            if m.geom(g).name.startswith(("objective", "illuminator"))}
        self.eq = None

    def reset(self, q_pre: np.ndarray) -> dict:
        self.eq = start_holding(self.model, self.data, q_pre, self.T_hO)
        self.released = False
        self.optics_contact = False
        self.t = 0.0
        return self.observe()

    def observe(self) -> dict:
        d = self.data
        obs = {"qpos": np.concatenate([d.qpos[:N_ARM], [d.qpos[7] + d.qpos[8]]]).astype(np.float32),
               "images": {}}
        for cam in self.cameras:
            self.renderer.update_scene(d, camera=cam)
            obs["images"][cam] = self.renderer.render().copy()
        return obs

    def step(self, action: np.ndarray) -> dict:
        d = self.data
        if action[7] > 0.5 and not self.released:
            self.released = True
            d.eq_active[self.eq] = 0
        d.ctrl[:N_ARM] = action[:N_ARM]
        d.ctrl[7] = GRIP_OPEN if self.released else GRIP_CLOSED
        for _ in range(self.substeps):
            # 実機の Franka と同じく重力（とコリオリ）を補償する
            d.qfrc_applied[:N_ARM] = d.qfrc_bias[:N_ARM]
            mujoco.mj_step(self.model, d)
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist <= 0 and (c.geom1 in self.optics_gids or c.geom2 in self.optics_gids):
                    self.optics_contact = True
        self.t += CTRL_DT
        return self.observe()

    def seat_error(self) -> tuple[float, float]:
        """ホルダー中心の着座位置からのずれ [mm] と姿勢のずれ [deg]。"""
        T = body_T(self.model, self.data, "holder")
        dp = np.linalg.norm(T[:3, 3] - self.data.site_xpos[self.seat]) * 1000
        da = np.degrees(Rot.from_matrix(T[:3, :3]).magnitude())
        return float(dp), float(da)

    def success(self, tol_mm=5.0, tol_deg=3.0) -> bool:
        dp, da = self.seat_error()
        return self.released and dp < tol_mm and da < tol_deg and not self.optics_contact
