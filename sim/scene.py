"""Panda＋簡略化した顕微鏡受け部＋ホルダーのMuJoCoシーンを組み立てる。

世界座標: 机上面 z=0、Panda基部が原点、x=ロボット前方。
受け部座標 F: 着座したホルダー中心が原点、x=差し込み方向（奥）、y=左、z=上。
F は世界の (receiver_x, 0, stage_top + 厚さ/2) に置き、軸は世界と平行。

実測値（config/receiver_dimensions.json）: ホルダー 127×84×4 mm、受け部 左右103 × 前後165 mm、
前面上縁→ステージ 100 mm、ホルダー→上部照明 45 mm、机→前面上縁 330 mm。
倒立型で対物レンズはホルダーの下。ホルダーは中空の枠。
仮定: ステージ上面＝前面上縁の高さ、ホルダーは受け部の奥側に着座、支持は左右の段、
照明は着座ホルダー中心の真上の円柱。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PANDA_XML = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
# 標準グリッパで薄板の後端を上下から挟むと、手の本体（指の開閉方向に約0.2 m）が
# 前面上縁〜ステージ間（100 mm）のデッキにめり込むため、手先に平板パドルを付けて延長する。
# 把持はパドル先端の下面でホルダー上面を保持する（吸着を想定、シミュレーション上は weld）。
PADDLE_START, PADDLE_LEN = 0.06, 0.25
PADDLE_TH, PADDLE_HALF_W = 0.006, 0.015
# 手前の帯は中央3 mm・両端最大11 mm（ユーザー実測）。直径8 mmのパッド2個を左右端に置く
PAD_R, PAD_SPAN, PRONG_LEN = 0.004, 0.066, 0.05
TCP_IN_HAND = np.array([0.0, 0.0, PADDLE_START + PADDLE_LEN])
# 未計測の仮定: 照明の半径（IMG_0808 でホルダー幅の約9割）、対物レンズ上端とステージ上面の隙間、
# 枠の桟の幅、手前の帯の幅
ILLUM_R = 0.0375
OBJECTIVE_GAP = 0.008  # ユーザー実測 2026-10-21
HOLDER_MASS = 0.035 + 3 * 0.005  # ホルダー35 g＋スライド3枚（各5 g）、ユーザー実測
FRAME_BAR, FRONT_BAND = 0.008, 0.011


@dataclass
class SceneGeom:
    holder_len: float  # 差し込み方向（127 mm）
    holder_wid: float  # 左右（84 mm）
    holder_th: float
    recv_wid: float
    recv_depth: float
    rim_to_stage: float
    illum_gap: float  # ホルダー上面→照明下面
    stage_top: float
    receiver_x: float
    receiver_y: float = 0.0

    @property
    def F_world(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = [self.receiver_x, self.receiver_y, self.stage_top + self.holder_th / 2]
        return T


def load_geom(receiver_x: float = 0.66, receiver_y: float = 0.0) -> SceneGeom:
    d = json.loads((ROOT / "config" / "receiver_dimensions.json").read_text())
    m = d["measurements"]
    h = d["holder_outer"]
    return SceneGeom(
        holder_len=h["width"] / 1000, holder_wid=h["depth"] / 1000, holder_th=h["thickness"] / 1000,
        recv_wid=m["4_stage_width"]["value"] / 1000, recv_depth=m["6_receiver_depth"]["value"] / 1000,
        rim_to_stage=m["3_front_rim_to_stage_depth"]["value"] / 1000,
        illum_gap=m["7_holder_to_illuminator"]["value"] / 1000,
        stage_top=m["1_table_to_front_rim_height"]["value"] / 1000,
        receiver_x=receiver_x, receiver_y=receiver_y,
    )


def _box(body, name, center, half, rgba, contype=1):
    g = body.add_geom()
    g.name = name
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = list(center)
    g.size = list(half)
    g.rgba = list(rgba)
    g.contype = contype
    g.conaffinity = contype
    return g


def add_paddle(spec: mujoco.MjSpec, holder_th: float, flip: int) -> None:
    """二股のパドル。先端の吸着パッドで手前の帯の左右端（太さ最大11 mm）を保持する。

    flip=+1: ホルダー上面が手先 +y 側（パドルは +y 側）、-1: 逆。手先 x がホルダーの左右方向。
    """
    hand = spec.body("hand")
    y_off = flip * (holder_th / 2 + PADDLE_TH / 2)
    stem_len = PADDLE_LEN - PRONG_LEN
    g = _box(hand, "paddle_stem", [0, y_off, PADDLE_START + stem_len / 2],
             [PADDLE_HALF_W, PADDLE_TH / 2, stem_len / 2], [0.8, 0.55, 0.1, 1])
    g.mass = 0.04
    _box(hand, "paddle_bridge", [0, y_off, PADDLE_START + stem_len - 0.004],
         [PAD_SPAN / 2 + 0.005, PADDLE_TH / 2, 0.004], [0.8, 0.55, 0.1, 1])
    for side, s in (("L", 1), ("R", -1)):
        x = s * flip * PAD_SPAN / 2
        _box(hand, f"paddle_prong_{side}", [x, y_off, PADDLE_START + stem_len + PRONG_LEN / 2],
             [0.004, PADDLE_TH / 2, PRONG_LEN / 2], [0.8, 0.55, 0.1, 1])
        pad = hand.add_geom()
        pad.name = f"suction_pad_{side}"
        pad.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        pad.size = [PAD_R, 0.001, 0]
        pad.pos = [x, flip * (holder_th / 2 + 0.001), PADDLE_START + PADDLE_LEN - PAD_R]
        pad.quat = [0.7071068, 0.7071068, 0, 0]
        pad.rgba = [0.1, 0.1, 0.1, 1]
        pad.contype = pad.conaffinity = 0


def add_wrist_camera(spec: mujoco.MjSpec) -> None:
    """手先カメラ。パドルの上（ホルダー上面側＝手先 +y）から、パドル先端とホルダーを見下ろす。"""
    pos, look = np.array([0.0, 0.07, 0.10]), np.array([0.0, 0.0, 0.42])
    fwd = (look - pos) / np.linalg.norm(look - pos)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    cam = spec.body("hand").add_camera()
    cam.name = "wrist"
    cam.pos = pos.tolist()
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = [*right, *np.cross(right, fwd)]
    cam.fovy = 70


def build_model(geom: SceneGeom, flip: int = 1, base_z: float = 0.0) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """base_z: 机上面に対する Panda 基部の高さ。負なら床に置き、装置は机（台）に載る。"""
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.body("link0").pos = [0, 0, base_z]
    add_paddle(spec, geom.holder_th, flip)
    spec.option.timestep = 0.002
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    wb = spec.worldbody
    light = wb.add_light()
    light.pos = [0.3, -0.6, 1.5]
    light.dir = [0, 0.3, -1]
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2, 2, 0.05]
    floor.pos = [0, 0, min(base_z, 0.0)]
    floor.rgba = [0.85, 0.85, 0.82, 1]

    g = geom
    if base_z < 0:
        x0 = g.receiver_x - 0.30
        _box(wb, "table", [x0 + 0.35, 0, base_z / 2], [0.35, 0.4, -base_z / 2], [0.6, 0.5, 0.4, 1])
    fx, fz = g.receiver_x, g.stage_top + g.holder_th / 2
    st = g.stage_top
    back_x = fx + g.holder_len / 2 + 0.001
    stage_front_x = back_x - g.recv_depth
    rim_x = stage_front_x - g.rim_to_stage
    dark, mid, lens_c = [0.12, 0.12, 0.13, 1], [0.25, 0.25, 0.27, 1], [0.35, 0.35, 0.38, 1]

    dev = wb.add_body()
    dev.name = "device"
    # 装置の各部は y=0 を中心に置き、本体ごと receiver_y だけずらす
    dev.pos = [0, g.receiver_y, 0]
    body_x0, body_x1 = rim_x, back_x + 0.12
    # 本体（ポケット床より下）、前面デッキ、左右の段、側壁、奥の当て、対物レンズ、ヘッド
    _box(dev, "dev_body", [(body_x0 + body_x1) / 2, 0, (st - 0.012) / 2],
         [(body_x1 - body_x0) / 2, 0.1665, (st - 0.012) / 2], dark)
    _box(dev, "deck", [(rim_x + stage_front_x) / 2, 0, st - 0.006],
         [(stage_front_x - rim_x) / 2, 0.1665, 0.006], mid)
    ledge_w = g.recv_wid / 2 - (g.holder_wid / 2 - 0.006)
    for side, sgn in (("L", 1), ("R", -1)):
        y_in = sgn * (g.holder_wid / 2 - 0.006)
        y_c = y_in + sgn * ledge_w / 2
        _box(dev, f"ledge_{side}", [(stage_front_x + back_x) / 2, y_c, st - 0.006],
             [g.recv_depth / 2, ledge_w / 2, 0.006], mid)
        y_wall = sgn * (g.recv_wid / 2 + 0.005)
        _box(dev, f"wall_{side}", [(stage_front_x + back_x) / 2, y_wall, st],
             [g.recv_depth / 2, 0.005, 0.012], dark)
        _box(dev, f"tower_{side}", [(stage_front_x + body_x1) / 2, sgn * 0.14, st + 0.15],
             [(body_x1 - stage_front_x) / 2, 0.0265, 0.15], [0.12, 0.12, 0.13, 0.15])
    _box(dev, "backstop", [back_x + 0.005, 0, st], [0.005, g.recv_wid / 2, 0.012], dark)
    # 倒立型: 対物レンズ6本のレボルバーはホルダーの下、照明はホルダーの上
    illum_bottom = st + g.holder_th + g.illum_gap
    illum = dev.add_geom()
    illum.name = "illuminator"
    illum.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    illum.size = [ILLUM_R, 0.04, 0]
    illum.pos = [fx, 0, illum_bottom + 0.04]
    illum.rgba = dark
    _box(dev, "head", [fx + 0.06, 0, illum_bottom + 0.08 + 0.06], [0.12, 0.09, 0.06], dark)
    for k in range(6):
        ang = 2 * np.pi * k / 6
        ob = dev.add_geom()
        ob.name = f"objective_{k}"
        ob.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        ob.size = [0.012, 0.025, 0]
        ob.pos = [fx + 0.028 * np.cos(ang), 0.028 * np.sin(ang), st - OBJECTIVE_GAP - 0.025]
        ob.rgba = lens_c
    target = dev.add_site()
    target.name = "seat_target"
    target.pos = [fx, 0, fz]
    target.size = [0.003, 0, 0]
    target.rgba = [0, 1, 0, 0.5]

    holder = wb.add_body()
    holder.name = "holder"
    holder.pos = [fx, g.receiver_y, fz]
    holder.add_freejoint()
    # 中空の枠: 手前の帯（把持部）、奥の桟、左右の桟
    L, W, th = g.holder_len, g.holder_wid, g.holder_th
    bars = {"holder_front": ([-L / 2 + FRONT_BAND / 2, 0], [FRONT_BAND / 2, W / 2]),
            "holder_back": ([L / 2 - FRAME_BAR / 2, 0], [FRAME_BAR / 2, W / 2]),
            "holder_left": ([0, W / 2 - FRAME_BAR / 2], [L / 2, FRAME_BAR / 2]),
            "holder_right": ([0, -W / 2 + FRAME_BAR / 2], [L / 2, FRAME_BAR / 2])}
    for name, (c, h) in bars.items():
        hg = _box(holder, name, [*c, 0], [*h, th / 2], [0.2, 0.5, 0.95, 1])
        hg.mass = HOLDER_MASS / len(bars)
        hg.friction = [0.6, 0.01, 0.0001]
    hs = holder.add_site()
    hs.name = "holder_center"
    hs.size = [0.003, 0, 0]

    for fb in ("left_finger", "right_finger", "hand"):
        ex = spec.add_exclude()
        ex.bodyname1 = fb
        ex.bodyname2 = "holder"

    eq = spec.add_equality()
    eq.name = "grasp"
    eq.type = mujoco.mjtEq.mjEQ_WELD
    eq.objtype = mujoco.mjtObj.mjOBJ_BODY
    eq.name1 = "hand"
    eq.name2 = "holder"
    eq.solref = [0.005, 1]

    # side_right: iPad と同じく装置の右隣から受け部を見る。overview: 全体
    for name, pos, look, fovy in (
            ("side_right", [fx + 0.02, -0.42, st + 0.16], [fx - 0.12, 0, st + 0.01], 45),
            ("overview", [fx - 0.25, -1.25, st + 0.55], [fx - 0.35, 0, st - 0.05], 45),
            # front_top: 開口部の手前上方から見下ろす。装置の左右のずれが画像の左右のずれになる
            ("front_top", [fx - 0.45, 0.0, st + 0.40], [fx - 0.02, 0, st], 50)):
        cam = wb.add_camera()
        cam.name = name
        cam.pos = pos
        fwd = np.subtract(look, pos)
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1])
        right /= np.linalg.norm(right)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = [*right, *np.cross(right, fwd)]
        cam.fovy = fovy

    add_wrist_camera(spec)
    model = spec.compile()
    return model, spec


def panda_only_model(base_z: float = 0.0) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.body("link0").pos = [0, 0, base_z]
    return spec.compile()


if __name__ == "__main__":
    g = load_geom()
    m, _ = build_model(g)
    print("nq", m.nq, "nu", m.nu, "neq", m.neq, "ngeom", m.ngeom)
