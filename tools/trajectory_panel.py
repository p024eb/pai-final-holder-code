"""提出動画用: 人の実演から求めたホルダーの軌跡の図に、動画の時刻に合わせて赤い点を動かす。

図は 3 段。
  (a) 上から見たホルダーの動き（受け部の壁と着座位置を描き、ホルダーの外形を動かす）
  (b) 受け部を基準にしたホルダー中心の位置の時間変化（x: 差し込み方向、y: 左右、z: 高さ）
  (c) 親指と人差し指の先の距離（つまみ距離）。把持・解放の判定に使った量
背景は matplotlib で一度だけ描き、軸の座標変換を使って各フレームの赤い点だけを PIL で重ねる。
"""

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()

GRIP_INSET = 0.010
HOLD_PINCH_CM = 1.9
RED = (230, 30, 30)


class TrajectoryPanel:
    def __init__(self, session: str, trial: str, width: int = 800, height: int = 620):
        d = np.load(ROOT / "data" / "processed" / session / f"{trial}_holder.npz")
        dims = json.loads((ROOT / "config" / "receiver_dimensions.json").read_text())
        self.L = dims["holder_outer"]["width"]  # mm, 差し込み方向
        self.Wd = dims["holder_outer"]["depth"]  # mm, 左右
        m = dims["measurements"]
        recv_w, recv_d = m["4_stage_width"]["value"], m["6_receiver_depth"]["value"]
        self.t = d["t_src"] - d["t_src"][0]
        self.dur = float(self.t[-1])
        # 水平と仮定したホルダーの中心 = つまみ点から差し込み方向へ (L/2 - つまむ位置) 先
        self.p = d["pinch_pt_F"] * 1000 + np.array([self.L / 2 - GRIP_INSET * 1000, 0, 0])
        self.pinch = d["pinch"] * 100

        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08)
        gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 1, 0.8], hspace=0.6)
        ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))

        back = self.L / 2 + 1
        front = back - recv_d
        ax0.add_patch(plt.Rectangle((-self.L / 2, -self.Wd / 2), self.L, self.Wd, fill=False,
                                    ls="--", ec="0.55", lw=1.2))
        ax0.text(0, 0, "着座位置", ha="center", va="center", color="0.45", fontsize=9)
        for s in (1, -1):
            ax0.plot([front, back], [s * recv_w / 2] * 2, color="0.15", lw=3)
        ax0.plot([back, back], [-recv_w / 2, recv_w / 2], color="0.15", lw=3)
        ax0.text(front - 4, recv_w / 2 + 6, "開口部", ha="right", va="bottom", fontsize=9)
        ax0.text(back + 4, 0, "奥の当て", ha="left", va="center", fontsize=9, rotation=90)
        ax0.plot(self.p[:, 0], self.p[:, 1], color="tab:blue", lw=1.2)
        ax0.set_xlim(min(self.p[:, 0].min() - self.L / 2, front) - 15, back + 25)
        ax0.set_ylim(-recv_w / 2 - 25, recv_w / 2 + 25)
        ax0.set_aspect("equal")
        ax0.set_xlabel("x: 差し込み方向 [mm]（着座位置 = 0）", fontsize=9)
        ax0.set_ylabel("y: 左右 [mm]", fontsize=9)
        ax0.set_title("(a) 上から見たホルダーの動き（青線: 中心の軌跡、赤: 現在のホルダー）", fontsize=10)

        for k, (lab, c) in enumerate((("x: 差し込み方向", "tab:blue"), ("y: 左右", "tab:orange"),
                                      ("z: 高さ", "tab:green"))):
            ax1.plot(self.t, self.p[:, k], color=c, label=lab, lw=1.4)
        ax1.axhline(0, color="0.6", lw=0.8)
        ax1.set_xlim(0, self.dur)
        ax1.set_ylabel("位置 [mm]", fontsize=9)
        ax1.legend(fontsize=8, loc="lower right", ncol=3)
        ax1.set_title("(b) 受け部を基準にしたホルダー中心の位置（0 で着座）", fontsize=10)

        ax2.plot(self.t, self.pinch, color="tab:purple", lw=1.4)
        ax2.axhline(HOLD_PINCH_CM, color="0.4", ls="--", lw=0.9)
        ax2.text(self.dur * 0.01, HOLD_PINCH_CM + 0.05, "これより小さいと「つまんでいる」", fontsize=8,
                 color="0.3", va="bottom")
        ax2.set_xlim(0, self.dur)
        ax2.set_ylim(0, max(2.6, self.pinch.max() + 0.3))
        ax2.set_ylabel("距離 [cm]", fontsize=9)
        ax2.set_xlabel("差し込み区間の開始からの時間 [s]（右端 = 手を離した時刻）", fontsize=9)
        ax2.set_title("(c) 親指と人差し指の先の距離（把持と解放の判定に使用）", fontsize=10)
        for ax in (ax0, ax1, ax2):
            ax.tick_params(labelsize=8)

        fig.canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        self.bg = Image.open(buf).convert("RGB")
        hpx = self.bg.height

        def to_px(ax, x, y):
            X, Y = ax.transData.transform(np.column_stack([x, y])).T
            return X, hpx - Y

        self.to_px = to_px

        def ax_box(ax):
            b = ax.get_window_extent()
            return b.x0, hpx - b.y1, b.x1, hpx - b.y0

        self.ax_box = ax_box
        self.axes = (ax0, ax1, ax2)
        self.fig = fig

    def frame(self, tau: float) -> np.ndarray:
        """差し込み区間の開始から tau 秒の図。区間外は端で止め、灰色の帯で示す。"""
        im = self.bg.copy()
        d = ImageDraw.Draw(im)
        ax0, ax1, ax2 = self.axes
        tc = float(np.clip(tau, 0, self.dur))
        i = int(np.clip(np.searchsorted(self.t, tc), 0, len(self.t) - 1))
        x, y, _ = self.p[i]
        cx, cy = self.to_px(ax0, [x - self.L / 2, x + self.L / 2], [y - self.Wd / 2, y + self.Wd / 2])
        bx0, by0, bx1, by1 = self.ax_box(ax0)
        r = (max(cx[0], bx0), max(cy[1], by0), min(cx[1], bx1), min(cy[0], by1))
        if r[0] < r[2] and r[1] < r[3]:
            d.rectangle(r, outline=RED, width=3)
        X, Y = self.to_px(ax0, [x], [y])
        d.ellipse((X[0] - 6, Y[0] - 6, X[0] + 6, Y[0] + 6), fill=RED)
        for ax, ys in ((ax1, self.p[i]), (ax2, [self.pinch[i]])):
            y0, y1 = ax.get_ylim()
            LX, LY = self.to_px(ax, [tc, tc], [y0, y1])
            d.line((LX[0], LY[0], LX[1], LY[1]), fill=RED, width=2)
            for v in ys:
                PX, PY = self.to_px(ax, [tc], [v])
                d.ellipse((PX[0] - 6, PY[0] - 6, PX[0] + 6, PY[0] + 6), fill=RED)
        if tau < 0 or tau > self.dur:
            msg = "差し込み区間の前" if tau < 0 else "手を離した後"
            bx0, by0, bx1, _ = self.ax_box(ax0)
            f = ImageFont.truetype(FONT, 16)
            tw = d.textlength(msg, font=f)
            d.rectangle((bx1 - tw - 14, by0 + 4, bx1 - 4, by0 + 28), fill=(110, 110, 110))
            d.text((bx1 - tw - 9, by0 + 5), msg, font=f, fill=(255, 255, 255))
        return np.asarray(im)


if __name__ == "__main__":
    import imageio.v2 as imageio
    p = TrajectoryPanel("H01", "trial_013")
    imageio.imwrite("/tmp/chk/panel.png", np.concatenate([p.frame(0.5), p.frame(p.dur * 0.8)], 1))
