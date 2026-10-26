"""iPad 動画に写るスキャナー側面のラベル（施設名・QR コード）をぼかす。

iPad は手持ちで画面内の位置が少しずつ動くため、基準フレームから切り出したラベルの見本を
各フレームでテンプレート照合して位置を求め、その範囲をぼかす。
見本は実行時に動画から作り、リポジトリには保存しない。
"""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

REF_VIDEO = "IMG_0833.MOV"
REF_TIME_S = 34.99
REF_BOX = (540, 715, 830, 900)  # 基準フレーム（1080×1920）でのラベル 4 枚の範囲 x0, y0, x1, y1
REF_SIZE = (1080, 1920)
MATCH_SCALE = 0.25
MARGIN = 0.12  # 見つけた範囲の外側にも、幅・高さのこの割合だけ広げてぼかす


def _gray(img: np.ndarray, s: float) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


class LabelBlur:
    def __init__(self, ipad_dir: Path):
        r = imageio.get_reader(ipad_dir / REF_VIDEO, "ffmpeg", input_params=["-ss", f"{REF_TIME_S:.3f}"])
        ref = r.get_next_data()
        r.close()
        assert ref.shape[1::-1] == REF_SIZE, ref.shape
        x0, y0, x1, y1 = REF_BOX
        self.template = ref[y0:y1, x0:x1]
        self.last = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        s = frame.shape[1] / REF_SIZE[0]
        k = MATCH_SCALE / s if s > MATCH_SCALE else 1.0
        tpl = _gray(self.template, s * k)
        res = cv2.matchTemplate(_gray(frame, k), tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        th, tw = tpl.shape
        box = (loc[0] / k, loc[1] / k, (loc[0] + tw) / k, (loc[1] + th) / k)
        if score < 0.4 and self.last is not None:
            box = self.last
        self.last = box
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        h, w = frame.shape[:2]
        x0, y0 = int(max(x0 - mx, 0)), int(max(y0 - my, 0))
        x1, y1 = int(min(x1 + mx, w)), int(min(y1 + my, h))
        out = frame.copy()
        roi = out[y0:y1, x0:x1]
        small = cv2.resize(roi, (max(roi.shape[1] // 16, 1), max(roi.shape[0] // 16, 1)), interpolation=cv2.INTER_AREA)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(cv2.resize(small, roi.shape[1::-1], interpolation=cv2.INTER_LINEAR),
                                             (0, 0), max(roi.shape[1] / 40, 1))
        return out
