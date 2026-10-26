"""Tracking Streamer用のArUcoマーカー印刷シート（A4 PDF）を生成する。

Tracking Streamerは DICT_4X4_50 の ID 0〜19 を認識し、全マーカーに共通の
実寸（黒い外枠の一辺、白余白を含まない）を設定のスライダーで指定する。
スライダーは1 cm刻みなので、マーカー一辺は整数cmにする。

    python generate_aruco_sheet.py --size-mm 50 --ids 0 1 2 3
"""

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

DICTIONARY = cv2.aruco.DICT_4X4_50
MODULES = 6  # 4x4のデータ部＋黒枠1マスずつ


def marker_png(marker_id: int, pixels: int) -> ImageReader:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"marker {marker_id} の画像化に失敗")
    return ImageReader(io.BytesIO(buf.tobytes()))


def draw_sheet(out: Path, ids: list[int], size_mm: float) -> None:
    page_w, page_h = A4
    c = canvas.Canvas(str(out), pagesize=A4)
    c.setTitle(f"ArUco DICT_4X4_50 {size_mm:.0f}mm")

    quiet = min(size_mm / MODULES * 2, 15)  # 白余白は2マス分（最大15 mm）
    cell = size_mm + 2 * quiet
    gap = 12
    y_top = page_h / mm - 40
    y_bottom = 35
    cols = max(1, int((page_w / mm - 20 + gap) // (cell + gap)))
    rows = max(1, int((y_top - y_bottom + gap + 6) // (cell + gap + 6)))
    if cell > page_w / mm - 20 or cell > y_top - y_bottom:
        raise ValueError(f"{size_mm:.0f} mm のマーカーはA4に収まらない")
    per_page = cols * rows
    grid_w = cols * cell + (cols - 1) * gap
    x0 = (page_w / mm - grid_w) / 2

    pixels = 600
    for i, marker_id in enumerate(ids):
        slot = i % per_page
        if slot == 0:
            if i:
                draw_scale_bar(c, x0)
                c.showPage()
            draw_header(c, x0, page_h, size_mm)
        col, row = slot % cols, slot // cols
        cx = x0 + col * (cell + gap)
        cy = y_top - (row + 1) * cell - row * (gap + 6)

        c.setDash(2, 2)
        c.setLineWidth(0.3)
        c.setStrokeGray(0.6)
        c.rect(cx * mm, cy * mm, cell * mm, cell * mm)
        c.setDash()

        c.drawImage(marker_png(marker_id, pixels),
                    (cx + quiet) * mm, (cy + quiet) * mm,
                    size_mm * mm, size_mm * mm)

        c.setFillGray(0.3)
        c.setFont("Helvetica", 7)
        c.drawString((cx + 1) * mm, (cy + 1.5) * mm,
                     f"ID {marker_id}   {size_mm:.0f} mm")
        c.setFillGray(0)

    draw_scale_bar(c, x0)
    c.showPage()
    c.save()


def draw_header(c: canvas.Canvas, x0: float, page_h: float, size_mm: float) -> None:
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0 * mm, (page_h / mm - 18) * mm,
                 f"ArUco DICT_4X4_50  /  black square = {size_mm:.0f} mm")
    c.setFont("Helvetica", 8)
    c.drawString(x0 * mm, (page_h / mm - 25) * mm,
                 "Print at 100% (Actual size). Measure the 100 mm bar below. "
                 f"Set Tracking Streamer marker size to {size_mm / 10:.0f} cm.")


def draw_scale_bar(c: canvas.Canvas, x0: float) -> None:
    bar_y = 22
    c.setStrokeGray(0)
    c.setLineWidth(0.8)
    c.line(x0 * mm, bar_y * mm, (x0 + 100) * mm, bar_y * mm)
    for t in range(0, 101, 10):
        h = 4 if t % 50 == 0 else 2
        c.line((x0 + t) * mm, bar_y * mm, (x0 + t) * mm, (bar_y + h) * mm)
    c.setFont("Helvetica", 8)
    c.drawString(x0 * mm, (bar_y - 5) * mm, "0")
    c.drawString((x0 + 96) * mm, (bar_y - 5) * mm, "100 mm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--size-mm", type=float, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if any(not 0 <= i <= 19 for i in args.ids):
        parser.error("Tracking StreamerはID 0〜19のみ認識する")
    if args.size_mm % 10:
        parser.error("アプリのスライダーが1 cm刻みのため、整数cmにする")

    out = args.out or Path(__file__).resolve().parent.parent / "markers" / \
        f"aruco_4x4_50_id{'-'.join(map(str, args.ids))}_{args.size_mm:.0f}mm.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_sheet(out, args.ids, args.size_mm)
    print(out)


if __name__ == "__main__":
    main()
