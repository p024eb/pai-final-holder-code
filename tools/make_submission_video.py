"""提出動画（約 1 分強）を組み立てる。

    python tools/make_submission_video.py --ipad-dir ../physicalAI/本番01 --run act_v7

構成: タイトル → 人の実演（iPad＋変換した軌跡の図）→ 人と Panda の再生の比較 → 方法の要点
      → ACT の実行例（成功 4 例＋失敗 1 例）→ 結果のまとめ
出力: data/video/submission.mp4（1280×720, 30 fps）
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from privacy import LabelBlur  # noqa: E402
from trajectory_panel import TrajectoryPanel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H, FPS = 1280, 720, 30


def font(size):
    return ImageFont.truetype(FONT, size)


def canvas() -> Image.Image:
    return Image.new("RGB", (W, H), (18, 18, 22))


def paste_fit(bg: Image.Image, img: np.ndarray, box) -> None:
    x0, y0, x1, y1 = box
    im = Image.fromarray(img)
    s = min((x1 - x0) / im.width, (y1 - y0) / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)))
    bg.paste(im, (x0 + (x1 - x0 - im.width) // 2, y0 + (y1 - y0 - im.height) // 2))


def caption(bg: Image.Image, text: str, sub: str = "") -> None:
    d = ImageDraw.Draw(bg)
    d.rectangle((0, H - 92, W, H), fill=(0, 0, 0))
    d.text((24, H - 84), text, font=font(30), fill=(255, 255, 255))
    if sub:
        d.text((24, H - 40), sub, font=font(20), fill=(200, 200, 200))


def text_card(lines, seconds):
    bg = canvas()
    d = ImageDraw.Draw(bg)
    y = 120
    for text, size in lines:
        d.text((80, y), text, font=font(size), fill=(255, 255, 255))
        y += int(size * 1.7)
    frame = np.asarray(bg)
    return [frame] * int(seconds * FPS)


def read_frames(path: Path, start_s=0.0, seconds=None):
    params = ["-ss", f"{max(start_s, 0):.3f}"]
    r = imageio.get_reader(path, "ffmpeg", input_params=params, output_params=["-r", str(FPS)])
    out = []
    for f in r:
        out.append(f)
        if seconds is not None and len(out) >= seconds * FPS:
            break
    r.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ipad-dir", type=Path, required=True)
    ap.add_argument("--run", default="act_v7")
    ap.add_argument("--eval-tag", default="_n50")
    ap.add_argument("--session", default="H01")
    ap.add_argument("--trial", default="trial_013")
    ap.add_argument("--success-eps", type=int, nargs="*", default=[0, 1, 4, 5])
    ap.add_argument("--failure-ep", type=int, default=2)
    args = ap.parse_args()

    out_dir = ROOT / "data" / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config" / "sessions.json").read_text())["use"][args.session]
    seg = json.loads((ROOT / "data" / "processed" / args.session / f"{args.trial}_holder.json").read_text())
    ev = json.loads((ROOT / "runs" / args.run / f"eval{args.eval_tag}.json").read_text())
    final = ROOT / "runs" / args.run / "eval_n100.json"
    final = json.loads(final.read_text()) if final.exists() else ev
    frames = []

    # タイトル
    frames += text_card([("病理標本ホルダーの差し込みを模倣学習する", 46),
                         ("Vision Pro による人の実演 → 受け部基準の軌跡 → Panda（MuJoCo）→ ACT", 26),
                         ("", 20),
                         ("Physical AI 応用1 最終課題", 24)], 5)

    # 人の実演（iPad）と、変換した軌跡の図（動画の時刻に合わせて赤い点が動く）
    ipad = args.ipad_dir / cfg["ipad_video"][args.trial]
    offset = cfg["video_offset_s"][args.trial]
    pre, post = 2.0, 2.0
    blur = LabelBlur(args.ipad_dir)
    clip = [blur(f) for f in read_frames(ipad, seg["t_start_s"] - offset - pre, seg["duration_s"] + pre + post)]
    panel = TrajectoryPanel(args.session, args.trial, width=820, height=620)
    for k, f in enumerate(clip):
        bg = canvas()
        paste_fit(bg, f, (10, 10, 440, H - 100))
        paste_fit(bg, panel.frame(k / FPS - pre), (450, 4, W - 6, H - 96))
        caption(bg, "① 人の実演を Vision Pro で収録（左: 同時に撮った iPad 動画、右: 求めたホルダーの軌跡）",
                "手首と指の関節の位置から、つまんだ位置・着座・解放を自動で検出し、受け部を基準にした軌跡にした")
        frames.append(np.asarray(bg))

    # 人と Panda の再生の比較
    cmp_path = ROOT / "data" / "sim" / args.session / f"{args.trial}_compare.mp4"
    if not cmp_path.exists():
        subprocess.run([sys.executable, str(ROOT / "tools" / "compare_video.py"), args.session, args.trial,
                        "--video-dir", str(args.ipad_dir)], check=True, cwd=ROOT)
    for f in read_frames(cmp_path):
        bg = canvas()
        paste_fit(bg, f, (0, 0, W, H - 92))
        caption(bg, "② ホルダーの軌跡を Panda アームの動きに変換し、MuJoCo で再生",
                "人の成功 22 本すべてが着座（ずれの中央値 0.3 mm）。手先には吸着パッド 2 個の付いたパドルを付けた")
        frames.append(np.asarray(bg))

    # 方法の要点
    frames += text_card([("③ ACT の学習", 40),
                         ("・22 本の軌跡から、装置の位置（±3 cm）・開始位置・速さを変えて 683 エピソードを作成", 24),
                         ("・観測: 手前上方のカメラ＋手先カメラの画像、関節角　行動: 7 関節の目標角＋解放", 24),
                         ("・実行時に揺らぎを加え（DART）、ずれた状態から戻る行動も学ばせた", 24),
                         ("・解放のラベルは「着座位置から 2 mm 以内に入ったら」で決めた", 24),
                         ("・逆運動学の解の枝をそろえ、腕の姿勢を一貫させた", 24)], 9)

    # ACT の実行例
    eps = {e["episode"]: e for e in ev["episodes"]}
    vid_dir = ROOT / "runs" / args.run / "videos"
    for k in args.success_eps + [args.failure_ep]:
        e = eps[k]
        ok = e["success"]
        head = "成功" if ok else "失敗例"
        sub = (f"装置のずれ 前後 {e['receiver_dx'] * 1000:+.0f} mm・左右 {e['receiver_dy'] * 1000:+.0f} mm"
               f"　→　着座のずれ {e['seat_err_mm']:.1f} mm"
               + ("" if ok else "（基準 5 mm を超えた）"))
        for f in read_frames(vid_dir / f"ep_{k:02d}{args.eval_tag}.mp4"):
            bg = canvas()
            paste_fit(bg, f, (0, 40, W, H - 100))
            caption(bg, f"④ 学習した ACT で差し込み（{head}）　左: 横から　右: 手先カメラ", sub)
            frames.append(np.asarray(bg))

    # 結果のまとめ
    fe = final["episodes"]
    n_ok = sum(e["success"] for e in fe)
    ok_err = np.median([e["seat_err_mm"] for e in fe if e["success"]])
    frames += text_card([("結果", 44),
                         (f"・ACT の成功率 {n_ok}/{len(fe)}（{100 * n_ok / len(fe):.0f}%）", 30),
                         ("　学習に使っていない乱数で、装置の位置と開始位置を決めた", 22),
                         (f"・成功時の着座のずれ 中央値 {ok_err:.1f} mm、光学系への接触 "
                          f"{sum(e['optics_contact'] for e in fe)} 回", 26),
                         ("・限界: はめ込み・スライドの配置・実機での検証は今後の課題", 26)], 8)

    w = imageio.get_writer(out_dir / "submission.mp4", fps=FPS, codec="libx264", quality=8,
                           macro_block_size=8)
    for f in frames:
        w.append_data(f)
    w.close()
    print(out_dir / "submission.mp4", f"{len(frames) / FPS:.1f} 秒")


if __name__ == "__main__":
    main()
