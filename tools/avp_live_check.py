"""Vision Proから手首とArUcoマーカーの受信状況を一定時間表示する確認用スクリプト。

    python avp_live_check.py WNAN-8809 --seconds 60     # Remote（ルームコード）
    python avp_live_check.py <VisionProのIP>              # Local（IP）

ファイルへの保存は行わない。
"""

import argparse
import time

import numpy as np
from avp_stream import VisionProStreamer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Vision ProのIP、またはRemoteのルームコード")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--relay-only", action="store_true",
                        help="Remote時にTURN中継のみを使う（直接経路が失敗する環境向け）")
    args = parser.parse_args()

    s = VisionProStreamer(ip=args.target, record=False, relay_only=args.relay_only)
    t_end = time.time() + args.seconds
    last_wrist = None
    while time.time() < t_end:
        time.sleep(args.interval)
        data = s.get_latest()
        if data is None:
            print("hand: 未受信")
            continue
        wrist = np.asarray(data.right.wrist)
        moved = "" if last_wrist is None else \
            f" moved={np.linalg.norm(wrist[:3, 3] - last_wrist[:3, 3]) * 1000:.1f}mm"
        last_wrist = wrist
        x, y, z = wrist[:3, 3]
        print(f"right wrist xyz=({x:+.3f},{y:+.3f},{z:+.3f}) m{moved}")

        markers = s.get_markers()
        if not markers:
            print("  markers: なし")
        for marker_id, info in sorted(markers.items()):
            px, py, pz = info["pose"][:3, 3]
            print(f"  marker {marker_id}: xyz=({px:+.3f},{py:+.3f},{pz:+.3f}) "
                  f"tracked={info['is_tracked']} fixed={info['is_fixed']}")


if __name__ == "__main__":
    main()
