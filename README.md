# 病理標本ホルダーの差し込み動作の模倣学習

Apple Vision Pro で収録した人の実演から、スキャナーの受け部を基準にしたホルダーの軌跡を作り、MuJoCo 上の Franka Emika Panda で再生して、ACT（Action Chunking with Transformers）の方策を学習するコード一式です。

![全体の流れ](docs/figures/fig1_pipeline.png)


## 結果

| 項目 | 値 |
|---|---|
| 人の実演（成功試行） | 22 本 |
| ロボットでの再生の着座誤差（中央値） | 0.3 mm |
| 学習データ（水増し後） | 683 エピソード |
| ACT の成功率（学習に使っていない装置の位置・開始位置、100 回） | 76% |

![ACT の実行例](docs/figures/fig6_rollout.png)

## 構成

```text
tools/     収録（record_demo.py）、区間の確認、一括処理、図・動画・PDF の作成
retarget/  人の手の動き → 受け部基準のホルダー軌跡（human_to_holder.py）
           ホルダー軌跡 → Panda の関節角（holder_to_panda.py、mink の逆運動学）
sim/       装置とロボットのシーン（scene.py）、学習・評価用の環境（env.py）、学習データの作成（make_dataset.py）
policy/    ACT の実装（act.py）、学習（train.py）、評価（eval.py）
config/    収録セッションの設定、装置の実測寸法、着座姿勢の基準解
```

## 環境

Python 3.12、Linux、NVIDIA GPU（学習時）で動作を確認しました。

```bash
pip install -r requirements.txt
git clone https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
```

## 実行コード

収録データは個人情報を含む可能性があるため、このリポジトリには入れていません。手順は次のとおりです。

```bash
# 1. 収録（Vision Pro の Tracking Streamer アプリと同じネットワークで）
python tools/record_demo.py <VisionProのIP> --session S01

# 2. セッションの全試行を、ホルダーの軌跡と Panda での再生に変換
python tools/process_session.py S01 --video-dir <iPad 動画のフォルダ>

# 3. 学習データを作る（装置の位置・開始位置・速さ・揺らぎを変えて水増し）
MUJOCO_GL=egl python sim/make_dataset.py --find-seat-reference   # 着座姿勢の基準解（1 回だけ）
MUJOCO_GL=egl python sim/make_dataset.py --per-demo 40 --out act_v7

# 4. ACT を学習して評価する
python policy/train.py --name act_v7 --data act_v7 --steps 40000 --release-tol-mm 2.0
MUJOCO_GL=egl python policy/eval.py --run act_v7 --n 100 --seed 2026
```

## 引用したソフトウェア

- [MuJoCo](https://github.com/google-deepmind/mujoco)、[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)（Panda のモデル）
- [mink](https://github.com/kevinzakka/mink)（逆運動学）
- [VisionProTeleop / avp_stream](https://github.com/Improbable-AI/VisionProTeleop)（Vision Pro の手の追跡の受信）
- ACT [Zhao et al., RSS 2023](https://tonyzhaozh.github.io/aloha/)
