"""ACT（Action Chunking with Transformers, Zhao et al. 2023）の最小実装。

学習時: CVAE エンコーダが（関節角, 正解の行動列）から潜在変数 z を推定し、
デコーダが（カメラ画像, 関節角, z）から chunk 分の行動列を出す。損失は L1 ＋ β·KL。
推論時: z = 0 として行動列を出し、時間的アンサンブルでつなぐ。
"""

import math

import torch
import torch.nn.functional as F
import torchvision
from torch import nn


def sine_pos_2d(h: int, w: int, dim: int, device) -> torch.Tensor:
    """(h*w, dim) の2次元正弦波位置埋め込み。"""
    d = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(d, device=device) / d))
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing="ij")
    ys, xs = ys.flatten()[:, None] * freq, xs.flatten()[:, None] * freq
    return torch.cat([ys.sin(), ys.cos(), xs.sin(), xs.cos()], dim=1)


def sine_pos_1d(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(n, dim, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe


class ACT(nn.Module):
    def __init__(self, n_cams=2, state_dim=8, action_dim=8, chunk=50, d_model=512, nhead=8,
                 enc_layers=4, dec_layers=4, ff=3200, dropout=0.1, latent_dim=32):
        super().__init__()
        self.chunk, self.latent_dim, self.n_cams = chunk, latent_dim, n_cams
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # (B,512,h/32,w/32)
        self.input_proj = nn.Conv2d(512, d_model, 1)
        self.cam_embed = nn.Embedding(n_cams, d_model)

        # CVAE エンコーダ
        self.cls = nn.Embedding(1, d_model)
        self.enc_state = nn.Linear(state_dim, d_model)
        self.enc_action = nn.Linear(action_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.cvae_encoder = nn.TransformerEncoder(layer, enc_layers)
        self.latent_proj = nn.Linear(d_model, 2 * latent_dim)

        # デコーダ側
        self.latent_out = nn.Linear(latent_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.extra_pos = nn.Embedding(2, d_model)  # z と関節角のトークン
        self.transformer = nn.Transformer(d_model, nhead, enc_layers, dec_layers, ff, dropout,
                                          batch_first=True)
        self.query = nn.Embedding(chunk, d_model)
        self.head = nn.Linear(d_model, action_dim)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode_latent(self, state, actions, is_pad):
        B = state.shape[0]
        tok = torch.cat([self.cls.weight[None].expand(B, -1, -1), self.enc_state(state)[:, None],
                         self.enc_action(actions)], dim=1)
        tok = tok + sine_pos_1d(tok.shape[1], tok.shape[2], tok.device)[None]
        pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=state.device), is_pad], 1)
        h = self.cvae_encoder(tok, src_key_padding_mask=pad)[:, 0]
        mu, logvar = self.latent_proj(h).chunk(2, dim=-1)
        return mu, logvar

    def forward(self, images, state, actions=None, is_pad=None):
        """images: (B, n_cams, 3, H, W) in [0,1]、state: (B, state_dim) 正規化済み。"""
        B = state.shape[0]
        if actions is not None:
            mu, logvar = self.encode_latent(state, actions, is_pad)
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            z = torch.zeros(B, self.latent_dim, device=state.device)

        imgs = (images.flatten(0, 1) - self.img_mean) / self.img_std
        feat = self.input_proj(self.backbone(imgs))  # (B*n, d, h, w)
        _, d, h, w = feat.shape
        pos = sine_pos_2d(h, w, d, feat.device)
        feat = feat.flatten(2).transpose(1, 2).view(B, self.n_cams, h * w, d)
        feat = feat + pos[None, None] + self.cam_embed.weight[None, :, None]
        src = torch.cat([torch.stack([self.latent_out(z), self.state_proj(state)], 1)
                         + self.extra_pos.weight[None], feat.flatten(1, 2)], dim=1)
        tgt = self.query.weight[None].expand(B, -1, -1)
        out = self.transformer(src, tgt)
        return self.head(out), mu, logvar


def act_loss(pred, target, is_pad, mu, logvar, kl_weight=10.0):
    l1 = (F.l1_loss(pred, target, reduction="none") * (~is_pad)[..., None]).sum() / \
        ((~is_pad).sum() * pred.shape[-1])
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
    return l1 + kl_weight * kl, {"l1": l1.item(), "kl": kl.item()}
