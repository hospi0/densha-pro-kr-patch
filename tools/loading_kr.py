# -*- coding: utf-8 -*-
r"""로딩 화면 발차 안내판 머리 그림 한글화 (2026-10-19)
  TXS (384,0,64,256) 5사본(7FBF800·7FC8800·7FD1800·7FDA800·7FE3800, 같은 그림) 4bpp 256×256.
  今度|の発|車は = u224~255 × v16·32·48 (32×16 칸 셋, 화면에서 가로로 이어 붙음) → «다음 발차는» 96px 에 한 번 그려 세 칸으로 나눔.
    원본 꼴: 획 1px = 색인 4, 바탕 0 → 갈무리14(1px 획) 획 4.
  停車駅 = u208~255 v0~15 연두 판(색인 3) 위 어두운 글자(1) → «정차역» 갈무리11 Bold, 판 그대로.
  python tools/loading_kr.py → work/gfx/prev_loading.png
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pixfont as PF  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
COPIES = [0x7FBF800, 0x7FC8800, 0x7FD1800, 0x7FDA800, 0x7FE3800]
HEAD = '다음 발차는'
CELLS = [(224, 16), (224, 32), (224, 48)]          # 32×16 칸(가로로 이어짐)
PLATE = (208, 0, 48, 16)


def load(pac, off):
    a = off + 0x800; raw = np.frombuffer(pac[a:a + 0x8000], np.uint8).reshape(256, 128)
    return np.stack([raw & 15, raw >> 4], -1).reshape(256, 256).copy()


def encode(px):
    p = px.reshape(256, 128, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def line(text, size, bold=False, pitch=None, space=4):
    cols = []; x = 0
    for ch in text:
        if ch == ' ':
            x += space; continue
        g = PF.cell(ch, size, bold)
        cols.append((x, g)); x += (pitch or g.shape[1] + 1)
    h = max(g.shape[0] for _, g in cols); w = cols[-1][0] + cols[-1][1].shape[1]
    out = np.zeros((h, w), bool)
    for x, g in cols:
        out[:g.shape[0], x:x + g.shape[1]] |= g > 0
    return out


def make(px):
    px = px.copy()
    g = line(HEAD, 14, pitch=15, space=5)                    # 갈무리14, 원본처럼 1px 획
    canvas = np.zeros((16, 96), bool)
    x0 = (96 - g.shape[1]) // 2; y0 = (16 - g.shape[0]) // 2
    canvas[y0:y0 + g.shape[0], x0:x0 + g.shape[1]] = g
    for k, (u, v) in enumerate(CELLS):
        cell = np.zeros((16, 32), np.uint8); cell[canvas[:, k * 32:(k + 1) * 32]] = 4
        px[v:v + 16, u:u + 32] = cell
    u, v, w, h = PLATE
    g = line('정차역', 11, bold=True, pitch=14)
    reg = px[v:v + h, u:u + w]; reg[:] = 3                  # 판 전체 다시 칠함 — 맨 아랫줄 15 에 원본 停車駅 획 끝(1)이 남아 «찌꺼기»(사용자 스샷)
    x0 = (w - g.shape[1]) // 2; y0 = (h - g.shape[0]) // 2
    sub = reg[y0:y0 + g.shape[0], x0:x0 + g.shape[1]]; sub[g] = 1
    return px


def apply(pac):
    for off in COPIES:
        new = make(load(bytes(pac), off)); enc = encode(new)
        pac[off + 0x800:off + 0x800 + len(enc)] = enc
    return len(COPIES)


def preview():
    pac = open(PAC, 'rb').read(); a = load(pac, COPIES[0]); b = make(a)
    pal = np.array([[0, 0, 0], [24, 40, 24], [96, 140, 80], [150, 230, 110], [240, 240, 240]] + [[255, 0, 255]] * 11, np.uint8)
    def strip(px):
        head = np.concatenate([px[v:v + 16, u:u + 32] for u, v in CELLS], 1)
        u, v, w, h = PLATE
        return np.concatenate([head, np.zeros((16, 8), np.uint8), px[v:v + h, u:u + w]], 1)
    sh = np.concatenate([pal[strip(a)], np.full((4, 152, 3), 255, np.uint8), pal[strip(b)]], 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_loading.png')
    Image.fromarray(sh).resize((sh.shape[1] * 6, sh.shape[0] * 6), Image.NEAREST).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
