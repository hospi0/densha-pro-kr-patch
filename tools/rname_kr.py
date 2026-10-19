# -*- coding: utf-8 -*-
r"""HUD 위 가운데 노선명·차종(TXS 사각형 (416,384,32,32) = 4bpp 128×32, 78장) 한글화 (2026-10-19)
  번역 work/text/rname_kr.tsv (off·원문 2줄·번역 2줄). 색인: 0 = 투명, 3 = 글자, 2 = 가장자리.
  배치(원본 실측): 1줄 0~8줄, 2줄 10~18줄, 줄마다 원본 잉크 가운데에 맞춤. 20줄~ 다른 자료(손대지 않음).
  글꼴 = 갈무리9(넘치면 갈무리7) — 나눔은 8~9px 에서 뭉개짐.
  python tools/rname_kr.py → work/gfx/prev_rname_N.png (원본|한글, 3배)
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import desc_kr as D  # noqa: E402
import pixfont as PF  # noqa: E402

TSV = os.path.join(ROOT, 'work', 'text', 'rname_kr.tsv')
W, H = 128, 32
LINES = ((0, 9), (10, 9))                      # 맨 윗줄, 줄 높이
H_INK = 8
PREV_PAL = np.array([[16, 16, 24], [60, 60, 120], [70, 100, 200], [96, 144, 248]] + [[255, 0, 255]] * 12, np.uint8)


def rows():
    out = []
    for ln in open(TSV, encoding='utf-8').read().split('\n')[1:]:
        if ln:
            a = ln.split('\t'); out.append((int(a[0], 16), a[3], a[4]))
    return out


def load(pac, off):
    a = off + 0x800
    raw = np.frombuffer(pac[a:a + W * H // 2], np.uint8).reshape(H, W // 2)
    return np.stack([raw & 15, raw >> 4], -1).reshape(H, W).copy()


def encode(px):
    p = px.reshape(H, W // 2, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def make_one(px, l1, l2):
    new = px.copy(); new[:20] = 0
    for text, (y, lh) in zip((l1, l2), LINES):
        ink = np.nonzero((px[y:y + lh] > 0).any(0))[0]
        cx = (ink.min() + ink.max() + 1) / 2 if len(ink) else W / 2
        c = PF.text(text, 9)                       # 갈무리9, 칸 폭 넘치면 갈무리7(사용자: 작은 글자는 갈무리)
        if c.shape[1] > W - 2:
            c = PF.text(text, 7)
        assert c.shape[1] <= W - 2, '넘침 %r' % text
        h = c.shape[0]; up = 0
        hh, ww = c.shape
        x0 = int(round(cx - ww / 2)); x0 = max(1, min(W - 1 - ww, x0))
        y0 = y + (lh - h) // 2 - up
        y0 = max(y, y0); hh = min(hh, y + lh + 1 - y0)
        lv = np.zeros((hh, ww), np.uint8); cc = c[:hh]
        lv[cc >= 0.25] = 2; lv[cc >= 0.5] = 3
        reg = new[y0:y0 + hh, x0:x0 + ww]; reg[:] = np.maximum(reg, lv)
    return new


def build(pac):
    return [(off, px, make_one(px, a, b)) for off, a, b in rows() for px in [load(pac, off)]]


def apply(pac):
    n = 0
    for off, _, new in build(bytes(pac)):
        enc = encode(new); pac[off + 0x800:off + 0x800 + len(enc)] = enc; n += 1
    return n


def preview():
    pac = open(D.PAC, 'rb').read(); res = build(pac)
    for k in range(0, len(res), 26):
        tiles = []
        for off, a, b in res[k:k + 26]:
            ia = PREV_PAL[a[:20]]; ib = PREV_PAL[b[:20]]
            tiles.append(np.concatenate([ia, np.full((20, 3, 3), 255, np.uint8), ib], 1))
            tiles.append(np.full((3, W * 2 + 3, 3), 90, np.uint8))
        half = (len(tiles) + 1) // 2 // 2 * 2
        colA = np.concatenate(tiles[:half], 0); colB = np.concatenate(tiles[half:], 0)
        colB = np.pad(colB, ((0, colA.shape[0] - colB.shape[0]), (0, 0), (0, 0)))
        sh = np.concatenate([colA, np.full((colA.shape[0], 8, 3), 255, np.uint8), colB], 1)
        p = os.path.join(ROOT, 'work', 'gfx', 'prev_rname_%d.png' % (k // 26))
        Image.fromarray(sh).resize((sh.shape[1] * 3, sh.shape[0] * 3), Image.NEAREST).save(p)
    print('미리보기', os.path.join(ROOT, 'work', 'gfx', 'prev_rname_0~%d.png' % ((len(res) - 1) // 26)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
