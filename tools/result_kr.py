# -*- coding: utf-8 -*-
r"""운전 결과 화면 제목 그림 한글화 (2026-10-19)
  運転終了 = TXS 98DD800·991E800(VRAM 640,0 256×512, 같은 제목) 4bpp 보기 u0~149 × v481~511.
    원본 꼴 = 14px 글꼴(획 + 둘레 1px) 2배, 둘레 = 색인 1, 채움 = 줄마다 다른 색인(4→A 위에서 아래로, 맨 아래 3) → 줄별 색인표 RAMP 를 원본에서 읽음.
    → 갈무리11 Bold 1배에서 둘레 계산 후 2배, 채움은 같은 줄의 원본 색인.
  全区間走破 = TXS 98DC000(VRAM 384,256 32×32) 4bpp 128×32 의 v0~24(아래 ALL ROUND CLEAR 는 영어 그대로).
    둘레 색인 8, 채움 = 원본 줄별 최빈 색인(위 6 → 아래 F).
  GAME OVER(같은 띠 u240~) 는 영어라 그대로.
  python tools/result_kr.py → work/gfx/prev_result.png
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pixfont as PF  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
# (TXS 들, 그림 폭 halfword, 높이, 영역 u0,u1,v0,v1, 둘레 색인, 한국어, (갈무리 크기, 굵게)) — 전부 2배
ITEMS = [([0x98DD800, 0x991E800], 256, 512, (0, 150, 481, 512), 1, '운전 종료', (11, True)),
         ([0x98DC000], 32, 32, (0, 128, 0, 25), 8, '전 구간 완주', (9, False))]   # 배지는 높이 25 → 갈무리9 2배


def load(pac, off, w, h):
    a = off + 0x800; b = np.frombuffer(pac[a:a + w * h * 2], np.uint8).reshape(h, w * 2)
    return a, np.stack([b & 15, b >> 4], -1).reshape(h, w * 4).copy()


def encode(px):
    h, w = px.shape; p = px.reshape(h, w // 2, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def ramp(reg, edge):
    """줄마다 채움 색인(0·둘레·F 뺀 최빈) — 없는 줄은 가까운 줄 값"""
    out = [None] * reg.shape[0]
    for y, row in enumerate(reg):
        v = row[(row != 0) & (row != edge) & (row != 15)]           # F = 배지 가운데 밝은 띠(획이 가는 한글에선 점으로 보임) → 뺌
        if len(v):
            out[y] = int(np.bincount(v, minlength=16).argmax())
    idx = [y for y, v in enumerate(out) if v is not None]
    v = reg[(reg != 0) & (reg != edge) & (reg != 15)]                  # 2026-10-19 사용자 «단색» — 줄별 색 대신 전체 최빈 하나
    one = int(np.bincount(v, minlength=16).argmax())
    return [one] * len(out)


def make(px, box, edge, text, font, scale=2):
    px = px.copy(); u0, u1, v0, v1 = box
    reg = px[v0:v1, u0:u1]; H, W = reg.shape
    ys = np.nonzero(reg.any(1))[0]; top, bot = ys.min(), ys.max() + 1        # 원본 잉크(둘레 포함) 세로 범위
    rp = ramp(reg[top:bot], edge)
    g = PF.text(text, *font) > 0
    ink = np.pad(g, 1); ring = ink.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ring |= np.roll(np.roll(ink, dy, 0), dx, 1)
    ring &= ~ink
    ink = np.kron(ink, np.ones((scale, scale), bool)); ring = np.kron(ring, np.ones((scale, scale), bool))
    gh, gw = ink.shape
    assert gw <= W and gh <= H, ('넘침', text, gw, W, gh, H)
    x0 = int(np.nonzero(reg.any(0))[0].min()) if edge == 1 else (W - gw) // 2  # 運転終了 = 원본 왼쪽 끝에 붙임, 배지 = 가운데
    y0 = top + ((bot - top) - gh) // 2
    y0 -= y0 % scale - top % scale if scale > 1 else 0
    reg[top:bot] = 0
    for y in range(gh):
        ry = y0 + y
        if not 0 <= ry < H:
            continue
        f = rp[min(max(ry - top, 0), len(rp) - 1)]
        row = reg[ry, x0:x0 + gw]
        row[ring[y]] = edge; row[ink[y]] = f
    return px


def apply(pac):
    n = 0
    for offs, w, h, box, edge, text, sc in ITEMS:
        for off in offs:
            a, px = load(bytes(pac), off, w, h)
            enc = encode(make(px, box, edge, text, sc)); pac[a:a + len(enc)] = enc; n += 1
    return n


def preview():
    pac = open(PAC, 'rb').read(); rows = []
    for offs, w, h, box, edge, text, sc in ITEMS:
        _, px = load(pac, offs[0], w, h); new = make(px, box, edge, text, sc)
        u0, u1, v0, v1 = box
        g = lambda a: (a[v0:v1, u0:u1] * 17).astype(np.uint8)
        r = np.concatenate([g(px), np.full((v1 - v0, 4), 128, np.uint8), g(new)], 1)
        rows += [np.pad(r, ((0, 4), (0, 304 - r.shape[1])), constant_values=60)]
    sh = np.concatenate(rows, 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_result.png')
    Image.fromarray(sh).resize((sh.shape[1] * 4, sh.shape[0] * 4), Image.NEAREST).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
