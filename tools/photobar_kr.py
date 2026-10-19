# -*- coding: utf-8 -*-
r"""사진 보기 위 안내 띠 한글화 (2026-10-19) — 출처 = TXS AC03800 (VRAM 0,480 1024×32, 16비트 직접 색)
  왼쪽 0~216 = 버튼 그림(△ □ ○ × 등, 그대로), 글자 구간(원본 실측) SEGS 마다 다시 그림.
  원본 꼴 = 10px 글꼴 2배(획 2px, 높이 19) + 둘레 1px → 갈무리9 2배 + 둘레 1px. 글자 색 0xF8F8, 둘레 0x0808, 바탕 0(투명).
  구간 = 원본 글자 범위 안(メニューに戻る·セレクト:表示消す 는 한 구간으로).
  python tools/photobar_kr.py → work/gfx/prev_photobar.png
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pixfont as PF  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
TXS = 0xAC03800
FILL, EDGE = 0xF8F8, 0x0808
Y0, Y1 = 8, 31
SEGS = [(217, 324, ':다음 페이지'), (387, 492, ':이전 페이지'), (555, 733, ':메뉴로 돌아가기'), (756, 959, '셀렉트:표시 끄기')]


def load(pac):
    a = TXS + 0x800
    return a, np.frombuffer(pac[a:a + 1024 * 32 * 2], '<u2').reshape(32, 1024).copy()


def glyph(ch):
    if ch == ':':                                                   # 원본 쌍점 = 4×4 덩이 둘(2배 글꼴의 2×2)
        g = np.zeros((9, 2)); g[2:4] = 1; g[6:8] = 1
    else:
        g = PF.cell(ch, 9) if ord(ch) >= 0xAC00 else PF.text(ch, 9)
    return np.kron(g > 0, np.ones((2, 2), bool))


def line(t, w):
    for pitch in (20, 19, 18, 17):
        cols = []; x = 0
        for ch in t:
            if ch == ' ':
                x += 6; continue
            g = glyph(ch); cols.append((x, g))
            x += pitch if ord(ch) >= 0xAC00 else g.shape[1] + 4
        total = cols[-1][0] + cols[-1][1].shape[1]
        if total <= w - 2:
            break
    else:
        raise SystemExit('넘침 %r %d > %d' % (t, total, w - 2))
    return cols, total


def make(v):
    v = v.copy()
    for x0, x1, t in SEGS:
        w = x1 - x0; h = Y1 - Y0
        cols, total = line(t, w)
        ink = np.zeros((h, w), bool); gh = max(g.shape[0] for _, g in cols); y0 = (h - gh) // 2
        for x, g in cols:
            ink[y0 + gh - g.shape[0]:y0 + gh, 1 + x:1 + x + g.shape[1]] |= g
        pad = np.pad(ink, 1); ring = pad.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ring |= np.roll(np.roll(pad, dy, 0), dx, 1)
        ring = ring[1:-1, 1:-1] & ~ink
        reg = v[Y0:Y1, x0:x1]; reg[:] = 0
        reg[ring] = EDGE; reg[ink] = FILL
    return v


def apply(pac):
    a, v = load(bytes(pac)); new = make(v).astype('<u2').tobytes()
    pac[a:a + len(new)] = new
    return len(SEGS)


def preview():
    pac = open(PAC, 'rb').read(); _, v = load(pac); n = make(v)
    rgb = lambda a: np.stack([(a & 31) << 3, ((a >> 5) & 31) << 3, ((a >> 10) & 31) << 3], -1).astype(np.uint8)
    sh = np.concatenate([rgb(v[:, :960]), np.full((4, 960, 3), 255, np.uint8), rgb(n[:, :960])], 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_photobar.png')
    Image.fromarray(sh).resize((960 * 2, sh.shape[0] * 2), Image.NEAREST).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
