# -*- coding: utf-8 -*-
r"""HUD 큰 글자(16px 칸) 원본 꼴 재현 (2026-10-19) — hud_words 의 나눔 축소 방식 폐기(사용자 «완벽하게 쓰레기»)
  원본 실측: 획 2px = 단계 3, 획 둘레 1px(8방향) = 단계 1, 단계 2 거의 없음, 한자 한 자 ≈14×14 를 16px 간격.
  꼴 A = 갈무리7 비트맵을 정확히 2배(획 2px) · 꼴 B = 갈무리11 Bold.
  word(text, w, h, kind) → 단계 배열(0/1/3), 음절 간격 16(넘치면 15·14), 공백 6px.
  python tools/hud_px.py → work/gfx/prev_hud_px.png (대표 문구 원본 | A | B, 4배)
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hud as H  # noqa: E402
import pixfont as PF  # noqa: E402


def syllable(ch, kind):
    if kind in ('G9', 'G7'):                                          # 칸이 좁을 때 작은 비트맵(축소 아님)
        return PF.cell(ch, 9 if kind == 'G9' else 7)
    if kind == 'A':
        g = PF.cell(ch, 7)
        xs = np.nonzero(g.max(0))[0]
        g = g[:, :xs.max() + 1] if len(xs) else g                    # 오른쪽 빈 열 잘라 2배(14px 안)
        return np.kron(g, np.ones((2, 2)))
    return PF.cell(ch, 11, bold=True)


PITCH = {'A': (16, 15, 14), 'B': (13, 12), 'G9': (11, 10, 9), 'G7': (9, 8)}                         # B 는 글자(11px)+2 간격 — 16 이면 성김


def word(text, w, h, kind='A', align='c'):
    """글자 0/1 → 둘레 1 · 획 3. 공백 = 6px. B 가 칸에 안 들어가면 갈무리9 → 7 로 내려감"""
    if not text.strip():
        return np.zeros((h, w), np.uint8)
    if kind in ('B', 'S'):                                            # S = 작은 칸(높이 14 이하) 묶음 → 갈무리9 부터(한 문구 안 크기 고르게)
        for k in (('B', 'G9', 'G7') if kind == 'B' else ('G9', 'G7')):
            lv = word(text, w, h, k + '!', align)
            if lv is not None:
                return lv
        return word(text, w, h, 'G7', align)
    strict = kind.endswith('!'); kind = kind.rstrip('!')
    for pitch in PITCH[kind]:
        cols = []; x = 0
        for ch in text:
            if ch == ' ':
                x += 6 if kind == 'A' else 5 if kind == 'B' else 3; continue
            g = syllable(ch, kind)
            cols.append((x, g)); x += pitch if g.shape[1] <= pitch else g.shape[1]
        total = cols[-1][0] + cols[-1][1].shape[1] if cols else 0
        if total <= (w - 2 if kind in ('A', 'B') else w):          # 작은 글자는 둘레가 칸 끝에 조금 잘려도 크기를 지킴(한 문구 안 크기 고르게)
            break
    else:
        if strict:
            return None
    gh = max(g.shape[0] for _, g in cols)
    if strict and gh + 2 > h:
        return None
    ink = np.zeros((h, w), bool)
    if isinstance(align, tuple):                                     # ('c', 가운데 x) — 조각 밖까지 이어진 판의 가운데에 맞출 때
        x0 = int(round(align[1] - total / 2))
    else:
        x0 = 1 if align == 'l' else (w - 1 - total if align == 'r' else (w - total) // 2)
    y0 = (h - gh) // 2
    for x, g in cols:
        gy = y0 + (gh - g.shape[0])
        sub = ink[gy:gy + g.shape[0], x0 + x:x0 + x + g.shape[1]]
        sub |= g[:sub.shape[0], :sub.shape[1]] > 0
    pad = np.pad(ink, 1); ring = np.zeros_like(pad)                  # np.roll 이 반대쪽 가장자리로 넘어가지 않게 1칸 덧대고 계산
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ring |= np.roll(np.roll(pad, dy, 0), dx, 1)
    ring = ring[1:-1, 1:-1]
    out = np.zeros((h, w), np.uint8); out[ring] = 1; out[ink] = 3
    return out


def put(px, u, v, lay, lv):
    h, w = lv.shape
    reg = px[v:v + h, u:u + w]
    keep = (reg & 3) if lay == 'H' else (reg & 12)
    reg[:] = keep | ((lv << 2) if lay == 'H' else lv)


# 대표 문구: (보기 이름, [(조각 id, 한국어, 맞춤)])
SAMPLES = [
    ('3D05C', [(364, '정지', 'c'), (123, ' 신호', 'l')]),
    ('3C90C', [(373, '출발', 'c'), (358, '진', 'c'), (374, '행', 'c')]),
    ('3CA50', [(385, '제한', 'c'), (362, ' 속도 초과', 'l')]),
    ('3CAB0', [(390, '알림등 확인 실수', ('c', 64)), (96, '', 'l')]),   # 빨간 상자(390+ス 128px) 가운데          # 수 까지 앞 조각(112px)에 가운데로, ス 조각은 비움(이 문구 전용)
    ('3CC54', [(365, '정', 'c'), (108, '차', 'c'), (370, '역', 'c'), (378, ' 통', 'r'), (363, '과', 'l')]),
    ('3DB54', [(50, '속도 제한', 'c')]),
    ('3E664', [(366, '정지 위치', 'c')]),
    ('3E724', [(381, '비상', 'c'), (376, ' 제동', 'l'), (365, ' 정', 'c'), (108, '차', 'c')]),
]


def preview():
    import json
    pac = open(H.PAC, 'rb').read(); exe = open(H.EXE, 'rb').read()
    px0, clut = H.load(pac)
    P = {p[0]: p for p in json.load(open(os.path.join(ROOT, 'work', 'hud_pieces.json')))}
    gs = {g[0]['o']: g for g in H.groups(H.records(exe))}
    rows = []
    for name, items in SAMPLES:
        g = gs[int(name, 16)]
        imgs = [H.compose(px0, clut, g)]
        for kind in 'AB':
            px = px0.copy()
            for i, t, al in items:
                _, u, v, w, h, lay = P[i][:6]
                put(px, u, v, lay, word(t, w, h, kind, al))
            imgs.append(H.compose(px, clut, g))
        hh = max(a.shape[0] for a in imgs)
        row = []
        for a in imgs:
            row += [np.pad(a, ((0, hh - a.shape[0]), (0, 0), (0, 0))), np.full((hh, 4, 4), 255, np.uint8)]
        rows.append(np.concatenate(row, 1))
    W = max(r.shape[1] for r in rows)
    sh = np.concatenate([np.pad(r, ((2, 2), (0, W - r.shape[1]), (0, 0))) for r in rows], 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_hud_px.png')
    Image.fromarray(sh).resize((sh.shape[1] * 4, sh.shape[0] * 4), Image.NEAREST).save(p)
    print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
