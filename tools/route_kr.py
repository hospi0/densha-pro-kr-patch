# -*- coding: utf-8 -*-
r"""운전 승무 노선 선택 큰 글자 한글화 (2026-10-18)
  그림 = TXS 7E54800 tpage B 의 u72~247 v0~191(48줄 띠 4개). 두 그림을 값 1~8 로 겹침:
    CLUT 3728(그림 A) 보이는 값 1·5·6 흰, 3·7·8 회색 / CLUT 3768(그림 B) 2·5·7 흰, 4·6·8 회색
    → (A, B) = (흰,0)1 (0,흰)2 (회,0)3 (0,회)4 (흰,흰)5 (흰,회)6 (회,흰)7 (회,회)8.
  레코드 = 34번 파일 0x936E~0x948E(20개): 줄마다 1~3 조각. 한국어는 줄 통째 한 조각(첫 레코드), 나머지 레코드 = 첫 레코드와 같게.
  줄 가운데 = x88(원본 조각들의 가운데). 모든 줄 같은 글자 높이(가장 긴 줄이 176 폭에 들어가게).
  python tools/route_kr.py → work/gfx/prev_route.png
"""
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import btn_kr as BT  # noqa: E402
import timetable_kr as TK  # noqa: E402
import tt_stations as TS  # noqa: E402

REGION = (72, 0, 176, 192)                    # tpage B 안 (u, v, w, h)
LINES = [  # (레코드들, 한국어)
    ([0x936E], '아키타・토호쿠신칸센'), ([0x937E], '호쿠호쿠・조에츠선'), ([0x938E], '야마노테선'),
    ([0x939E, 0x93AE, 0x93BE], '케이힌토호쿠선 북행'), ([0x93CE, 0x93DE, 0x93EE], '케이힌토호쿠선 남행'),
    ([0x93FE, 0x940E], '▼다음 노선'), ([0x941E, 0x942E], '▲이전 노선'), ([0x943E], '토카이도선'),
    ([0x944E], 'JR 사가노선'), ([0x945E, 0x946E], 'JR 쿄토선'), ([0x947E], '오사카순환선'), ([0x948E], '카고시마본선'),
]
CENTER = 88
ROWH = 32                                     # 새 띠 높이(글자 높이 + 여유)
ENC = {(1, 0): 1, (0, 1): 2, (2, 0): 3, (0, 2): 4, (1, 1): 5, (1, 2): 6, (2, 1): 7, (2, 2): 8}   # 1 = 흰, 2 = 회색
DEC = {v: k for k, v in ENC.items()}
CLUTS = (0x3728, 0x3768)


def rec(F, o):
    w = struct.unpack_from('<8H', F, o)
    return dict(o=o, u=w[0] & 0xFF, v=w[0] >> 8, x=TS.MK.s16(w[1]), w=w[3], h=w[4], tp=w[5], clut=w[6])


def levels(cov):
    return np.where(cov >= 0.5, 1, np.where(cov >= 0.2, 2, 0)).astype(np.uint8)


def make(pac, px=None):
    """px(timetable_kr 결과 화소)를 이어받아 노선 이름을 그림 → px, {레코드: (u, v, x, w, h, clut)}"""
    F = pac[TS.FILE34:TS.FILE34 + 0xE000]
    if px is None:
        px = TK.load_px(pac)
    u0, v0, W0, H0 = REGION; x0 = 256 + u0
    reg = px[v0:v0 + H0, x0:x0 + W0]
    assert reg.max() <= 8
    reg[:] = 0
    # 같은 높이로, 가장 긴 줄이 W0 에 들어가게
    for h in range(30, 14, -1):
        cov = {t: BT.cover(t, h, 0.7)[0] for _, t in LINES}
        if max(c.shape[1] for c in cov.values()) <= W0:
            break
    # 띠(ROWH) × 두 그림(A·B) 에 배치
    rows = H0 // ROWH
    cur = {(r, L): 0 for r in range(rows) for L in (0, 1)}
    out = {}
    A = np.zeros((H0, W0), np.uint8); B = np.zeros((H0, W0), np.uint8)
    for recs, t in sorted(LINES, key=lambda a: -cov[a[1]].shape[1]):
        lv = levels(cov[t]); hh, ww = lv.shape
        spot = min(((W0 - cur[k] - ww, k) for k in cur if cur[k] + ww <= W0), default=None)
        assert spot, '자리 모자람 ' + t
        (r, L) = spot[1]; u = cur[(r, L)]; cur[(r, L)] = u + ww + 1
        y = r * ROWH + (ROWH - hh) // 2
        (A if L == 0 else B)[y:y + hh, u:u + ww] = lv
        for o in recs:
            out[o] = (u0 + u, v0 + r * ROWH, CENTER - ww // 2, ww, ROWH, CLUTS[L])
    reg[:] = np.vectorize(lambda a, b: ENC.get((a, b), 0))(A, B).astype(np.uint8)
    return px, out, h


def patch34(F, out):
    F = bytearray(F)
    for o, (u, v, x, w, h, clut) in out.items():
        F[o] = u; F[o + 1] = v
        struct.pack_into('<h', F, o + 2, x)
        struct.pack_into('<H', F, o + 6, w)
        struct.pack_into('<H', F, o + 8, h)
        struct.pack_into('<H', F, o + 12, clut)
    return bytes(F)


def preview():
    import vram as V
    pac = open(TS.MK.PAC, 'rb').read(); F = pac[TS.FILE34:TS.FILE34 + 0xE000]
    orig = TK.load_px(pac); px, out, h = make(pac, orig.copy())
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '15.sav'))

    def pal(c):
        cx, cy = (c & 0x3F) * 16, c >> 6; cc = v[cy, cx:cx + 16].astype(int)
        return np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8), cc == 0

    def line(P, recs, geo):
        cv = np.zeros((48, 260, 3), np.uint8)
        for o in recs:
            u, vv, x, w, hh, c = geo(o)
            a = P[vv:vv + hh, 256 + u:256 + u + w]; p, tr = pal(c); m = ~tr[a]
            sub = cv[:hh, 40 + x:40 + x + w]; sub[m] = p[a][m]
        return cv
    rows = []
    for recs, t in LINES:
        a = line(orig, recs, lambda o: (lambda r: (r['u'], r['v'], r['x'], r['w'], r['h'], r['clut']))(rec(F, o)))
        b = line(px, recs, lambda o: out[o])
        rows.append(np.concatenate([a, np.full((48, 3, 3), 200, np.uint8), b], 1))
    im = np.concatenate(rows, 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_route.png')
    Image.fromarray(im).resize((im.shape[1] * 2, im.shape[0] * 2), Image.NEAREST).save(p)
    print('미리보기', p, '글자 높이', h)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
