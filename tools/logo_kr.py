# -*- coding: utf-8 -*-
r"""HUD 아틀라스 4비트 로고 한글화 — 運転評価·ボーナス·完通 (2026-10-19)
  조각(hud_words SKIP 220·221·240·249) — 다른 레코드와 화소 안 나눔(검사함). 241·250(8×8 혜성 꼬리)은 글자 없음 → 그대로.
  글자 = 색인 2~6(은색 줄 색) + 1(어두운 획) + 둘레(運転評価 = F, 혜성 위 = 9·A·B 중 최빈).
  ① 글자 가리개 = 2~6 ∪ (그것에 닿은 1) ∪ (그 둘에 닿은 9·A·B) → 같은 줄 좌우 가까운 바탕(혜성) 색으로 메움(運転評価 는 바탕 0).
  ② 갈무리9 2배 + 오른쪽 1px 굵게 한글, 채움 = 원본 글자의 줄별 최빈 색인(은색 계조 그대로), 둘레 1px = 원본 둘레 색인.
  python tools/logo_kr.py → work/gfx/prev_logo.png (실제 CLUT 색)
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bonus_kr as BK  # noqa: E402
import hud as H  # noqa: E402

CORE = [2, 3, 4, 5, 6]
# (영역 u0,u1,v0,v1, 바탕 0?, [(한국어, 가운데 u, 가운데 v)])
LOGOS = [((0, 88, 256, 276), True, [('운전 평가', 44, 265)]),
         ((0, 72, 296, 320), False, [('보너스', 39, 307)]),
         ((0, 72, 320, 348), False, [('완', 22, 330), ('주', 55, 330)])]
VIEW = [((0, 88, 256, 276), 0x7810), ((0, 128, 256, 288), 0x7A90), ((0, 80, 296, 320), 0x7850), ((0, 80, 320, 348), 0x7850)]


def make(px):
    px = px.copy()
    for (u0, u1, v0, v1), zero, words in LOGOS:
        reg = px[v0:v1, u0:u1]; orig = reg.copy()
        core = np.isin(orig, CORE)
        l1 = (orig == 1) & BK.dil(core)
        rim = np.isin(orig, [9, 10, 11, 15]) & BK.dil(core | l1)
        mask = core | l1 | rim
        edge = 15 if zero else 10                                      # 혜성 위 둘레 = 주황 A(원본 9·A·B 섞임, B 는 너무 노랗다)
        rows = [np.bincount(orig[y][core[y]], minlength=16).argmax() if core[y].any() else None for y in range(len(orig))]
        idx = [y for y, v in enumerate(rows) if v is not None]
        one = int(np.bincount(orig[core], minlength=16).argmax())     # 2026-10-19 사용자 «조잡해, 깔끔하게 단색» — 줄별 은색 계조 대신 원본 글자 최빈 색 하나
        ramp = [one] * len(rows)
        if zero:
            reg[orig != 0] = 0
        else:                                                          # 혜성 바탕: 같은 줄 좌우 가까운 안 가린 화소
            for y in range(reg.shape[0]):
                free = np.nonzero(~mask[y] & (orig[y] != 1))[0]            # 1 = 혜성 그림자 가장자리 → 번지면 검은 덩어리
                for x in np.nonzero(mask[y])[0]:
                    k = free[np.abs(free - x).argmin()] if len(free) else None
                    reg[y, x] = orig[y, k] if k is not None and abs(k - x) <= 6 else 0
        H_, W_ = reg.shape; body = np.zeros((H_, W_), bool)
        for t, cu, cv in words:
            ink = BK.render(t, (9, False, 2), W_ - 3)
            gh, gw = ink.shape; b = np.zeros((gh, gw + 1), bool); b[:, :gw] |= ink; b[:, 1:] |= ink   # 세로획 3px(원본 굵기)
            ink = b; gw += 1
            x0 = max(1, min(W_ - 1 - gw, cu - u0 - gw // 2)); y0 = max(1, min(H_ - 1 - gh, cv - v0 - gh // 2))
            m = np.zeros((H_, W_), bool); m[y0:y0 + gh, x0:x0 + gw] = ink
            ring = BK.dil(m) & ~m
            reg[ring] = edge
            for y in range(H_):
                reg[y][m[y]] = ramp[y]
            body |= m | ring
        if not zero:
            clean(reg, body, u0, v0)
    return px


# 완주(혜성 위) 찌꺼기 손질(2026-10-19 사용자 «완주 주 밑에 찌꺼기들 지우고»): 아틀라스 좌표 (u0,u1,v0,v1) 상자 안 «새 글자·둘레 아닌» 화소
#   CLEAR = 0 으로(通 그림자 줄·주황 덩어리) · HOLE = 같은 줄 좌우가 혜성 몸통(D·E)이면 그 색으로(완 아래 검은 점)
CLEAR = [(44, 72, 333, 341)]
HOLE = [(13, 33, 328, 341)]


def clean(reg, body, u0, v0):
    for x0, x1, y0, y1 in CLEAR:
        for y in range(y0 - v0, y1 - v0):
            for x in range(x0 - u0, x1 - u0):
                if 0 <= y < reg.shape[0] and 0 <= x < reg.shape[1] and not body[y, x]:
                    reg[y, x] = 0
    for x0, x1, y0, y1 in HOLE:
        for y in range(y0 - v0, y1 - v0):
            if not 0 <= y < reg.shape[0]:
                continue
            row = reg[y]
            for x in range(x0 - u0, x1 - u0):
                if body[y, x] or row[x] in (13, 14):
                    continue
                l = [row[k] for k in range(x - 1, -1, -1) if not body[y, k] and row[k] in (13, 14)][:1]
                r = [row[k] for k in range(x + 1, len(row)) if not body[y, k] and row[k] in (13, 14)][:1]
                if l and r and row[x] in (0, 1):
                    row[x] = l[0]


# 왼쪽 작은 로고(2026-10-19 사용자 스샷, 신칸센 전 구간 완주 화면): 조각 219 電車でGO!(u96 v232 64×24, CLUT 7512 4비트) → «전차로» 만 한글(제목 로고 규칙, GO! 그대로)
#   조각 196(u0 v208 40×12)+208(u0 v220 28×12) プロフェッショナル仕様 판(CLUT 7552: 1 테두리·F 바탕·C 글자) → «프로페셔널 사양», 화면에서 196 x-12 · 208 x+28 로 이어 붙음
DENSHA = (96, 132, 232, 256)          # 電車で 영역 u96~131 — GO! 의 G 왼쪽 둘레가 u132(v240~), u133 은 G 세로획(사용자 «고 테두리 잘림»)


RAMP = {233: 2, 234: 2, 235: 2, 236: 2, 237: 2, 238: 3, 239: 3, 240: 4, 241: 4, 242: 5, 243: 6, 244: 6, 245: 7, 246: 7, 247: 8, 248: 9, 249: 9, 250: 10}   # GO! 줄별 색(원본에서 잼)


def small_logo(px):
    reg = px[DENSHA[2]:DENSHA[3], DENSHA[0]:DENSHA[1]]
    reg[:] = 0
    px[234:239, 132:135] = 0; px[239, 132:134] = 0; px[234:237, 135] = 0                 # で 탁점(゛) 나머지(u131~134 v235~239) — G 윗둘레는 u135~(v237~)·u134(v239)
    # 사용자 «전차로도 GO! 랑 색깔·폰트 맞춰, 그라데이션 원본처럼»: 나눔 ExtraBold 굵은 글씨를 GO! 처럼 기울이고(4줄에 1px) 줄별 색(RAMP), 검은 둘레
    g = BK.render('전차로', (11, True, 1), 33)                       # 갈무리11 Bold(가는 나눔은 이 크기에서 안 읽힘) → 세로 1.5배(GO! 높이) + 기울임
    ys = (np.arange(int(g.shape[0] * 1.5)) / 1.5).astype(int)
    m = g[ys]
    gh, gw = m.shape; sh = (gh - 1) // 8
    it = np.zeros((gh, gw + sh), bool)
    for y in range(gh):
        k = (gh - 1 - y) // 8; it[y, k:k + gw] = m[y]
    H_, W_ = reg.shape; gh, gw = it.shape
    assert gw + 2 <= W_, ('전차로 폭', gw, W_)
    x0 = max(1, W_ - gw - 1); y0 = 250 - DENSHA[2] - gh + 1                  # 바닥을 GO! 바닥(v250)에
    big = np.zeros((H_, W_), bool); big[y0:y0 + gh, x0:x0 + gw] = it
    reg[BK.dil(big) & ~big] = 1
    for y in range(H_):
        reg[y][big[y]] = RAMP.get(DENSHA[2] + y, 10)
    # 판: 두 조각 안쪽(196 행 210~216 열 2~39, 208 행 222~228 열 0~25)을 한 캔버스(화면 열 -10~53)로
    m = BK.render('프로페셔널 사양', (7, False, 1), 64)
    cv = np.zeros((7, 64), bool); gh, gw = m.shape; cv[(7 - gh) // 2:(7 - gh) // 2 + gh, (64 - gw) // 2:(64 - gw) // 2 + gw] = m
    a = px[210:217, 2:40]; a[:] = 15; a[cv[:, 0:38]] = 12            # 화면 -10~27
    b = px[222:229, 0:26]; b[:] = 15; b[cv[:, 38:64]] = 12           # 화면 28~53
    return px


def apply(pac):
    px, _ = H.load(bytes(pac)); H.store(pac, small_logo(make(px))); return len(LOGOS) + 2


def preview():
    pac = open(H.PAC, 'rb').read(); px, clut = H.load(pac); new = make(px); rows = []
    for (u0, u1, v0, v1), c in VIEW:
        p = H.pal(clut, c)
        def img(a):
            im = p[a[v0:v1, u0:u1]]; bg = np.zeros_like(im); bg[:] = (40, 40, 48, 255); m = im[..., 3] > 0; bg[m] = im[m]; return bg
        r = np.concatenate([img(px), np.full((v1 - v0, 4, 4), 255, np.uint8), img(new)], 1)
        rows.append(np.pad(r, ((0, 4), (0, 260 - r.shape[1]), (0, 0))))
    sh = np.concatenate(rows, 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_logo.png')
    Image.fromarray(sh).resize((sh.shape[1] * 4, sh.shape[0] * 4), Image.NEAREST).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
