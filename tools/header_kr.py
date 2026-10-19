# -*- coding: utf-8 -*-
r"""화면 왼쪽 위 머리 제목(運転乗務·鉄道資料館·ロード・セーブ·ゲーム設定) 한글화 (2026-10-18)
  그림 = 메뉴 아틀라스 TXS 989A000 사각형 0 의 tpage E(VRAM 896,0) u0~95 v0~95, 레코드 = 22번 파일 0x19CDA~0x19D2A(제자리, 레코드 그대로).
  ★두 그림을 값 0~8 로 겹침: CLUT 7E79(그림 A) 보이는 1·7·8 회색, 2·4·5 흰 / CLUT 7EB9(그림 B) 3·4·7 흰, 5·6·8 회색
    (A, B) = (회,0)1 (흰,0)2 (0,흰)3 (흰,흰)4 (흰,회)5 (0,회)6 (회,흰)7 (회,회)8.
  칸 (0,0): A = 鉄道資料館 · B = ゲーム設定 / 칸 (0,48): A = ロード・セーブ(96폭) · B = 運転乗務 — 넷 다 같은 글자 크기.
  menu_kr 다음에 적용(같은 아틀라스). python tools/header_kr.py → work/gfx/prev_header.png
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import btn_kr as BT  # noqa: E402
import menu_kr as MK  # noqa: E402

TP = 0xE
ENC = {(0, 0): 0, (2, 0): 1, (1, 0): 2, (0, 1): 3, (1, 1): 4, (1, 2): 5, (0, 2): 6, (2, 1): 7, (2, 2): 8}   # 1 = 흰·2 = 회색
DEC = {v: k for k, v in ENC.items()}
CELLS = {(0, 0): ('철도 자료관', '게임 설정', 88, 88), (0, 48): ('로드・세이브', '운전 승무', 96, 88)}   # (A, B, A폭, B폭)


def lv(cov):
    return np.where(cov >= 0.5, 1, np.where(cov >= 0.2, 2, 0)).astype(np.uint8)


def make(pac):
    at = MK.Atlas(pac); orig = at.px.copy()
    ox, oy = at.page(TP)
    texts = [t for a, b, wa, wb in CELLS.values() for t in (a, b)]
    for h in range(36, 14, -1):                          # 넷 다 같은 높이·비율, 각 칸 폭 안
        cov = {t: BT.cover(t, h, 0.58)[0] for t in texts}          # 원본도 세로로 긴 좁은 글꼴
        if all(cov[a].shape[1] <= wa - 2 and cov[b].shape[1] <= wb - 2 for a, b, wa, wb in CELLS.values()):
            break
    for (u, v), (a, b, wa, wb) in CELLS.items():
        reg = at.px[oy + v:oy + v + 48, ox + u:ox + u + 96]
        assert reg.max() <= 8
        A = np.zeros((48, 96), np.uint8); B = np.zeros((48, 96), np.uint8)
        for t, L, W in ((a, A, wa), (b, B, wb)):
            g = lv(cov[t]); hh, ww = g.shape
            y0 = (48 - hh) // 2; x0 = (W - ww) // 2
            L[y0:y0 + hh, x0:x0 + ww] = g
        reg[:] = np.vectorize(lambda p, q: ENC[(p, q)])(A, B).astype(np.uint8)
    return at, orig, h


def apply(pac):
    at, _, _ = make(bytes(pac))
    a = MK.TXS + 0x800; enc = at.encode()
    pac[a:a + len(enc)] = enc
    return len(CELLS) * 2


def preview():
    import vram as V
    pac = open(MK.PAC, 'rb').read()
    at, orig, h = make(pac)
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '16.sav'))
    ox, oy = at.page(TP); rows = []
    for (u, vv), (a, b, wa, wb) in CELLS.items():
        for c, W in ((0x7E79, wa), (0x7EB9, wb)):
            cx, cy = (c & 0x3F) * 16, c >> 6; cc = v[cy, cx:cx + 16].astype(int)
            pal = np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8); pal[cc == 0] = (60, 50, 70)
            x = pal[orig[oy + vv:oy + vv + 48, ox + u:ox + u + W]]; y = pal[at.px[oy + vv:oy + vv + 48, ox + u:ox + u + W]]
            r = np.concatenate([x, np.full((48, 3, 3), 255, np.uint8), y], 1)
            rows.append(np.pad(r, ((0, 4), (0, 200 - r.shape[1]), (0, 0)), constant_values=25))
    im = np.concatenate(rows, 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_header.png')
    Image.fromarray(im).resize((im.shape[1] * 3, im.shape[0] * 3), Image.NEAREST).save(p); print('미리보기', p, '높이', h)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
