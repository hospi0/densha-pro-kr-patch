# -*- coding: utf-8 -*-
r"""컨트롤 화면 패드 그림 라벨 한글화 (2026-10-19)
  그림 = 메뉴 아틀라스 TXS 989A000 사각형 0 (menu_kr.Atlas) 의 u112~255 × v0~255, tpage 0x20D, CLUT 7EF8(16색 4비트, 라벨마다 밝은색·어두운색 한 쌍).
    화면은 이 영역을 조각(SPRT)으로 붙임 — 상자 안 라벨은 30×16 조각(非常 114,0 / 解除 114,16 / 加速 114,32 / 切る 114,112)을 배치마다 바꿔 끼움.
  라벨 상자 안에서 «그 라벨 색» 화소만 지우고(선·상자는 상자 밖) 같은 밝은색으로 갈무리11 Bold, 오른쪽 아래 1px 어두운색 그림자.
  용어 = 파이널 한글판(마스콘·끔·가속·비상·브레이크 해제·경적).
  python tools/ctrl_kr.py → work/gfx/prev_ctrl.png (원본 | 한글, CLUT 7EF8 실제 색)
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import menu_kr as MK  # noqa: E402
import pixfont as PF  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
BLUE, GREEN, PINK, YELLOW, RED, PURPLE, CYAN = (11, 7), (10, 8), (9, 6), (13, 5), (12, 4), (3, 2), (15, 14)
LABELS = [                                                   # (u, v, w, h 상자 = 선·틀에 안 닿게, 글자, (밝은, 어두운))
    (114, 0, 30, 16, '비상', BLUE), (114, 16, 30, 16, '해제', BLUE), (114, 32, 30, 16, '가속', GREEN),
    (158, 2, 28, 26, '끔', PINK), (194, 4, 56, 22, '비상BK', YELLOW), (230, 48, 26, 23, '경적', RED),
    (112, 50, 40, 20, '마스콘', GREEN), (114, 112, 30, 16, '끔', GREEN), (114, 128, 30, 16, '비상', GREEN),
    (152, 122, 32, 18, 'PRO', PURPLE), (150, 141, 36, 19, '모드', PURPLE),
    (198, 112, 48, 23, '브레이크', BLUE), (116, 212, 70, 23, '마스콘 끔', PINK), (200, 213, 50, 22, 'BK 해제', CYAN),
]
PAL = [(40, 40, 40), (0, 0, 0), (96, 0, 128), (160, 32, 232), (160, 0, 24), (160, 104, 0), (136, 24, 80), (0, 48, 184),
       (0, 144, 24), (248, 88, 176), (0, 232, 72), (0, 128, 248), (248, 0, 48), (248, 200, 0), (0, 128, 128), (0, 208, 208)]   # 스테이트 VRAM CLUT 7EF8


def text(t):
    cols = []
    for ch in t:
        if ch == ' ':
            cols.append(np.zeros((11, 3), bool)); continue
        g = PF.text(ch, 11, True) > 0 if ord(ch) < 128 else PF.cell(ch, 11, True) > 0
        if g.shape[0] < 11:
            g = np.vstack([np.zeros((11 - g.shape[0], g.shape[1]), bool), g])
        cols.append(g); cols.append(np.zeros((11, 1), bool))
    g = np.concatenate(cols[:-1], 1)
    return g[:, :np.nonzero(g.any(0))[0].max() + 1]


def make(px):
    px = px.copy()
    for u, v, w, h, t, (hi, lo) in LABELS:
        reg = px[v:v + h, u:u + w]
        reg[(reg == hi) | (reg == lo)] = 0
        g = text(t); gh, gw = g.shape
        assert gw + 1 <= w and gh + 1 <= h, (t, gw, w)
        y0 = (h - gh - 1) // 2; x0 = (w - gw - 1) // 2
        sub = reg[y0:y0 + gh + 1, x0:x0 + gw + 1]
        sh = np.zeros(sub.shape, bool); sh[1:, 1:] = g
        sub[sh & (sub == 0)] = lo
        sub[:gh, :gw][g] = hi
    return px


def apply(pac):
    at = MK.Atlas(bytes(pac)); at.px[0:256, 112:256] = make(at.px)[0:256, 112:256]
    a = MK.TXS + 0x800; enc = at.encode(); pac[a:a + len(enc)] = enc
    return len(LABELS)


def preview():
    at = MK.Atlas(open(PAC, 'rb').read()); pal = np.array(PAL, np.uint8)
    a = at.px[0:256, 112:256]; b = make(at.px)[0:256, 112:256]
    sh = np.concatenate([pal[a], np.full((256, 4, 3), 255, np.uint8), pal[b]], 1)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_ctrl2.png')
    Image.fromarray(sh).resize((sh.shape[1] * 3, sh.shape[0] * 3), Image.NEAREST).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
