# -*- coding: utf-8 -*-
r"""주행 중 풍경 속 장소 이름표(교량·도로·강·신호장) 한글화 (2026-10-19)
  노선 TXS ((512,0,512,512),(448,0,64,512),(0,464,256,48)) 의 사각형 1(4bpp 256×512) 아래쪽, 이름표 = 한 글자 16px 칸 × 16줄 회색 상자.
  칸 좌표(눈금 그림으로 잼, 상자 = 글자 수×16): LABELS. 같은 그림 사본(아키타 2·케이힌 4)은 COPIES 로 함께.
  꼴: 상자 바탕(칸 최빈 색인) 다시 칠하고 갈무리9 글자(전부 같은 크기) = 칸에서 가장 어두운 쪽 색인(1). 폭 넘치면 갈무리7.
  python tools/scen_kr.py → work/gfx/prev_scen.png
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hud_px as HP  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
COPIES = {0x145000: [0x145000, 0x21A000], 0x539800: [0x539800, 0x5D0800, 0x667800, 0x80B800]}
LABELS = {
    0x145000: [(0, 464, 6, '제1 타마가와 교량'), (0, 480, 6, '아키타 자동차도'), (120, 480, 6, '아키타 차량 기지'),
               (0, 496, 3, '이와미강'), (64, 496, 7, '아키타 북부 우회로')],
    0x2EF800: [(0, 496, 6, '제1 타마가와 교량'), (96, 496, 6, '시도나이 신호장'), (192, 496, 4, '아키타 가도')],
    0x3DA000: [(64, 424, 6, '호쿠리쿠 자동차도'), (64, 440, 6, '칸에츠 자동차도'), (64, 456, 6, '야쿠시토게 신호장'),
               (64, 472, 5, '아카쿠라 신호장'), (64, 488, 5, '기묘 신호장'), (144, 488, 2, '세키강'), (176, 488, 2, '가타강'),
               (208, 488, 3, '시나노강')],
    0x539800: [(0, 464, 3, '난부선'), (0, 480, 3, '츠루미선'), (64, 480, 3, '요코하마선'), (128, 480, 3, '타마강'),
               (192, 480, 3, '츠루미강')],
    0xA17800: [(0, 496, 3, '오미야 거리'), (48, 496, 3, '호즈강')],
    0xCE9000: [(80, 480, 7, '부도 중앙 환상선'), (192, 480, 2, '카츠라강'), (224, 480, 2, '아쿠타강'),
               (0, 496, 6, '메이신 고속도로'), (96, 496, 5, '한큐 쿄토선'), (176, 496, 2, '요도강'), (208, 496, 3, '아이강')],
}


def rect1(pac, off):
    a = off + 0x800 + 512 * 512 * 2
    raw = np.frombuffer(pac[a:a + 64 * 512 * 2], np.uint8).reshape(512, 128)
    return a, np.stack([raw & 15, raw >> 4], -1).reshape(512, 256).copy()


def encode(px):
    p = px.reshape(512, 128, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def make(px, labels):
    px = px.copy()
    for x, y, n, text in labels:
        w = 16 * n; reg = px[y:y + 16, x:x + w]
        cnt = np.bincount(reg.ravel(), minlength=16)
        bg = int(cnt.argmax()); dark = min(k for k in range(1, 16) if cnt[k] > 0 and k != bg)
        lv = HP.word(text, w, 16, 'S', 'c')                     # 전부 갈무리9(크기 통일), 안 들어가면 7 · 획(3)만 씀
        reg[:] = bg; reg[lv == 3] = dark
    return px


def apply(pac):
    n = 0
    for key, labels in LABELS.items():
        for off in COPIES.get(key, [key]):
            a, px = rect1(bytes(pac), off)
            enc = encode(make(px, labels)); pac[a:a + len(enc)] = enc; n += 1
    return n


def preview():
    pac = open(PAC, 'rb').read(); rows = []
    for key, labels in LABELS.items():
        _, px = rect1(pac, key); new = make(px, labels)
        y0 = min(l[1] for l in labels); y1 = max(l[1] for l in labels) + 16
        g = lambda a: np.where(a[y0:y1] == 0, 0, np.where(a[y0:y1] >= 3, 210, 50)).astype(np.uint8)
        rows += [np.concatenate([g(px), np.full((y1 - y0, 4), 255, np.uint8), g(new)], 1), np.full((4, 516), 128, np.uint8)]
    sh = np.concatenate(rows, 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_scen.png')
    Image.fromarray(sh).resize((sh.shape[1] * 2, sh.shape[0] * 2), Image.NEAREST).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
