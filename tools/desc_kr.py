# -*- coding: utf-8 -*-
r"""주행 중 열차 소개 상자(TXS 사각형 (320,160,64,64) = 4bpp 256×64, 90장) 한글화 (2026-10-19)
  문장은 그림에만 있음(PAC·EXE 평문 없음) → 그림을 읽어 번역 work/text/desc_kr.tsv (off·원문 3줄·번역 3줄).
  색인: 3 = 상자 바탕, 0 = 글자(흰), 1·2 = 가장자리 중간. 줄 0~47 만 글자 자리(48줄~ 다른 자료 — 손대지 않음).
  배치(원본 실측): 제목 4~14줄(x≈9), 본문 1줄 20~28, 2줄 32~40(x≈16).
  글꼴 = 나눔고딕 ExtraBold, 넘치면 가로 85% 까지 좁히고 그래도 넘치면 실패(번역을 줄일 것).
  python tools/desc_kr.py → work/gfx/prev_desc_NN.png (원본|한글, 2배)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
TSV = os.path.join(ROOT, 'work', 'text', 'desc_kr.tsv')
FONT = r'C:\claude\utils\font\nanum-gothic\NanumGothicExtraBold.ttf'
W, H, SS = 256, 64, 8
TITLE = (9, 4, 11)                 # x, 맨 윗줄, 잉크 높이
LINES = ((15, 20, 9), (15, 32, 9))
PREV_PAL = np.array([[248, 248, 248], [196, 196, 196], [130, 130, 136], [52, 52, 60]] + [[255, 0, 255]] * 12, np.uint8)


def rows():
    out = []
    for ln in open(TSV, encoding='utf-8').read().split('\n')[1:]:
        if ln:
            a = ln.split('\t'); out.append((int(a[0], 16), a[4], a[5], a[6]))
    return out


def load(pac, off):
    a = off + 0x800
    raw = np.frombuffer(pac[a:a + W * H // 2], np.uint8).reshape(H, W // 2)
    return np.stack([raw & 15, raw >> 4], -1).reshape(H, W).copy()


def encode(px):
    p = px.reshape(H, W // 2, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def cover(text, h_ink, maxw):
    f = ImageFont.truetype(FONT, int(h_ink * SS * 1.3))
    hl, ht, hr, hb = f.getbbox('한')
    l, t, r, b = f.getbbox(text)
    top = min(t, ht); bot = max(b, hb)
    im = Image.new('L', (r - l + 8, bot - top + 8))
    ImageDraw.Draw(im).text((4 - l, 4 - top), text, font=f, fill=255)
    a = np.array(im)
    y0 = 4 + ht - top; extra_top = min(y0, int((hb - ht) * 0.15))           # 위로 조금 튀어나오는 획(따옴표 등) 여유
    a = a[y0 - extra_top:4 + hb - top]
    cols = np.nonzero(a.max(0))[0]; a = a[:, cols.min():cols.max() + 1]
    sc = h_ink / (hb - ht)
    Wd = int(round(a.shape[1] * sc)); Hd = int(round(a.shape[0] * sc))
    if Wd > maxw:
        assert Wd * 0.85 <= maxw, '넘침 %r 폭 %d > %d' % (text, Wd, maxw)
        Wd = maxw
    return np.array(Image.fromarray(a).resize((Wd, Hd), Image.BOX)).astype(float) / 255, Hd - h_ink


def put(px, cov, x, y):
    hh, ww = cov.shape
    reg = px[y:y + hh, x:x + ww]
    lv = np.full(cov.shape, 3, np.uint8)
    lv[cov >= 0.25] = 2; lv[cov >= 0.5] = 1; lv[cov >= 0.75] = 0
    reg[:] = np.minimum(reg, lv)


def make_one(px, title, l1, l2):
    px = px.copy()
    px[:48] = 3
    for text, (x, y, h) in zip((title, l1, l2), (TITLE,) + LINES):
        c, up = cover(text, h, W - x - 4)
        put(px, c, x, y - up)
    return px


def build(pac):
    out = []
    for off, t, a, b in rows():
        px = load(pac, off)
        out.append((off, px, make_one(px, t, a, b)))
    return out


def apply(pac):
    n = 0
    for off, _, new in build(bytes(pac)):
        enc = encode(new); pac[off + 0x800:off + 0x800 + len(enc)] = enc; n += 1
    return n


def preview():
    pac = open(PAC, 'rb').read(); res = build(pac)
    for k in range(0, len(res), 10):
        tiles = []
        for off, a, b in res[k:k + 10]:
            ia = PREV_PAL[a[:48]]; ib = PREV_PAL[b[:48]]
            tiles.append(np.concatenate([ia, np.full((48, 4, 3), 255, np.uint8), ib], 1))
            tiles.append(np.full((4, ia.shape[1] * 2 + 4, 3), 0, np.uint8))
        sh = np.concatenate(tiles, 0)
        p = os.path.join(ROOT, 'work', 'gfx', 'prev_desc_%02d.png' % (k // 10))
        Image.fromarray(sh).resize((sh.shape[1] * 2, sh.shape[0] * 2), Image.NEAREST).save(p)
    print('미리보기', os.path.join(ROOT, 'work', 'gfx', 'prev_desc_00~%02d.png' % ((len(res) - 1) // 10)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
