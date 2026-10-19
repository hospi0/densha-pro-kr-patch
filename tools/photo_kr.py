# -*- coding: utf-8 -*-
r"""사진 플로어 397장(MDEC BS 640×480) 아래 설명 글 한글화 (2026-10-19, 사용자 B안 «반투명 띠»)
  번역 = work/text/photo_kr_1~3.tsv (idx·원문·번역, 줄 = '|'). idx = tools/photo_dump.photos() 순서.
  원본 자리(실측): 글 줄은 아래 맞춤 — 1줄 y414~434, 2줄 y389~409 / 414~434. 왼쪽 맞춤 = x42 시작, 오른쪽 맞춤 = x597 끝.
  ① 글자 가리개 = 밝음(>165) + 오른쪽 아래 1~2px 그림자(55 이상 어두움), 줄 띠 안에서만 → inpaint
  ② 줄 띠 전체 폭을 어둡게(×0.45) ③ 나눔고딕 Bold 22 흰 글자 + 검은 그림자(+2,+2), 원래 맞춤대로. 넓으면 글자 크기 줄임.
  python tools/photo_kr.py [idx…] → work/gfx/prev_photo.png (원본|한글)
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import photo_dump as PD  # noqa: E402

FONT = r'C:\claude\utils\font\nanum-gothic\NanumGothicBold.ttf'
LINES = {1: [(414, 435)], 2: [(389, 410), (414, 435)]}
LEFT, RIGHT = 42, 598


def table():
    out = {}
    for k in (1, 2, 3):
        for l in open(os.path.join(ROOT, 'work', 'text', 'photo_kr_%d.tsv' % k), encoding='utf-8').read().split('\n')[1:]:
            if l:
                i, jp, kr = l.split('\t'); out[int(i)] = (jp.split('|'), kr.split('|'))
    return out


def text_mask(L):
    sh = np.full(L.shape, 255.0); sh[:-2, :-2] = L[2:, 2:]
    sh1 = np.full(L.shape, 255.0); sh1[:-1, :-1] = L[1:, 1:]
    return (L > 165) & (np.minimum(sh, sh1) < L - 55)


def make(rgb, jp, kr):
    n = len(jp); assert len(kr) == n, (jp, kr)
    rows = LINES[n]
    L = rgb.astype(np.float32).mean(2)
    t = text_mask(L)
    band = np.zeros(L.shape, bool)
    for y0, y1 in rows:
        band[y0 - 3:y1 + 3] = True
    tl = t & band
    left = tl[:, 36:90].sum(); right = tl[:, 550:604].sum()
    al = 'l' if left >= right else 'r'
    m = tl.copy(); m[2:, 2:] |= tl[:-2, :-2]; m[1:, 1:] |= tl[:-1, :-1]
    m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8))
    out = cv2.inpaint(np.ascontiguousarray(rgb), m, 3, cv2.INPAINT_TELEA).astype(np.float32)
    y0, y1 = rows[0][0] - 6, rows[-1][1] + 5
    out[y0:y1] *= 0.45
    im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    for size in (22, 21, 20, 19, 18):                               # 한 설명 안은 같은 크기
        f = ImageFont.truetype(FONT, size)
        bw = max(f.getbbox(s)[2] - f.getbbox(s)[0] for s in kr)
        if bw <= RIGHT - LEFT:
            break
    hl, ht, hr, hb = f.getbbox('한')
    bx = LEFT if al == 'l' else RIGHT - bw                          # 오른쪽 자리도 덩어리 왼쪽 맞춤(원본: 두 줄 시작 x 같음)
    for (ly0, ly1), s in zip(rows, kr):
        l = f.getbbox(s)[0]
        x = bx - l
        y = ly0 + ((ly1 - ly0) - (hb - ht)) // 2 - ht
        d.text((x + 2, y + 2), s, font=f, fill=(0, 0, 0)); d.text((x, y), s, font=f, fill=(255, 255, 255))
    return np.array(im), al


def build_one(args):
    k, o, n, jp, kr = args
    import bs
    rgb = np.array(Image.open(os.path.join(PD.OUT, '%X.png' % o)).convert('RGB'))
    new, al = make(rgb, jp, kr)
    return k, o, n, rgb, new


def build_all(pac=None):
    """→ [(위치, 자리 크기, 원본, 새것)] (mdec_kr.make 형식)"""
    T = table(); P = PD.photos(); out = []
    for k, (o, n) in enumerate(P):
        jp, kr = T[k]
        rgb = np.array(Image.open(os.path.join(PD.OUT, '%X.png' % o)).convert('RGB'))
        out.append((o, n, rgb, make(rgb, jp, kr)[0]))
    return out


def preview(ids):
    T = table(); P = PD.photos(); rows = []
    for k in ids:
        o, n = P[k]; jp, kr = T[k]
        rgb = np.array(Image.open(os.path.join(PD.OUT, '%X.png' % o)).convert('RGB'))
        new, al = make(rgb, jp, kr)
        rows.append(np.concatenate([rgb[370:470], np.full((100, 4, 3), 255, np.uint8), new[370:470]], 1))
    sh = np.concatenate(rows, 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_photo.png'); Image.fromarray(sh).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview([int(a) for a in sys.argv[1:]] or [0, 1, 6, 9, 12, 21, 31, 52, 63, 111, 177, 232, 339, 383])
