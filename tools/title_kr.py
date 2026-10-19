# -*- coding: utf-8 -*-
r"""제목 화면(MDEC BS 정지 영상 24장, 640×480) 로고 «電車で» → «전차로» · 판 «プロフェッショナル仕様» → «프로페셔널 사양» (2026-10-19)
  사용자: «전차로도 하고, 파이널 때 했던 거 응용» → densha-final tools/logo_kr.py 의 글자 그리기(4배 그려 줄이기·흰 안쪽 테두리 3단·
  검정 바깥 2px + 그림자(오른쪽 4·아래 5)·줄마다 채움 색·전차 기울임 0.15·로 그림자 얇게)를 RGB 로 옮김.
  로고는 24장 모두 같고 위치만 기준/왼쪽 −244px. 24장 정렬 중앙값·편차로 로고 화소(가림막)를 뽑음 → work/gfx/bs/logo_*.npy.
  옛 電車で 자리(가림막 중 GO 초록에서 7px 밖·크롭 x<175·y<100)는 장마다 배경을 inpaint 로 메우고 새 글자를 얹음.
  판 안쪽은 장마다 같음 → 중앙값 판에서 글자를 지우고(줄마다 초록 최빈색) 금색 글자를 다시 그려 붙임.
  굽기 = bs.encode(v3, 원래 자리 크기). python tools/title_kr.py preview → work/gfx/prev_title.png
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bs  # noqa: E402
import btn_kr as BT  # noqa: E402

FONT = r'C:\claude\utils\font\logo\BlackHanSans.ttf'
SS = 4
X0, Y0, CW, CH = 280, 30, 330, 205                 # 크롭(기준 위치)
FIRST, COUNT = 0x95E3800, 24
BIG = ('전차', (10, 10, 146, 80))                  # 電車 몸통(크롭 좌표) — 원본만큼 크게
SMALL = ('로', (136, 54, 172, 94))                 # で 자리
SHADOW, SHADOW_S, SHEAR = (4, 5), (2, 2), 0.15
PLATE = (24, 156, 304, 200)                        # 판 둘레 대략(크롭) — 판 속은 plate_new 가 잼
PLATE_TXT = '프로페셔널 사양'


def titles(pac):
    F = bs.frames(pac); ks = [k for k, f in enumerate(F) if f[0] >= FIRST][:COUNT]
    return [(F[k][0], F[k + 1][0] - F[k][0]) for k in ks]


def offsets(ims, med_plate):
    out = []
    for im in ims:
        r = cv2.matchTemplate(im, med_plate, cv2.TM_CCOEFF_NORMED); _, _, _, loc = cv2.minMaxLoc(r)
        out.append(loc[0] - (X0 + PLATE[0]))
    return out


def mask_text(text, box, gap=1, grow=1, shear=0.0):
    x0, y0, x1, y1 = box; H = (y1 - y0) * SS
    f = ImageFont.truetype(FONT, H * 2); gl = []
    for ch in text:
        l, t, r, b = f.getbbox(ch)
        im = Image.new('L', (r - l + 8, b - t + 8)); ImageDraw.Draw(im).text((4 - l, 4 - t), ch, font=f, fill=255)
        a = np.array(im); ys = np.nonzero(a.max(1))[0]; xs = np.nonzero(a.max(0))[0]
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        gl.append(np.array(Image.fromarray(a).resize((max(1, int(round(a.shape[1] * H / a.shape[0]))), H), Image.LANCZOS)))
    g4, w4 = gap * SS, grow * SS
    W = sum(g.shape[1] for g in gl) + g4 * (len(gl) - 1) + 2 * w4 + int(shear * H)
    sx = min(1.0, (x1 - x0) * SS / W)
    out = np.zeros((CH * SS, CW * SS), np.uint8)
    x = x0 * SS + int(round(((x1 - x0) * SS - W * sx) / 2)) + w4; Yb = y0 * SS
    for g in gl:
        w = max(1, int(round(g.shape[1] * sx)))
        g = np.array(Image.fromarray(g).resize((w, H), Image.LANCZOS))
        for yy in range(H):
            dx = int(round(shear * sx * (H - 1 - yy)))
            out[Yb + yy, x + dx:x + dx + w] = np.maximum(out[Yb + yy, x + dx:x + dx + w], g[yy])
        x += w + int(round(g4 * sx))
    m = out >= 128
    if grow:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w4 + 1, 2 * w4 + 1))
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    return m


def layer(m, edge, fill, thin=False, ring=2, shadow=SHADOW):
    """4배 해상도 RGBA(0~1)"""
    H4, W4 = m.shape; img = np.zeros((H4, W4, 4))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring * SS + 1, 2 * ring * SS + 1))
    o = cv2.dilate(m.astype(np.uint8), k)
    dx, dy = shadow[0] * SS, shadow[1] * SS
    sh = np.zeros_like(o); sh[dy:, dx:] = o[:-dy, :-dx]
    outl = (o | sh).astype(bool)
    dd = cv2.distanceTransform((~outl).astype(np.uint8), cv2.DIST_L2, 5) / SS
    soft = (~outl) & (dd <= 2.0)
    img[soft, :3] = 0; img[soft, 3] = (1 - dd[soft] / 2.0) * 0.8
    img[outl] = [0, 0, 0, 1]
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) / SS
    col = np.broadcast_to(np.repeat(fill, SS, 0)[:, None, :], (H4, W4, 3)).copy()
    if thin:
        col[d <= 1.0] = edge[1]
    else:
        col[d <= 3.0] = edge[2]; col[d <= 2.0] = edge[1]; col[d <= 1.0] = edge[0]
    img[m, :3] = col[m]; img[m, 3] = 1
    return img


def down(img):
    H, W = img.shape[0] // SS, img.shape[1] // SS
    a = img.reshape(H, SS, W, SS, 4); al = a[..., 3]
    A = al.mean((1, 3))
    C = (a[..., :3] * al[..., None]).sum((1, 3)) / np.maximum(al.sum((1, 3)), 1e-6)[..., None]
    return np.concatenate([C, A[..., None]], -1)


def learn(med, mask):
    r, g, b = med[..., 0], med[..., 1], med[..., 2]
    red = mask & (r > 140) & (g < 90) & (b < 110)
    blue = mask & (b > 150) & (r < 120)
    green = mask & (g > 140) & (r < 140) & (b < 100)
    core = cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    dd = cv2.distanceTransform(cv2.dilate(red.astype(np.uint8), np.ones((9, 9), np.uint8)), cv2.DIST_L2, 3)
    white = mask & (med.min(-1) > 170) & (dd > 0)
    edge = [np.array([250, 250, 250.]), np.array([235, 235, 235.]), np.array([215, 215, 220.])]
    if white.any():
        w = med[white]; edge = [w.max(0), np.median(w, 0), np.percentile(w, 25, 0)]

    def rows(m):
        out = {}
        for y in range(CH):
            v = med[y][m[y]]
            if len(v) >= 3:
                out[y] = np.median(v, 0)
        c = np.median(np.array(list(out.values())), 0)   # 글자 속 한 색(사용자: 색 균일하게)
        return np.repeat(c[None, :], CH, 0)
    return edge, rows(red), rows(blue), red, blue, green


def old_footprint(mask, red, blue, green):
    X = np.arange(CW)[None, :]; Y = np.arange(CH)[:, None]
    full = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    dG = cv2.distanceTransform((~green).astype(np.uint8), cv2.DIST_L2, 3)
    dB = cv2.distanceTransform((~blue).astype(np.uint8), cv2.DIST_L2, 3)
    keepG = (dG <= 7) & (dG < dB)
    fp = full & ((X < 182) & (Y < 100) | (X < 196) & (Y < 55)) & ~keepG   # で 탁점·바깥 테두리까지
    return cv2.dilate(fp.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~keepG   # 넉넉히(가장자리 얼룩 방지)


def spaced(text, h, width):
    """글자마다 같은 크기로 그려 사이를 고르게 벌려 width 폭으로(공백은 글자 반 칸)"""
    gl = [None if t == ' ' else BT.cover(t, h, 1.0)[0] for t in text]
    gh = max(g.shape[0] for g in gl if g is not None)
    ws = [g.shape[1] if g is not None else 0 for g in gl]
    nchar = sum(1 for g in gl if g is not None); nsp = text.count(' ')
    unit = (width - sum(ws)) / max(1, (len(text) - 1) + nsp)   # 글자 사이 = unit, 공백 = 2 unit
    out = np.zeros((gh, width)); x = 0.0
    for g in gl:
        if g is None:
            x += unit; continue
        xi = int(round(x)); w = min(g.shape[1], width - xi)
        out[:g.shape[0], xi:xi + w] = np.maximum(out[:g.shape[0], xi:xi + w], g[:, :w])
        x += g.shape[1] + unit
    return out


def plate_new(med):
    """판 안쪽: 글자(금색) 지우고 «프로페셔널 사양» 금색으로 → (y0, y1, x0, x1, 새 RGB)"""
    x0, y0, x1, y1 = PLATE
    reg = med[y0:y1, x0:x1].astype(float)
    r, g, b = reg[..., 0], reg[..., 1], reg[..., 2]
    green = (g > r + 15) & (g > b) & (reg.max(-1) < 140)

    def span(f):                                         # 첫 «초록 30% 이상» 칸부터, 초록 0 이 4칸 이어지는 곳(금테) 직전까지 — 바깥 배경 초록은 빠짐
        a = int(np.argmax(f > 0.3)); b_ = a; z = 0
        for i in range(a, len(f)):
            if f[i] > 0.05:
                b_ = i; z = 0
            else:
                z += 1
                if z >= 4:
                    break
        return a, b_
    iy0, iy1 = span(green.mean(1)); ix0, ix1 = span(green[iy0:iy1 + 1].mean(0))
    inner = reg[iy0:iy1 + 1, ix0:ix1 + 1]
    gi = green[iy0:iy1 + 1, ix0:ix1 + 1]
    gold = (inner[..., 0] > 120) & (inner[..., 1] > 100) & ~gi
    sat = gold & (inner[..., 0] - inner[..., 2] > 70)
    goldc = np.median(inner[sat], 0) if sat.any() else np.array([220, 180, 60.])   # 채도 있는 금색(밝은 반사광 빼고)
    body = cv2.morphologyEx(gi.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)   # 초록 속(글자 구멍 메움) — 금테는 안 건드림
    body = np.ones_like(body)                            # 판 속 사각형 통째로(금테는 span 이 이미 뺌) — 글자 끝 조각 남지 않게
    out = inner.copy()
    for y in range(inner.shape[0]):
        v = inner[y][gi[y]]
        bg = np.median(v, 0) if len(v) else np.median(inner[gi], 0)
        out[y][body[y]] = bg
    ys2 = np.nonzero(gold.any(1))[0]
    th = ys2.max() - ys2.min() + 1 if len(ys2) else inner.shape[0] - 4
    bys, bxs = np.nonzero(body)
    bw = bxs.max() - bxs.min() - 4
    c = spaced(PLATE_TXT, 24, int(bw * 0.8))            # 판 폭 80% 까지 글자 사이를 벌림(같은 크기 글자)
    hh, ww = c.shape; oy = bys.min() + (bys.max() - bys.min() + 1 - hh) // 2; ox = bxs.min() + (bxs.max() - bxs.min() + 1 - ww) // 2
    sub = out[oy:oy + hh, ox:ox + ww]
    sub[:] = sub * (1 - c[..., None]) + goldc * c[..., None]
    return y0 + iy0, y0 + iy1 + 1, x0 + ix0, x0 + ix1 + 1, out


def compose(img, ov):
    a = ov[..., 3:4]
    return img * (1 - a) + ov[..., :3] * a


def build_all(pac):
    T = titles(pac)
    ims = [bs.decode(pac[o:o + n], 640, 480) for o, n in T]
    med = np.load(os.path.join(ROOT, 'work', 'gfx', 'bs', 'logo_med.npy'))
    mask = np.load(os.path.join(ROOT, 'work', 'gfx', 'bs', 'logo_mask.npy'))
    offs = offsets(ims, med[PLATE[1]:PLATE[3], PLATE[0]:PLATE[2]].astype(np.uint8))
    edge, red_rows, blue_rows, red, blue, green = learn(med, mask)
    fp = old_footprint(mask, red, blue, green)
    big = down(layer(mask_text(*BIG, shear=SHEAR), edge, red_rows))
    small = down(layer(mask_text(*SMALL, grow=0, shear=SHEAR), edge, blue_rows, thin=True, ring=1, shadow=SHADOW_S))
    py0, py1, px0, px1, pnew = plate_new(med)
    out = []
    for (o, n), im, off in zip(T, ims, offs):
        x0 = X0 + off
        img = im.copy()
        m = np.zeros(img.shape[:2], np.uint8); m[Y0:Y0 + CH, x0:x0 + CW][fp] = 255
        img = cv2.inpaint(img, m, 7, cv2.INPAINT_TELEA)
        crop = img[Y0:Y0 + CH, x0:x0 + CW].astype(float)
        crop = compose(compose(crop, big), small)
        crop[py0:py1, px0:px1] = pnew
        img[Y0:Y0 + CH, x0:x0 + CW] = np.clip(crop, 0, 255).astype(np.uint8)
        out.append((o, n, im, img))
    return out


def save(im, p):
    """미리보기 파일이 그림 보기로 열려 있으면 잠깐 막힘 → 몇 번 다시"""
    import time
    for _ in range(20):
        try:
            im.save(p); return
        except OSError:
            time.sleep(0.5)
    im.save(p)


def preview():
    pac = open(bs.PAC, 'rb').read()
    res = build_all(pac)
    rows = []
    for o, n, a, b in res[:4]:
        rows.append(np.concatenate([a, np.full((480, 4, 3), 255, np.uint8), b], 1))
    im = np.concatenate(rows[:2], 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_title.png'); save(Image.fromarray(im), p)
    z = np.concatenate([res[0][2][20:240, 260:620], res[0][3][20:240, 260:620]], 1)
    save(Image.fromarray(z).resize((z.shape[1] * 2, z.shape[0] * 2), Image.NEAREST), os.path.join(ROOT, 'work', 'gfx', 'prev_title_zoom.png'))
    print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
