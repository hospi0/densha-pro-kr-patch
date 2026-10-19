# -*- coding: utf-8 -*-
r"""총 주행 거리 달성 화면(MDEC BS 정지 영상 11장, 640×480) 한글화 (2026-10-19)
  큰 제목(금색 붓글씨 + 갈색 그림자) «総走行距離 / ○○km達成！» → «총 주행 거리 / ○○km 달성!»
    지우기 = 금색 화소(크롭 y<175·x<520)를 넓혀 inpaint(배경: 별밤·파란 하늘). 그리기 = 나눔명조 ExtraBold, 줄마다 원본 금색 띠(줄 높이 비율), 그림자 (3,3).
  설명 상자(오른쪽 아래 회색 줄무늬) 흰 글씨 두 줄 → 줄마다 상자 색(글자 아닌 화소 중앙값)으로 지우고 나눔고딕 Bold 흰 글씨.
  굽기 = bs.encode(v3, 원래 자리 크기). python tools/distance_kr.py → work/gfx/prev_distance.png
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bs  # noqa: E402

SERIF = r'C:\claude\utils\font\nanum-myeongjo\NanumMyeongjoExtraBold.ttf'
GOTHIC = r'C:\claude\utils\font\nanum-gothic\NanumGothicBold.ttf'
FIRST = 0x96E6000
ITEMS = [  # (km, 설명 첫 줄, 둘째 줄)
    ('10000', '10000km는 지구 1/4 둘레의', '거리와 거의 같습니다.'),
    ('5000', '5000km는 삿포로·사세보 간', '왕복 거리와 거의 같습니다.'),
    ('3000', '3000km는 카고시마·쿠시로 간의', '거리와 거의 같습니다.'),
    ('2000', '2000km는 코쿠라·하코다테 간의', '거리와 거의 같습니다.'),
    ('1750', '1750km는 하카타·아키타 간의', '거리와 거의 같습니다.'),
    ('1500', '1500km는 도쿄·카고시마 간의', '거리와 거의 같습니다.'),
    ('1250', '1250km는 도쿄·오이타 간의', '거리와 거의 같습니다.'),
    ('1000', '1000km는 오사카·히로사키 간의', '거리와 거의 같습니다.'),
    ('750', '750km는 도쿄·쿠라시키 간의', '거리와 거의 같습니다.'),
    ('500', '500km는 도쿄·신하나마키 간의', '거리와 거의 같습니다.'),
    ('250', '250km는 도쿄·하마마츠 간의', '거리와 거의 같습니다.'),
]
TX, TY = 520, 175                                   # 큰 제목 범위(왼쪽 위)


def frames(pac):
    F = bs.frames(pac); ks = [k for k, f in enumerate(F) if f[0] >= FIRST][:len(ITEMS)]
    return [(F[k][0], F[k + 1][0] - F[k][0]) for k in ks]


def gold_mask(im):
    a = im[:TY, :TX].astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    gold = (r > 130) & (g > 80) & (r > b + 40) & (g > b + 10)
    near = cv2.dilate(gold.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    bright = (a.min(-1) > 150) | ((r > 170) & (g > 140) & (r > b + 20))   # 흰 반사광·연노랑(사용자: «거» 뒤 찌꺼기)
    return gold | (bright & near)


def lines(m):
    """금색 화소 줄 범위 두 개(제목 두 줄)"""
    rows = np.nonzero(m.sum(1) > 3)[0]
    gaps = np.nonzero(np.diff(rows) > 4)[0]
    cut = gaps[np.argmax(np.diff(rows)[gaps])] if len(gaps) else len(rows) // 2
    l1 = rows[:cut + 1]; l2 = rows[cut + 1:]
    out = []
    for L in (l1, l2):
        xs = np.nonzero(m[L.min():L.max() + 1].any(0))[0]
        out.append((L.min(), L.max(), xs.min(), xs.max()))
    return out


def band(im, m, box):
    """원본 금색 띠: 줄 높이 비율 0~1 → 색"""
    y0, y1, x0, x1 = box; out = []
    for y in range(y0, y1 + 1):
        v = im[y, x0:x1 + 1][m[y, x0:x1 + 1]]
        out.append(np.median(v, 0) if len(v) >= 3 else None)
    good = [i for i, c in enumerate(out) if c is not None]
    return np.array([out[min(good, key=lambda g: abs(g - i))] for i in range(len(out))], float)


def render(text, font, h_cap, width_max):
    """text 를 한 번에 그림 → 덮임(높이 = 글자 잉크 높이 h_cap 근처)"""
    f = ImageFont.truetype(font, int(h_cap * 1.35) * 4)
    l, t, r, b = f.getbbox(text)
    im = Image.new('L', (r - l + 8, b - t + 8)); ImageDraw.Draw(im).text((4 - l, 4 - t), text, font=f, fill=255)
    a = np.array(im); ys = np.nonzero(a.max(1))[0]; xs = np.nonzero(a.max(0))[0]
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = h_cap / a.shape[0]; W = int(round(a.shape[1] * sc))
    if W > width_max:
        W = width_max
    return np.array(Image.fromarray(a).resize((W, h_cap), Image.LANCZOS)) / 255.0


def spread(text, font, h_cap, width):
    """글자마다 같은 크기로(«한» 높이 기준) 그려 사이를 고르게 벌려 width 폭으로 — 원래 줄 폭을 덮게"""
    f = ImageFont.truetype(font, int(h_cap * 1.35) * 4)
    hl, ht, hr, hb = f.getbbox('한'); sc = h_cap / (hb - ht)
    gl = []
    for ch in text:
        if ch == ' ':
            gl.append(None); continue
        l, t, r, b = f.getbbox(ch)
        im = Image.new('L', (r - l + 8, hb - ht + (b - hb if b > hb else 0) + 8 + max(0, ht - t)))
        ImageDraw.Draw(im).text((4 - l, 4 - min(t, ht)), ch, font=f, fill=255)
        a = np.array(im); xs = np.nonzero(a.max(0))[0]; a = a[:, xs.min():xs.max() + 1]
        a = a[max(0, ht - min(t, ht)) + 4 - 4:]
        H = int(round(a.shape[0] * sc)); W = max(1, int(round(a.shape[1] * sc)))
        gl.append(np.array(Image.fromarray(a).resize((W, H), Image.LANCZOS)) / 255.0)
    H = max(g.shape[0] for g in gl if g is not None)
    nat = sum(g.shape[1] for g in gl if g is not None)
    tight = max(1, int(h_cap * 0.06))                    # 낱말 안 글자 사이
    nsp = text.count(' ')
    inner = tight * (len(text) - 1 - 2 * nsp)
    minsp = int(h_cap * 0.35)                            # 띄어쓰기는 빼지 않음(최소 폭)
    sp = max(minsp, (width - nat - inner) / max(1, nsp)) if nsp else 0   # 남는 폭은 띄어쓰기에만
    total = nat + inner + sp * nsp
    out = np.zeros((H, int(total) + 2)); x = 0.0
    for i, g in enumerate(gl):
        if g is None:
            x += sp - tight; continue
        xi = int(round(x)); out[:g.shape[0], xi:xi + g.shape[1]] = np.maximum(out[:g.shape[0], xi:xi + g.shape[1]], g)
        x += g.shape[1] + tight
    if out.shape[1] > width:                             # 넘치면 가로를 좁힘
        out = np.array(Image.fromarray((out * 255).astype(np.uint8)).resize((width, H), Image.LANCZOS)) / 255.0
    return out

def earth_mask(im):
    """10000km 장 지구 = 원(가장자리 어두운 쪽까지 잡아 맞춤) — 위쪽은 제목 글자 뒤 y≈118 까지 올라감(타원은 위를 잘랐음)"""
    g = (cv2.GaussianBlur(im, (7, 7), 0).max(-1) > 22).astype(np.uint8)
    g[:185] = 0; g[365:] = 0; g[:, :150] = 0; g[:, 520:] = 0; g[360:, 250:] = 0      # 제목·설명 상자 빼고
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    _, lab, st, _ = cv2.connectedComponentsWithStats(g); k = 1 + np.argmax(st[1:, 4])
    P = []
    for y in range(190, 360):
        xs = np.nonzero(lab[y] == k)[0]
        if len(xs):
            P += [(xs.min(), y), (xs.max(), y)]
    P = np.array(P, float)
    cx, cy, c = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)[0]
    r = np.sqrt(c + cx * cx + cy * cy)                  # 원본: (319, 240) r 122
    yy, xx = np.mgrid[:im.shape[0], :im.shape[1]]
    return np.hypot(xx - cx, yy - cy) <= r + 1


def title(im, km):
    img = im.copy(); m = gold_mask(im)
    L = lines(m)
    bands = [band(im, m, b) for b in L]
    er = np.zeros(img.shape[:2], np.uint8)
    sh = np.zeros_like(m); sh[6:, 6:] = m[:-6, :-6]          # 갈색 그림자(오른쪽 아래로 밀림)까지
    big = cv2.dilate((m | sh).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    E = np.zeros(img.shape[:2], bool); E[:TY, :TX] = big > 0
    # 배경(파란 하늘·별밤)은 위아래로만 바뀜 → 제목 범위 안 «줄 배경색과 다른 화소»를 전부 그 줄 색으로 깨끗하게(사용자: 흐리게 메우지 말고 밀 것, 찌꺼기 없이)
    a = im.astype(float)
    bgc = np.full((TY, 3), np.nan); good = []
    for y in range(TY):
        v = a[y, :TX][~E[y, :TX]]
        if len(v) >= 40:                                # 표본 적은 줄(글자가 거의 덮음)은 그림자 색이 섞임 → 위아래에서 이어 받음
            bgc[y] = np.median(v, 0); good.append(y)
    for y in range(TY):
        if np.isnan(bgc[y, 0]):
            bgc[y] = bgc[min(good, key=lambda g: abs(g - y))]
    bgc = np.array([np.median(bgc[max(0, y - 4):y + 5], 0) for y in range(TY)])   # 세로로 부드럽게(9줄 중앙값)
    diff = np.zeros(img.shape[:2], bool)
    diff[:TY, :TX] = np.abs(a[:TY, :TX] - bgc[:, None, :]).max(-1) > 28
    # 지도·지구·화살표 = 제목 범위 밖(아래·오른쪽)까지 이어진 덩어리 → 남김
    n, lab = cv2.connectedComponents(((diff & ~E) | (~np.pad(np.ones((TY, TX), bool), ((0, 480 - TY), (0, 640 - TX))))).astype(np.uint8), connectivity=8)
    keep_lab = set(np.unique(lab[TY:, :])) | set(np.unique(lab[:, TX:]))
    obj = np.isin(lab, list(keep_lab - {0})) & (diff & ~E)
    obj_near = cv2.dilate(obj.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    wipe = (E | diff) & ~obj
    wipe[TY:, :] = False; wipe[:, TX:] = False
    img = img.astype(float)
    flat = wipe & ~obj_near
    for y in range(TY):
        img[y][flat[y]] = bgc[y]
    rest = wipe & obj_near                               # 물체에 붙은 곳만 주변으로 메움
    if km == '10000':                                    # 지구 안은 글자·그림자만 지우고 지구 무늬로 메움(사용자: 지구 윗부분이 깎임)
        ins = np.zeros_like(rest); ins[:TY, :TX] = earth_mask(im)[:TY, :TX]
        txt = np.zeros_like(rest); txt[:TY, :TX] = cv2.dilate((m | sh).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        img[ins & ~txt] = im[ins & ~txt]
        rest = (rest & ~ins) | (ins & txt)
    rest = rest.astype(np.uint8) * 255
    if rest.any():
        img = cv2.inpaint(np.clip(img, 0, 255).astype(np.uint8), rest, 5, cv2.INPAINT_TELEA).astype(float)
    if km != '10000' and MAPS:                           # 원본 «！» 자리(지도 해안)만 다른 장의 멀쩡한 지도로(사용자: 국지적으로 복사)
        img = bang_patch(img, m, L, wipe, im)
    for (y0, y1, x0, x1), bd, text in zip(L, bands, ('총 주행 거리', '%skm 달성!' % km)):
        h = y1 - y0 + 1
        want = x1 - x0 + 1
        c = render(text, SERIF, h, min(TX - x0 - 8, want))   # 줄 통째로 한 번에, 원래 줄 폭 안(10000km 줄이 오른쪽 끝에 붙던 것)
        if c.shape[1] < want * 0.9:                      # 원래 줄보다 많이 짧으면 가로로 최대 1.2배(지운 자리 덮기)
            W = int(min(want, c.shape[1] * 1.2))
            c = np.array(Image.fromarray((c * 255).astype(np.uint8)).resize((W, c.shape[0]), Image.LANCZOS)) / 255.0
        hh, ww = c.shape
        col = np.array([bd[min(len(bd) - 1, int(i * len(bd) / hh))] for i in range(hh)])[:, None, :]
        sy, sx = y0 + 3, x0 + 3                          # 그림자
        sh = img[sy:sy + hh, sx:sx + ww]
        sh[:] = sh * (1 - 0.85 * c[..., None]) + np.array([35, 25, 8.]) * 0.85 * c[..., None]
        reg = img[y0:y0 + hh, x0:x0 + ww]
        reg[:] = reg * (1 - c[..., None]) + col * c[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def caption(im, l1, l2):
    img = im.astype(float)
    a = im[360:, 250:].astype(int)
    white = a.min(-1) > 190
    ys = np.nonzero(white.sum(1) > 2)[0]; xs = np.nonzero(white.any(0))[0]
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    light = np.zeros_like(white); light[y0 - 1:y1 + 2] = a[y0 - 1:y1 + 2].min(-1) > 110   # 옅은 가장자리까지(상자 줄무늬는 더 어두움)
    txt = cv2.dilate((white | light).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    reg = img[360:, 250:]
    for y in range(y0 - 2, min(a.shape[0], y1 + 3)):
        row = reg[y]; ok = ~txt[y]; ok[:max(0, x0 - 4)] = False
        if ok.any():
            row[txt[y]] = np.median(row[ok], 0)
    rows = np.nonzero(white.sum(1) > 2)[0]
    gap = np.nonzero(np.diff(rows) > 2)[0]
    cut = gap[np.argmax(np.diff(rows)[gap])] if len(gap) else len(rows) // 2
    L = [(rows[:cut + 1].min(), rows[:cut + 1].max()), (rows[cut + 1:].min(), rows[cut + 1:].max())]
    for (ly0, ly1), t in zip(L, (l1, l2)):
        h = ly1 - ly0 + 1
        c = render(t, GOTHIC, int(h * 0.85), reg.shape[1] - x0 - 6)
        hh, ww = c.shape; oy = ly0 + (h - hh) // 2
        sub = reg[oy:oy + hh, x0:x0 + ww]
        sub[:] = sub * (1 - c[..., None]) + 250 * c[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


MAPS = []                                            # 지도 장(5000~250) 원본 — build_all 이 채움


def bang_patch(img, m, L, wipe, im):
    """원본 «！»(둘째 줄 맨 오른쪽 덩어리)+그림자 둘레의 지운 화소를, 그 자리에 글자가 없던 다른 지도 장 원본의 중앙값으로 덮음"""
    y0, y1 = L[1][:2]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m[y0:y1 + 1].astype(np.uint8))
    j = 1 + np.argmax(st[1:, 0] + st[1:, 2]); bx0, bx1 = st[j, 0], st[j, 0] + st[j, 2]    # «！» 아래 점
    band = np.zeros(img.shape[:2], bool); band[max(0, y0 - 8):TY, bx0 - 12:min(TX, bx1 + 26)] = True
    tgt = band & wipe
    stack, ok = [], []
    for d in MAPS:
        dm = gold_mask(d); dsh = np.zeros_like(dm); dsh[6:, 6:] = dm[:-6, :-6]
        clean = np.ones(img.shape[:2], bool)
        clean[:TY, :TX] = ~cv2.dilate((dm | dsh).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        stack.append(d.astype(float)); ok.append(clean)
    stack = np.array(stack); ok = np.array(ok)
    left = np.zeros(img.shape[:2], np.uint8)
    for y, x in zip(*np.nonzero(tgt)):
        v = stack[ok[:, y, x], y, x]
        if len(v):
            img[y, x] = np.median(v, 0)
        else:
            left[y, x] = 255                             # 어느 장에서도 글자에 가렸던 몇 화소만 둘레로 메움
    if left.any():
        img = cv2.inpaint(np.clip(img, 0, 255).astype(np.uint8), left, 3, cv2.INPAINT_TELEA).astype(float)
    a = im.astype(int); red = (a[..., 0] > 140) & (a[..., 1] < 90) & (a[..., 2] < 90)   # 이 장 자신의 빨간 화살표는 원본 그대로
    red = cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & tgt & (m_full(m) == 0)
    img[red] = im[red]
    ys, xs = np.nonzero(cv2.dilate(((a[..., 0] > 140) & (a[..., 1] < 90) & (a[..., 2] < 90)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & band & (m_full(m) == 0))
    if len(xs) > 30:                                     # 원본 «！»에 가렸던 화살표 몸통은 보이는 조각을 직선으로 이음
        k, b = np.polyfit(xs, ys, 1); col = np.median(im[ys, xs].astype(float), 0)
        for x in range(xs.min(), xs.max() + 1):
            for y in range(int(k * x + b) - 2, int(k * x + b) + 3):
                if 0 <= y < 480 and tgt[y, x] and abs(y - (k * x + b)) <= 2.0:
                    img[y, x] = col
    return img


def m_full(m):
    out = np.zeros((480, 640), np.uint8); out[:TY, :TX] = m; return out


def build_all(pac):
    out = []
    MAPS[:] = [bs.decode(pac[o:o + n], 640, 480) for (o, n), it in zip(frames(pac), ITEMS) if it[0] != '10000']
    for (o, n), (km, l1, l2) in zip(frames(pac), ITEMS):
        im = bs.decode(pac[o:o + n], 640, 480)
        out.append((o, n, im, caption(title(im, km), l1, l2)))
    return out


def preview():
    pac = open(bs.PAC, 'rb').read(); res = build_all(pac)
    rows = [np.concatenate([a, np.full((480, 4, 3), 255, np.uint8), b], 1) for _, _, a, b in res[:3]]
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_distance.png')
    from title_kr import save
    save(Image.fromarray(np.concatenate(rows, 0)), p); print('미리보기', p)
    allb = [b for _, _, _, b in res]
    sheet = np.concatenate([np.concatenate([np.array(Image.fromarray(x).resize((320, 240))) for x in allb[r * 4:r * 4 + 4]] + [np.zeros((240, 320 * (4 - len(allb[r * 4:r * 4 + 4])), 3), np.uint8)], 1) for r in range(3)], 0)
    save(Image.fromarray(sheet), os.path.join(ROOT, 'work', 'gfx', 'prev_distance_all.png'))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
