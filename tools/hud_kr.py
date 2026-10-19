# -*- coding: utf-8 -*-
r"""게임 중 HUD 아틀라스 글자 한글화 (2026-10-19) — 공용 읽기·쓰기 = tools/hud.py
  아틀라스 4bpp 색인 = 2비트 층 두 개(CLUT 가 위 2비트(H)나 아래 2비트(L) 중 하나만 봄) → 그 조각 층만 다시 그리고 다른 층은 보존.
  원본 꼴: 획 = 단계 3(밝음), 획 둘레 1px = 단계 1(어두움), 덮임 중간 = 2. 작은 라벨(8px)은 둘레 대신 오른쪽 아래 그림자 2.
  글꼴 = 나눔고딕 ExtraBold(사용자: 작은 글자도 버튼 글꼴로 통일).
  ITEMS: (레코드 주소, 한국어, 꼴) — 칸(u,v,w,h)은 레코드 그대로, 글자는 칸 안 왼쪽/가운데.
    낱말 조합(次駅+到着+時刻 등)은 칸 경계 그대로 두고 칸마다 한 낱말 → 조합하면 «다음 도착 시각».
  python tools/hud_kr.py → work/gfx/prev_hud_top.png (원본|한글, 4배)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hud as H  # noqa: E402

FONT = r'C:\claude\utils\font\nanum-gothic\NanumGothicExtraBold.ttf'
SS = 8
LABEL_FONT = 'galmuri'                      # 8px 라벨 = 갈무리7 비트맵(사용자 2026-10-19 선택 — 나눔은 8px 에서 뭉개짐)
GALMURI7 = r'C:\claude\utils\font\Galmuri-v2.40.3\Galmuri7.bdf'

# (레코드, 한국어, 꼴, 맞춤) — 꼴: 'label' 8px 라벨(그림자) / 'ring' 둘레 1px
ITEMS = [
    (0x3C2A0, '다음', 'label', 'l'), (0x3C2AC, '통과', 'label', 'l'), (0x3C2B8, '시각', 'label', 'l'),   # 次駅·通過·時刻
    (0x3C2C4, '현재', 'label', 'l'),                                                                  # 現在
    (0x3C51C, '남은 시간', 'ring', 'c'), (0x3C528, '정지 위치', 'ring', 'c'),
    (0x3C45C, '초', 'ring', 'c'), (0x3C4E0, '남은', 'ring', 'c'), (0x3C4EC, '까지', 'ring', 'c'),   # あと = 숫자 앞 고정 조각, «잔여» 사용자 거절 → 일단 «남은»
    (0x3C558, '해제', 'ring', 'c'), (0x3C714, '비상', 'ring', 'c'), (0x3C750, '끔', 'ring', 'c'),
]
FULL = (0x3C294, '도착')                    # 次駅到着時刻 한 조각(0~59) = 次駅(0~18)+到着(19~38)+時刻(39~59) — 到着 몫만 따로 그림
SPLIT = {0x3C294: (19, 20)}                 # 到着 칸 = u 19 부터 폭 20


def cover(text, h_ink, maxw, stretch=False):
    """한 번에 그림 → 덮임(높이 h_ink = '한' 잉크 높이), 넘치면 가로로 좁힘(85% 까지)"""
    f = ImageFont.truetype(FONT, int(h_ink * SS * 1.3))
    hl, ht, hr, hb = f.getbbox('한')
    l, t, r, b = f.getbbox(text)
    top = min(t, ht); bot = max(b, hb)
    im = Image.new('L', (r - l + 8, bot - top + 8))
    ImageDraw.Draw(im).text((4 - l, 4 - top), text, font=f, fill=255)
    a = np.array(im)[4 + ht - top:4 + hb - top]
    cols = np.nonzero(a.max(0))[0]; a = a[:, cols.min():cols.max() + 1]
    W = int(round(a.shape[1] * h_ink / (hb - ht)))
    if stretch and W < maxw:                             # 원본처럼 칸 폭을 채움(가로 1.2배까지)
        W = min(maxw, int(W * 1.2))
    if W > maxw:
        assert W * 0.85 <= maxw, '%s 폭 %d > %d' % (text, W, maxw)
        W = maxw
    return np.array(Image.fromarray(a).resize((W, h_ink), Image.BOX)).astype(float) / 255


def galmuri(text):
    """갈무리7 비트맵(한글 7×7, 전진 8, 공백 3) → 덮임 0/1"""
    import bdf
    if not hasattr(galmuri, 'g'):
        galmuri.g = bdf.load(GALMURI7)
    g, asc = galmuri.g
    cols = []
    for ch in text:
        a, adv = bdf.render(g, asc, ord(ch), 8, 9, asc)
        cols.append(np.array(a, float)[:, :adv] if adv else np.zeros((9, 0)))
    a = np.concatenate(cols, 1)
    ys = np.nonzero(a.max(1))[0]; xs = np.nonzero(a.max(0))[0]
    return a[ys.min():ys.max() + 1, :xs.max() + 1]


def label_cover(text, hi, w):
    return galmuri(text) if LABEL_FONT == 'galmuri' else cover(text, hi, w)


def glyph_levels(cov, style, green=False):
    """덮임 → 단계(0~3)"""
    ink = cov >= 0.5
    lv = np.zeros(cov.shape, np.uint8)
    if style == 'label':
        out = np.zeros((cov.shape[0] + 1, cov.shape[1] + 1), np.uint8)
        sh = np.zeros_like(out, bool); sh[1:, 1:] = ink
        out[sh] = 2
        core = np.zeros_like(out, bool); core[:-1, :-1] = ink
        out[core] = 3
        part = np.zeros_like(out, bool); part[:-1, :-1] = (cov >= 0.25) & ~ink
        out[part & (out == 0)] = 2
        return out
    out = np.zeros((cov.shape[0] + 2, cov.shape[1] + 2), np.uint8)
    core = np.zeros_like(out, bool); core[1:-1, 1:-1] = ink
    ring = np.zeros_like(core)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ring |= np.roll(np.roll(core, dy, 0), dx, 1)
    out[ring] = 1; out[core] = 3
    if green:                                            # 초록 글자만 중간 단계(원본도 2·3 섞임), 흰 글자는 3·1 만(회색 끼면 지저분)
        part = np.zeros_like(core); part[1:-1, 1:-1] = (cov >= 0.25) & ~ink
        out[part] = 2
    return out


def layer_of(clut, c):
    p = H.pal(clut, c)[:, :3].astype(int).sum(1)
    return 'H' if len(set(p[[0, 4, 8, 12]])) > 1 else 'L'


def put(px, u, v, w, h, lay, lv, align):
    reg = px[v:v + h, u:u + w]
    keep = (reg & 3) if lay == 'H' else (reg & 12)
    new = np.zeros((h, w), np.uint8)
    hh, ww = lv.shape
    assert hh <= h and ww <= w, (u, v, w, h, lv.shape)
    x0 = 0 if align == 'l' else (w - ww) // 2
    y0 = (h - hh) // 2
    new[y0:y0 + hh, x0:x0 + ww] = lv
    reg[:] = keep | ((new << 2) if lay == 'H' else new)


def size_for(style, h):
    return h - 1 if style == 'label' else h - 2


def make(pac, exe):
    px, clut = H.load(pac); orig = px.copy()
    recs = {r['o']: r for r in H.records(exe)}
    for o, text, style, align in ITEMS:
        if text is None:
            continue
        r = recs[o]; lay = layer_of(clut, r['clut'])
        hi = size_for(style, r['h'])
        w = r['w'] - (1 if style == 'label' else 2)
        if style == 'label':
            put(px, r['u'], r['v'], r['w'], r['h'], lay, glyph_levels(label_cover(text, hi, w), style), align); continue
        import hud_px as HP                               # 갈무리 비트맵(B안, 사용자 선택) — 나눔 축소·중간 단계 폐기
        lv = HP.word(text, r['w'], r['h'], 'B' if r['h'] > 14 else 'S', align)
        put(px, r['u'], r['v'], r['w'], r['h'], lay, lv, 'l')
    o, text = FULL; r = recs[o]; u, w = SPLIT[o]
    lv = glyph_levels(label_cover(text, r['h'] - 1, w - 1), 'label')
    put(px, u, r['v'], w, r['h'], layer_of(clut, r['clut']), lv, 'l')
    return px, orig, clut, recs


def apply(pac):
    """빌드: HUD 아틀라스 7 사본 전부에 같은 화소"""
    px = make(bytes(pac), open(H.EXE, 'rb').read())[0]
    H.store(pac, px)
    return len(ITEMS) + 1


PREV = [  # 미리보기 조합(레코드 목록)
    [0x3C294], [0x3C2A0, 0x3C2AC, 0x3C2B8], [0x3C2C4, 0x3C2D0], [0x3C51C], [0x3C528], [0x3C45C], [0x3C4E0], [0x3C4EC],
    [0x3C558], [0x3C714], [0x3C750],
]


def preview():
    global LABEL_FONT
    pac = open(H.PAC, 'rb').read(); exe = open(H.EXE, 'rb').read()
    px, orig, clut, recs = make(pac, exe)
    rows = []
    for i, g in enumerate(PREV):
        gr = [recs[o] for o in g]
        a = H.compose(orig, clut, gr); b = H.compose(px, clut, gr)
        sep = np.full((a.shape[0], 4, 4), 255, np.uint8)
        parts = [a, sep, b]                                    # (예전 나눔/갈무리 비교 열 뺌 — 라벨은 갈무리7 로 확정)
        rows.append(np.concatenate(parts, 1))
    W = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((2, 2), (0, W - r.shape[1]), (0, 0))) for r in rows], 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_hud_top2.png')
    Image.fromarray(sheet).resize((sheet.shape[1] * 4, sheet.shape[0] * 4), Image.NEAREST).save(p)
    print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
