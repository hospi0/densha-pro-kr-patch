# -*- coding: utf-8 -*-
r"""운전 승무(시각표) 화면 작은 낱말 한글화 — 열차 이름·종류·天気·乗務記録·号·系·形·세로 노선 이름·次のページ (2026-10-18)
  그림 = TXS 7E54800 tpage A. ★한 칸에 두 그림(층 P·Q)을 값 0~8 로 겹침:
    층 P(CLUT 3228 검정·3268·32A8·34A8·35E8): 값 1 = 획, 2 = 옅음 / 층 Q(32E8·3328·3368·33A8): 값 4 = 획, 3 = 옅음
    (P, Q) → 값: (s,0)1 (a,0)2 (0,a)3 (0,s)4 (s,s)5 (a,s)6 (s,a)7 (a,a)8   (s = 획, a = 옅음)
  칸 크기·레코드 그대로(제자리) — 칸 폭 안에 한국어를 한 번에 그림(넘치면 가로를 좁힘), 세로 낱말은 음절을 세로로 쌓음.
  python tools/unten_kr.py → work/gfx/prev_unten.png
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import btn_kr as BT  # noqa: E402
import timetable_kr as TK  # noqa: E402
import tt_stations as TS  # noqa: E402

P = {0x3228, 0x3268, 0x32A8, 0x34A8, 0x35E8}
Q = {0x32E8, 0x3328, 0x3368, 0x33A8}
ENC = {(0, 0): 0, (1, 0): 1, (2, 0): 2, (0, 2): 3, (0, 1): 4, (1, 1): 5, (2, 1): 6, (1, 2): 7, (2, 2): 8}   # (P, Q), 1 = 획·2 = 옅음
DEC = {v: k for k, v in ENC.items()}
# (u, v, CLUT 대표) → 한국어. 같은 칸을 다른 색 CLUT 로 쓰는 것은 층만 같으면 한 번만
H = {  # 가로 낱말
    (0, 0, 'P'): '칸쿠쾌속', (88, 0, 'P'): '키슈지・', (144, 0, 'P'): '하쿠타카', (200, 0, 'P'): '사가노',
    (0, 24, 'P'): '라이초', (32, 24, 'P'): '슈퍼라이초', (96, 24, 'P'): '라이너', (144, 24, 'P'): '아사시오',
    (200, 24, 'P'): '소닉', (0, 48, 'P'): '하쿠초', (34, 48, 'P'): '하야부사', (88, 48, 'P'): '썬더버드',
    (152, 48, 'P'): '오도리코', (200, 48, 'P'): '니치린', (0, 72, 'P'): '슈퍼뷰', (64, 72, 'P'): '윙',
    (112, 72, 'P'): '구쾌', (112, 72, 'Q'): '야마토지쾌속', (152, 72, 'P'): '액티', (176, 72, 'Q'): '오도리코1호',
    (200, 72, 'P'): '슈푸르', (0, 96, 'P'): '시가이아5호', (64, 96, 'P'): '홀리데이', (112, 96, 'Q'): '타카야마',
    (128, 96, 'P'): '제1코다마', (176, 96, 'Q'): '아키타릴레이', (192, 96, 'P'): '코마쿠사',
    (0, 120, 'P'): '야마토지', (56, 120, 'P'): '카마쿠라', (112, 120, 'Q'): '승무 기록', (128, 120, 'P'): '키타구니',
    (186, 120, 'P'): '토키', (216, 120, 'P'): '탄고',
    (152, 24, 'Q'): '계', (168, 24, 'Q'): '형', (192, 24, 'Q'): '날씨', (128, 48, 'Q'): '호',
}
BOXED = {(112, 72, 'P')}                          # 区快: 테두리 상자 남기고 안쪽만
V = {  # 세로 낱말(폭 12·16)
    (0, 24, 'Q'): '아키타・토호쿠신칸센', (12, 24, 'Q'): '야마노테선 내선순환', (24, 24, 'Q'): '케이힌토호쿠선 북행',
    (36, 24, 'Q'): 'JR쿄토선', (48, 24, 'Q'): 'JR사가노선', (60, 24, 'Q'): '카고시마본선', (72, 24, 'Q'): '토카이도선',
    (84, 24, 'Q'): '호쿠호쿠・조에츠선', (96, 24, 'Q'): '오사카순환선 내선순환', (36, 104, 'Q'): '남행',
    (240, 24, 'Q'): '다음 페이지',
}


def cells(pac):
    """34번 파일 레코드 → {(u, v, 층): (w, h, 레코드들)} (tpage A, 층 P/Q CLUT 만) — 같은 칸 여러 크기면 가장 큰 것"""
    F = pac[TS.FILE34:TS.FILE34 + 0xE000]; out = {}
    for o in range(0x936E, 0xE000 - 16, 2):
        w = struct.unpack_from('<8H', F, o)
        if w[5] != 0xA or not (0 < w[3] <= 256 and 0 < w[4] <= 160) or (w[0] & 255) + w[3] > 256 or (w[0] >> 8) + w[4] > 256:
            continue
        L = 'P' if w[6] in P else 'Q' if w[6] in Q else None
        if L is None:
            continue
        k = (w[0] & 255, w[0] >> 8, L)
        a = out.setdefault(k, [0, 0, []])
        a[0] = max(a[0], w[3]); a[1] = max(a[1], w[4]); a[2].append(o)
    return out


def lv(cov):
    return np.where(cov >= 0.5, 1, np.where(cov >= 0.2, 2, 0)).astype(np.uint8)


def ink_rows(layer):
    ys = np.nonzero(layer.any(1))[0]
    return (ys.min(), ys.max()) if len(ys) else (2, layer.shape[0] - 3)


def horiz(text, w, h_ink):
    xs = 1.0
    while True:
        c, _ = BT.cover(text, h_ink, xs)
        if c.shape[1] <= w or xs <= 0.5:
            break
        xs = max(0.5, xs * w / c.shape[1] - 0.01)
    if c.shape[1] > w:
        c = np.array(Image.fromarray((c * 255).astype(np.uint8)).resize((w, c.shape[0]), Image.BOX)) / 255
    return lv(c)


G11 = 'C:/claude/utils/font/Galmuri-v2.40.3/Galmuri9.bdf'   # 세로 배지 안쪽 폭 12 — 11px 는 곁가지가 가장자리에 붙어 «하→히» (사용자 지적)
_g11 = []


def bdf_syl(t):
    """갈무리11 굵은 비트맵 음절(좁은 세로 배지용, 작은 윤곽 글꼴은 흐려짐) → 1 획 배열"""
    import bdf
    if not _g11:
        _g11.append(bdf.load(G11))
    g, asc = _g11[0]
    cell, adv = bdf.render(g, asc, ord(t), 12, 10, 9)
    a = np.array(cell, np.uint8)
    xs = np.nonzero(a.any(0))[0]
    return a[:, xs.min():xs.max() + 1] if len(xs) else a


def vert(text, w, h):
    """음절을 세로로 쌓음(공백 = 반 칸) — 칸 높이에 맞춰 음절 크기를 정함, 폭 14 이하면 비트맵 글꼴"""
    syl = [t for t in text]
    n = sum(0.5 if t == ' ' else 1 for t in syl)
    step = min(w + 2, h / n)
    size = int(min(w - 1, step - 1))
    out = np.zeros((h, w), np.uint8); y = 0.0
    for t in syl:
        if t == ' ':
            y += step / 2; continue
        if w <= 14 and step >= 10:
            c = bdf_syl(t)
        else:
            c, _ = BT.cover(t, max(6, size), 1.0); c = lv(c)
        hh, ww = c.shape
        ww = min(ww, w); yy = int(round(y + (step - hh) / 2)); xx = (w - ww) // 2
        yy = max(0, min(yy, h - 1)); hh = min(hh, h - yy)
        out[yy:yy + hh, xx:xx + ww] = np.maximum(out[yy:yy + hh, xx:xx + ww], c[:hh, :ww])
        y += step
    return out


def make(pac, px=None):
    if px is None:
        px = TK.load_px(pac)
    C = cells(pac); done = []
    for table, kind in ((H, 'h'), (V, 'v')):
        for (u, v, L), t in table.items():
            assert (u, v, L) in C, ('칸 없음', u, v, L)
            w, h, _ = C[(u, v, L)]
            reg = px[v:v + h, u:u + w]
            assert reg.max() <= 8, (u, v, reg.max())
            pq = np.vectorize(lambda a: DEC[a][0 if L == 'P' else 1])(reg)
            y0, y1 = ink_rows(pq)
            new = horiz(t, w - 1, max(8, y1 - y0 + 1)) if kind == 'h' else vert(t, w, h)
            lay = np.zeros((h, w), np.uint8)
            if (u, v, L) in BOXED:                          # 상자(잉크 범위 테두리 2px) 보존, 안쪽에 그림
                ys, xs = np.nonzero(pq); by0, by1, bx0, bx1 = ys.min(), ys.max(), xs.min(), xs.max()
                lay[:] = pq; lay[by0 + 2:by1 - 1, bx0 + 2:bx1 - 1] = 0
                new = horiz(t, bx1 - bx0 - 4, by1 - by0 - 5)
                hh, ww = new.shape; oy = by0 + 3 + (by1 - by0 - 5 - hh) // 2; ox = bx0 + 3 + (bx1 - bx0 - 5 - ww) // 2
                lay[oy:oy + hh, ox:ox + ww] = new
            elif kind == 'h':
                hh, ww = new.shape; oy = max(0, min(h - hh, y0 + (y1 - y0 + 1 - hh) // 2))
                lay[oy:oy + hh, :ww] = new
            else:
                lay[:] = new
            other = np.vectorize(lambda a: DEC[a][1 if L == 'P' else 0])(reg)
            pair = (lambda a, b: (a, b)) if L == 'P' else (lambda a, b: (b, a))
            reg[:] = np.vectorize(lambda a, b: ENC[pair(a, b)])(lay, other)
            done.append((u, v, L, w, h))
    return px, done


# ── tpage C 배지·판, 難易度 ────────────────────────────────────────────
# 꼴: 'ko' = 색 채운 판에 글자를 비워 냄(채움 f, 가장자리 a, 글자 = 0) · 'txt' = 보통 글자(획 1·옅음 2) · 'frame' = 테두리 안 보통 글자
#     'outline' = 1px 윤곽선 배지 안 보통 글자(下り) · 'plate' = 숫자판(글자 7, 가름줄 11·12줄은 4·6) · 'nan' = 흰 5·회색 6 글자(오른쪽 점은 그대로)
C_ITEMS = [  # (tpage, u, v, w, h, 꼴, 한국어, 세로)
    (0xC, 0, 64, 36, 24, 'ko', '특급', False), (0xC, 36, 64, 36, 24, 'txt', '임시', False), (0xC, 120, 64, 36, 24, 'ko', '특쾌', False),
    (0xC, 0, 88, 52, 24, 'ko', '코마치', False), (0xC, 52, 88, 62, 24, 'ko', '야마비코', False), (0xC, 152, 88, 36, 24, 'ko', '급행', False),
    (0xC, 240, 112, 16, 40, 'ko', '상행', True), (0xC, 240, 152, 16, 40, 'outline', '하행', True),
    (0xC, 168, 232, 36, 24, 'frame', '쾌속', False), (0xC, 208, 232, 36, 24, 'frame', '신쾌', False),
    (0xC, 120, 232, 24, 24, 'plate', '계', False), (0xC, 144, 232, 24, 24, 'plate', '형', False),
    (0xA, 0, 152, 48, 16, 'nan', '난이도', False),
]


def text_lv(t, w, h, vertical):
    """칸(w×h) 안에 들어가는 글자 → 1 획·2 옅음 (가운데 정렬된 w×h 배열)"""
    out = np.zeros((h, w), np.uint8)
    if vertical:
        out[:] = vert(t, w, h)
        return out
    c = horiz(t, w, h)
    hh, ww = c.shape; oy = (h - hh) // 2; ox = (w - ww) // 2
    out[oy:oy + hh, ox:ox + ww] = c
    return out


def do_item(px, it):
    tp, u, v, w, h, kind, t, vertical = it
    x0 = (tp - 0xA) * 256
    reg = px[v:v + h, x0 + u:x0 + u + w]
    o = reg.copy()
    if kind == 'ko':
        nz = o[o > 0]; f = np.bincount(nz).argmax(); a = [k for k in np.unique(nz) if k != f]
        a = a[0] if a else f
        inner = np.zeros_like(o, bool)
        for y in range(h):                               # 판 모양 = 줄마다 처음~끝 0 아닌 화소
            xs = np.nonzero(o[y])[0]
            if len(xs):
                inner[y, xs.min():xs.max() + 1] = True
        ring = inner & ~np.pad(inner, 1)[2:, 1:-1] | inner & ~np.pad(inner, 1)[:-2, 1:-1] | inner & ~np.pad(inner, 1)[1:-1, 2:] | inner & ~np.pad(inner, 1)[1:-1, :-2]
        body = inner & ~ring
        reg[body] = f
        ys, xs = np.nonzero(body)
        if vertical:                                     # 화살표 꼭지 빼고 «온전한 폭» 줄만 글자 자리
            wid = body.sum(1); full = np.nonzero(wid >= wid.max() - 1)[0]
            ys = ys[(ys >= full.min()) & (ys <= full.max())]
        bw, bh = xs.max() - xs.min() - 1, ys.max() - ys.min() - 1
        lvx = text_lv(t, bw, bh if not vertical else bh, vertical)
        sub = reg[ys.min() + 1:ys.min() + 1 + bh, xs.min() + 1:xs.min() + 1 + bw]
        sub[lvx == 1] = 0; sub[lvx == 2] = a
    elif kind == 'outline':
        inner = np.zeros_like(o, bool)
        for y in range(h):
            xs = np.nonzero(o[y])[0]
            if len(xs) >= 2:
                inner[y, xs.min() + 1:xs.max()] = True
        reg[inner] = 0                                   # 윤곽선만 남김
        wid = inner.sum(1); full = np.nonzero(wid >= wid.max() - 1)[0]
        xs = np.nonzero(inner[full[0]])[0]
        y0, y1 = full.min() + 4, full.max() - 1; x0_, x1_ = xs.min() + 1, xs.max() - 1   # 3px 내림(사용자: 위에 붙음)
        lvx = text_lv(t, x1_ - x0_ + 1, y1 - y0 + 1, True)
        reg[y0:y1 + 1, x0_:x1_ + 1] |= lvx
    elif kind in ('txt', 'frame'):
        if kind == 'frame':
            keep = np.zeros_like(o, bool); keep[:2] = keep[-2:] = True; keep[:, :2] = keep[:, -2:] = True
            ys, xs = np.nonzero(o)                       # 테두리 = 잉크 바깥 2px
            keep = np.zeros_like(o, bool)
            keep[ys.min():ys.min() + 2] = keep[ys.max() - 1:ys.max() + 1] = True
            keep[:, xs.min():xs.min() + 2] = keep[:, xs.max() - 1:xs.max() + 1] = True
            reg[~keep] = 0
            iw, ih = xs.max() - xs.min() - 5, ys.max() - ys.min() - 5
            lvx = text_lv(t, iw, ih, False)
            reg[ys.min() + 3:ys.min() + 3 + ih, xs.min() + 3:xs.min() + 3 + iw] |= lvx
        else:
            reg[:] = 0; reg |= text_lv(t, w - 2, h - 4, vertical) if False else 0
            lvx = text_lv(t, w - 2, h - 4, vertical); reg[2:h - 2, 1:w - 1] = lvx
    elif kind == 'plate':
        for y in range(h):
            row = reg[y]; gv = {7} | ({4} if y == 11 else set()) | ({6} if y == 12 else set())
            bg = np.bincount([c for c in o[y] if c not in gv] or [1]).argmax()
            row[np.isin(o[y], list(gv)) & (np.arange(w) > 2) & (np.arange(w) < w - 3)] = bg
        c = horiz(t, w - 8, 17)
        hh, ww = c.shape; oy = 3 + (17 - hh) // 2; ox = (w - ww) // 2          # 가운데(사용자: 위로 붙이지 말 것)
        for y in range(hh):
            Y = oy + y; g = 7                                   # 가름줄(11·12줄) 위 획도 흰색 — 원본처럼 어둡게 하면 ㅖ 가로획이 사라짐
            reg[Y, ox:ox + ww][c[y] == 1] = g               # 진한 획만(옅은 가장자리까지 칠하면 ㅖ 두 획이 뭉쳐 «게»)
    elif kind == 'nan':
        m = np.isin(o, [5, 6]); reg[m] = 0
        ys = np.nonzero(m.any(1))[0]
        c = horiz(t, w, ys.max() - ys.min() + 1)
        hh, ww = c.shape; oy = ys.min() + (ys.max() - ys.min() + 1 - hh) // 2
        sub = reg[oy:oy + hh, :ww]; sub[c == 1] = 5; sub[c == 2] = 6


def make_c(pac, px):
    for it in C_ITEMS:
        do_item(px, it)
    return px


def preview():
    import vram as Vr
    pac = open(TS.MK.PAC, 'rb').read()
    orig = TK.load_px(pac); px, done = make(pac, orig.copy())
    v = Vr.vram(os.path.join(ROOT, 'work', 'shots', '16.sav'))

    def show(P_, u, vv, w, h, clut):
        cx, cy = (clut & 0x3F) * 16, clut >> 6; cc = v[cy, cx:cx + 16].astype(int)
        pal = np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8)
        a = P_[vv:vv + h, u:u + w]; im = pal[a]; im[cc[a] == 0] = (200, 190, 160)
        return im
    W = 1400; cv = Image.new('RGB', (W, 2400), (25, 25, 25)); d = ImageDraw.Draw(cv); X = Y = RH = 0
    for u, vv, L, w, h in done:
        clut = 0x3228 if L == 'P' else 0x32E8
        pair = np.concatenate([show(orig, u, vv, w, h, clut), np.full((h, 2, 3), 255, np.uint8), show(px, u, vv, w, h, clut)], 1)
        I = Image.fromarray(pair).resize((pair.shape[1] * 2, h * 2), Image.NEAREST)
        if X + I.width > W:
            X = 0; Y += RH + 14; RH = 0
        cv.paste(I, (X, Y + 11)); d.text((X, Y), '%d,%d%s' % (u, vv, L), fill=(255, 255, 0)); X += I.width + 10; RH = max(RH, I.height + 11)
    px2 = make_c(pac, px.copy())
    pal16 = {(0xC, 0, 64): 0x34A8, (0xC, 36, 64): 0x3468, (0xC, 120, 64): 0x3468, (0xC, 0, 88): 0x3468, (0xC, 52, 88): 0x3468, (0xC, 152, 88): 0x3468,
             (0xC, 240, 112): 0x35E8, (0xC, 240, 152): 0x35E8, (0xC, 168, 232): 0x35E8, (0xC, 208, 232): 0x35E8, (0xC, 120, 232): 0x3528, (0xC, 144, 232): 0x3528, (0xA, 0, 152): 0x37A8}
    for it in C_ITEMS:
        tp, u, vv, w, h = it[:5]; x0 = (tp - 0xA) * 256; c = pal16[(tp, u, vv)]
        pair = np.concatenate([show(orig, x0 + u, vv, w, h, c), np.full((h, 2, 3), 255, np.uint8), show(px2, x0 + u, vv, w, h, c)], 1)
        I = Image.fromarray(pair).resize((pair.shape[1] * 3, h * 3), Image.NEAREST)
        if X + I.width > W:
            X = 0; Y += RH + 14; RH = 0
        cv.paste(I, (X, Y + 11)); d.text((X, Y), it[6], fill=(255, 255, 0)); X += I.width + 10; RH = max(RH, I.height + 11)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_unten.png')
    cv.crop((0, 0, W, Y + RH + 5)).save(p); print('미리보기', p, len(done))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()


def apply_all(pac):
    """운전 승무 아틀라스(7E54800)와 34번 파일 레코드: 역 이름(timetable_kr) → 노선 선택(route_kr) → 작은 낱말·배지(이 파일) 차례로 한 화소판에"""
    import route_kr as RK
    recs, px, _, out_tt, _, _ = TK.make(bytes(pac))
    px, out_rt, _ = RK.make(bytes(pac), px)
    px, _ = make(bytes(pac), px)
    px = make_c(bytes(pac), px)
    a = TS.FILE34
    F = TK.patch34(bytes(pac[a:a + 0xE000]), out_tt)
    F = RK.patch34(F, out_rt)
    pac[a:a + 0xE000] = F
    enc = TK.encode(px); pac[TK.DATA:TK.DATA + len(enc)] = enc
    return len(out_tt), len(out_rt)
