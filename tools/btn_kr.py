# -*- coding: utf-8 -*-
r"""버튼 안내 아틀라스(TXS 98D3000, 사각형 (960,0,64,256) = tpage 0x0F) 글자 한글화 (2026-10-18)
  조각 표 = 22번 파일 16 B 레코드(tpage 0x0F) 0x1864A~0x18CDA. 글자 CLUT 3F7C/3FFC = 1~8 회색 계조(8 흰), 選択 상자 = 3FFC 의 d·e·f(흰 쪽).
  글자 칸은 «낱말:» 이 버튼 아이콘(x116) 앞에 오른쪽 맞춤 → 칸 폭 그대로 두고 오른쪽 맞춤으로 다시 그림(레코드 x 그대로).
  ★次のムービー: 칸의 왼쪽 30px 를 «次の音楽:»(次の+音楽+:) 조합이 같이 씀 → 한국어는 조합용 «다음» 을 빈 칸(VNEW)에 따로 두고 그 레코드 u·v 만 옮김.
  python tools/btn_kr.py → work/gfx/prev_btn.png (원본|한글, 화면 조합)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import menu_kr as MK

TXS = 0x98D3000
TP = 0x0F
LO, HI = 0x1864A, 0x18CEA                  # 레코드 범위(16 B 정렬 = …A)
FONT = MK.FONT
SS = 4
H_INK = 13                                 # 원본 글자 잉크 높이 14(탁점 포함)

# 칸 (u, v, w) → 한국어(높이 16, 오른쪽 맞춤). ページ送り:/キャンセル: 두 줄 칸(148,144,76,32)은 아래 반이 キャンセル: 칸과 같음
KR = {
    (148, 144, 76): '페이지 넘김:', (148, 160, 76): '취소:', (186, 176, 38): '결정:',
    (176, 192, 76): '다음 영상:', (176, 208, 76): '재생/정지:', (40, 208, 92): '방향키 선택',
}
COMBO = ((176, 192, 30), (144, 208, 32), '다음', '음악')   # 次の+音楽 (+ : 칸 그대로) → «다음 음악» 한 번에 그려 나눔
VNEW = (0, 100)                            # 조합용 «다음» 새 칸 자리(빈 영역 v64~143)
BOX = (80, 192, 34, 16)                    # 選択 상자: 틀(u·u+33 세로, v·v+14 가로, 모서리 v+1·v+13) 안쪽 (u+2~u+32, v+2~v+12)만 다시 그림
BOXKR = '선택'


def load_px(pac):
    raw = np.frombuffer(pac[TXS + 0x800:TXS + 0x800 + 0x8000], np.uint8).reshape(256, 128)
    return np.stack([raw & 15, raw >> 4], -1).reshape(256, 256).copy()


def encode(px):
    p = px.reshape(256, 128, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def records(ov):
    out = []
    for o in range(LO, HI, 16):
        w = struct.unpack_from('<8H', ov, o)
        if w[5] == TP:
            out.append(dict(o=o, u=w[0] & 0xFF, v=w[0] >> 8, x=MK.s16(w[1]), y=MK.s16(w[2]), w=w[3], h=w[4], clut=w[6]))
    return out


def cover(text, h_ink, xsc=1.0, split=None, font=None):
    """나눔고딕 ExtraBold(font 로 바꿀 수 있음) 로 한 번에 그림 → 덮임 비율(0~1) 배열(높이 = '한' 잉크 높이 h_ink 기준), split 위치 열"""
    f = ImageFont.truetype(font or FONT, int(h_ink * SS * 1.25))
    hl, ht, hr, hb = f.getbbox('한')
    l, t, r, b = f.getbbox(text)
    top = min(t, ht); bot = max(b, hb)
    im = Image.new('L', (r - l + 8, bot - top + 8))
    ImageDraw.Draw(im).text((4 - l, 4 - top), text, font=f, fill=255)
    a = np.array(im)
    a = a[4 + ht - top:4 + hb - top]
    cols = np.nonzero(a.max(0))[0]; x0 = cols.min(); a = a[:, x0:cols.max() + 1]
    sc = h_ink * SS / (hb - ht)
    W = max(1, int(round(a.shape[1] * sc * xsc / SS)))
    cut = None
    if split:
        cut = int(round((f.getlength(text[:split]) - (x0 - (4 - l)) - l) * sc * xsc / SS))
    return np.array(Image.fromarray(a).resize((W, h_ink), Image.BOX)).astype(float) / 255, cut


def levels(cov, lo=1, hi=8):
    """덮임 → 계조 값(0 투명, lo~hi)"""
    v = np.rint(cov * (hi - lo + 1) + lo - 1).astype(int)
    v = np.clip(v, lo, hi); v[cov < 0.12] = 0
    return v.astype(np.uint8)


def fit(text, w, split=None):
    """폭 w 안에 들어가게(가로 85% 까지 좁힘)"""
    for xsc in (1.0, 0.95, 0.9, 0.85):
        c, cut = cover(text, H_INK, xsc, split)
        if c.shape[1] <= w:
            return c, cut
    raise ValueError('%s 폭 %d > %d' % (text, c.shape[1], w))


def put(px, u, v, w, cov, align='r', pad=0):
    reg = px[v:v + 16, u:u + w]
    assert reg.max() <= 8, (u, v)
    reg[:] = 0
    hh, ww = cov.shape
    x0 = w - ww - pad if align == 'r' else pad
    y0 = 1 + (14 - hh) // 2 + 1
    reg[y0:y0 + hh, x0:x0 + ww] = levels(cov)


def make(pac):
    """→ recs, 새 화소, 원본 화소, {레코드: (u, v)}"""
    ov = pac[MK.OV:MK.OV + 0x2F800]
    recs = records(ov); px = load_px(pac); orig = px.copy()
    assert not px[VNEW[1]:VNEW[1] + 16, VNEW[0]:VNEW[0] + COMBO[0][2]].any(), '새 칸 자리 비어 있지 않음'
    for (u, v, w), t in KR.items():
        put(px, u, v, w, fit(t, w)[0])
    # 다음+음악: 한 번에 그려 음악 칸 오른쪽 끝(콜론 칸 앞)에 맞추고 «다음» 몫은 새 칸 오른쪽에
    (ua, va, wa), (ub, vb, wb), ta, tb = COMBO
    c, cut = fit(ta + ' ' + tb, wa + wb, len(ta) + 1)
    W = c.shape[1]; s = W - wb                              # 새 칸에 들어갈 왼쪽 몫 폭(음악 칸이 오른쪽 wb 차지)
    assert 0 <= s <= wa, ('다음 음악', W)
    cb = c[:, s:]; ca = c[:, :s]
    put(px, ub, vb, wb, cb, 'r')
    if s:
        put(px, VNEW[0], VNEW[1], wa, ca, 'r')
    moved = {r['o']: VNEW for r in recs if (r['u'], r['v'], r['w']) == (ua, va, wa)}
    # 選択 상자 안쪽
    u, v, w, h = BOX
    inner = px[v + 2:v + 13, u + 2:u + 33]
    assert set(np.unique(inner)) <= {0, 13, 14, 15}
    inner[:] = 0
    cv, _ = cover(BOXKR, 11)
    lv = levels(cv, 13, 15)                                 # d·e·f
    lv = np.where(lv == 0, 0, 28 - lv)                      # 원본은 f 가 가장 밝은 획 → 진할수록 f
    lv[lv == 28] = 0
    hh, ww = lv.shape; x0 = (inner.shape[1] - ww) // 2; y0 = (inner.shape[0] - hh) // 2
    inner[y0:y0 + hh, x0:x0 + ww] = lv
    return recs, px, orig, moved


def patch_ov(ov, moved):
    ov = bytearray(ov)
    for o, (u, v) in moved.items():
        ov[o] = u; ov[o + 1] = v
    return bytes(ov)


def apply(pac):
    recs, px, _, moved = make(bytes(pac))
    pac[MK.OV:MK.OV + 0x2F800] = patch_ov(bytes(pac[MK.OV:MK.OV + 0x2F800]), moved)
    enc = encode(px); a = TXS + 0x800
    pac[a:a + len(enc)] = enc
    return len(moved)


def groups(recs):
    out = []; cur = []
    for r in recs:
        big = r['h'] >= 40 and r['x'] <= 1                       # 화면마다 맨 앞 = 왼쪽 큰 조작기 그림
        if cur and (r['o'] - cur[-1]['o'] != 16 or big or any(q['x'] < r['x'] + r['w'] and r['x'] < q['x'] + q['w'] and q['y'] < r['y'] + r['h'] and r['y'] < q['y'] + q['h'] for q in cur)):   # 앞 조각과 화면 자리가 겹치면 다른 화면
            out.append(cur); cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def preview():
    import vram as V
    pac = open(MK.PAC, 'rb').read()
    recs, px, orig, moved = make(pac)
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '3.sav'))

    def pal(c):
        cx, cy = (c & 0x3F) * 16, c >> 6; cc = v[cy, cx:cx + 16].astype(int)
        return np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8), cc == 0

    def screen(P, g, mv):
        x0 = min(r['x'] for r in g); y0 = min(r['y'] for r in g)
        W = max(r['x'] + r['w'] for r in g) - x0; H = max(r['y'] + r['h'] for r in g) - y0
        cv = np.zeros((H, W, 3), np.uint8); cv[:] = (60, 50, 70)
        for r in g:
            u, vv = mv.get(r['o'], (r['u'], r['v']))
            a = P[vv:vv + r['h'], u:u + r['w']]
            p, tr = pal(r['clut']); m = ~tr[a]
            sub = cv[r['y'] - y0:r['y'] - y0 + r['h'], r['x'] - x0:r['x'] - x0 + r['w']]
            sub[m] = p[a][m]
        return cv

    W = 1400; cvs = Image.new('RGB', (W, 2400), (25, 25, 25)); d = ImageDraw.Draw(cvs); X = Y = RH = 0
    done = set()
    for g in groups(recs):
        key = tuple((r['u'], r['v'], r['x'], r['y'], r['w']) for r in g)
        if key in done or not any(r['clut'] in (0x3F7C, 0x3FFC) and r['v'] >= 144 for r in g):
            continue
        done.add(key)
        a = screen(orig, g, {}); b = screen(px, g, moved)
        pair = np.concatenate([a, np.full((a.shape[0], 3, 3), 200, np.uint8), b], 1)
        I = Image.fromarray(pair).resize((pair.shape[1] * 2, pair.shape[0] * 2), Image.NEAREST)
        if X + I.width > W:
            X = 0; Y += RH + 18; RH = 0
        cvs.paste(I, (X, Y + 14)); d.text((X, Y), '%X' % g[0]['o'], fill=(255, 255, 0))
        X += I.width + 14; RH = max(RH, I.height + 14)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_btn.png')
    cvs.crop((0, 0, W, Y + RH + 10)).save(p)
    print('미리보기', p, '· 옮긴 레코드', ['%X' % o for o in moved])


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
