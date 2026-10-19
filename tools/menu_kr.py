# -*- coding: utf-8 -*-
r"""메뉴 아틀라스(TXS 989A000 = 섹터 지도 896번, VRAM 832~959 × 0~511) 조각 글자 한글화 (2026-10-18)
  조각 표 = 22번 파일(오버레이) 16 B 레코드 [v<<8|u][x][y][w][h][tpage][clut][flag] — work/menu_recs.txt(5 구간 117개).
  ★아틀라스 한 칸 = 2비트 두 층: CLUT 7F78/7F70 = 위 2비트, 7FB8/7FB0 = 아래 2비트(0 투명·1 진회·2 연회·3 흰). 다시 그릴 땐 그 층 비트만 바꾼다.
  ★여러 조각을 잇는 항목(같은 y, x 가 0 에서 늘어남)은 한국어 폭으로 레코드 w·x 를 다시 계산(조각 사이 2px).
  python tools/menu_kr.py preview → work/gfx/prev_menu.png (원본|한글, 조각별 + 항목 조립)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
OV = 0x4B800                       # 22번 파일 PAC 위치
TXS = 0x989A000                    # 메뉴 아틀라스 TXS
RECT0 = (832, 0, 128, 512)         # 데이터 = TXS + 0x800
RUNS = ((0x1939A, 1), (0x193CA, 1), (0x193DA, 6), (0x194BA, 69), (0x19BCA, 13), (0x1BEC6, 14), (0x1C016, 15))
FONT = r'C:\claude\utils\font\nanum-gothic\NanumGothicExtraBold.ttf'
SS = 4
GAP = 2

# 조각(레코드 위치) → 한국어. 같은 그림 칸을 여럿이 쓰면 첫 레코드에만 적어도 됨(칸 단위로 그림).
KR = {
    0x1939A: '운전 승무', 0x193CA: '철도 자료관',   # 메인 메뉴 큰 제목(위 층) — 아래 영어 부제는 그대로
    0x193FA: '게임 설정', 0x1942A: '로드・세이브',
    0x194BA: '로드', 0x194CA: '세이브', 0x194DA: '오토', 0x194FA: '포켓스테이션',
    0x1950A: '다운', 0x1954A: '컨트롤', 0x1955A: '난이도', 0x1956A: '사운드',
    0x1958A: '거리', 0x195AA: '화면위치조정', 0x195BA: '진동',
    0x195CA: '투', 0x196FA: '타입', 0x1972A: '마메콘',
    0x1974A: '이지', 0x1975A: '노멀', 0x1976A: '하드', 0x1977A: '베리',
    0x1979A: '끔', 0x197AA: '약', 0x197BA: '중', 0x197CA: '강',
    0x197DA: '스테레오', 0x197EA: '모노', 0x1981A: '디지털',
    0x1987A: '메모리카드', 0x1988A: '슬롯', 0x198DA: '취소',
}
# 같은 층에서 그림을 나눠 쓰는 조각 묶음: (앞 조각 레코드, 앞 글자, 뒤 조각 레코드, 뒤 글자)
#   앞 글자는 «앞 조각 칸 − 뒤 조각 칸» 영역 오른쪽 끝에, 뒤 글자는 뒤 조각 칸 왼쪽에 그려 이어지게 하고,
#   앞 조각(긴 낱말) 레코드는 u 를 당기고 w 를 두 글자 폭 합으로.
SHARED = [(0x1957A, '속도', 0x1959A, '미터', 'concat'),     # 速度メーター ⊃ メーター → «속도미터» 이어 그리고 メーター 레코드 u = «미터» 자리
          (0x1968A, '원', 0x195DA, '핸들', 'concat'),        # ワンハンドル ⊃ ハンドル → «원핸들», ハンドル 레코드 u = «핸들» 자리
          (0x197FA, '비', 0x1984A, '표시', 'split')]         # 非表示 ⊃ 表示 (칸 끝이 tpage 끝이라 늘릴 수 없음 → 원래 경계에서 나눔)


def s16(x):
    return x - 0x10000 if x & 0x8000 else x


def records(ov):
    out = []
    for base, n in RUNS:
        for k in range(n):
            o = base + 16 * k; w = struct.unpack_from('<8H', ov, o)
            out.append(dict(o=o, u=w[0] & 0xFF, v=w[0] >> 8, x=s16(w[1]), y=s16(w[2]), w=w[3], h=w[4], tp=w[5], clut=w[6], fl=w[7]))
    return out


LAYER = {0x7F78: 2, 0x7F70: 2, 0x7EF9: 2, 0x7FB8: 0, 0x7FB0: 0, 0x7F39: 0}   # CLUT 팔레트 모양(스테이트 3 VRAM)으로 판정: 색이 i>>2 에만 → 위, i&3 에만 → 아래


def layer_of(clut):
    """CLUT → 쓰는 비트 자리(위 2비트면 2, 아래면 0)"""
    return LAYER.get(clut)


class Atlas:
    """TXS 사각형 0 = VRAM (832,0) 128×512 halfword → 4bpp 512×512"""
    def __init__(self, pac):
        x, y, w, h = RECT0
        raw = np.frombuffer(pac[TXS + 0x800:TXS + 0x800 + w * h * 2], np.uint8).reshape(h, w * 2)
        self.px = np.stack([raw & 15, raw >> 4], -1).reshape(h, w * 4).copy()

    def page(self, tp):
        tx, ty = (tp & 15) * 64, ((tp >> 4) & 1) * 256
        return (tx - RECT0[0]) * 4, ty            # 아틀라스 안 화소 원점

    def region(self, r):
        ox, oy = self.page(r['tp'])
        return self.px[oy + r['v']:oy + r['v'] + r['h'], ox + r['u']:ox + r['u'] + r['w']]

    def encode(self):
        p = self.px.reshape(self.px.shape[0], -1, 2)
        return ((p[..., 0] | (p[..., 1] << 4)).astype(np.uint8)).tobytes()


def render(text, h_ink, maxw, maxh, xsc=1.0, split=None):
    """text 를 한 번에 그림(낱말을 나눠 그리지 않음 — 크기 어긋남 방지). 세로 칸 = 글꼴의 «한» 높이 기준(모든 낱말 같은 기준선),
    가로 = 잉크 범위. split 이 주어지면 text[:split] 이 끝나는 열도 돌려줌. → (단계 배열, 나눔 열)"""
    f = ImageFont.truetype(FONT, int(h_ink * SS * 1.25))
    hl, ht, hr, hb = f.getbbox('한')
    l, t, r, b = f.getbbox(text)
    pad = (hb - ht) // 10
    im = Image.new('L', (r - l + 8, max(b, hb) - min(t, ht) + 2 * pad + 8))
    oy = 4 + pad - min(t, ht)
    ImageDraw.Draw(im).text((4 - l, oy), text, font=f, fill=255)
    a = np.array(im)[oy + ht - pad:oy + hb + pad]
    cols = np.nonzero(a.max(0))[0]; x0 = cols.min()
    a = a[:, x0:cols.max() + 1]
    sc = h_ink * SS / (hb - ht)
    W = max(1, int(round(a.shape[1] * sc * xsc / SS))); H = max(1, int(round(a.shape[0] * sc / SS)))
    cut = None
    if split:
        cut = int(round((f.getlength(text[:split]) - (x0 - (4 - l)) - l) * sc * xsc / SS))
    W = min(W, maxw); H = min(H, maxh)
    cov = np.array(Image.fromarray(a).resize((W, H), Image.BOX)).astype(float) / 255
    return np.digitize(cov, [0.2, 0.45, 0.75]), cut


def draw(at, r, sh, text, u0, u1, align, h_ink=None, top=None):
    """r 의 tpage·v·h 안 [u0,u1) 칸에 text 를 그 층으로 그림 → (실제 글자 왼쪽 u, 폭)"""
    ox, oy = at.page(r['tp'])
    reg = at.px[oy + r['v']:oy + r['v'] + r['h'], ox + u0:ox + u1]
    lay = (reg >> sh) & 3
    if r['h'] >= 96:                                   # 큰 제목(216×112): 영어 부제 = 80줄 이후 «빈 줄 다음 6줄 이상 연속 잉크» 시작 → 그 앞 줄까지만 다시 그림
        rows = lay.any(1)
        cut = next(k for k in range(80, r['h'] - 6) if not rows[k - 1] and rows[k:k + 6].all())
        reg = reg[:cut]; lay = lay[:cut]
    if h_ink is None:
        ys = np.nonzero(lay.any(1))[0]
        h_ink = (ys.max() - ys.min() + 1) if len(ys) else r['h'] - 8
        top = ys.min() if len(ys) else 4
    lv, _ = render(text, int(h_ink * 0.92), u1 - u0, reg.shape[0] - top)
    reg &= ~np.uint8(3 << sh)
    hh, ww = lv.shape
    x0 = (u1 - u0 - ww) if align == 'r' else 0
    y0 = top + max(0, (h_ink - hh) // 2)
    reg[y0:y0 + hh, x0:x0 + ww] |= (lv << sh).astype(np.uint8)
    return u0 + x0, ww, h_ink, top


XS = 0.85          # 32px 조각 글자 가로 비율(공간 맞춤)
H32 = 23           # 32px 조각 글자 높이(옮기는 조각 칸 합 안에 다 들어가는 크기)


def make(pac):
    """32px 조각: 우리가 옮기는 조각 칸을 (tpage, 층) 단위로 비우고 한국어 낱말을 그 빈 칸들에 다시 배치(줄 넘나듦, 레코드 u·v 갱신).
    큰 제목(h≥96)은 제자리. → recs, 아틀라스, 원본 화소, {레코드: (u, v, w)}"""
    ov = pac[OV:OV + 0x2F800]
    recs = records(ov); at = Atlas(pac); orig = at.px.copy()
    byo = {r['o']: r for r in recs}
    reg = lambda r: (r['tp'], layer_of(r['clut']), r['u'], r['v'], r['w'], r['h'])
    out = {}
    # 큰 제목
    for r in recs:
        if r['o'] in KR and r['h'] >= 96:
            u, w, _, _ = draw(at, r, layer_of(r['clut']), KR[r['o']], r['u'], r['u'] + r['w'], 'l')
            out[r['o']] = (r['u'], r['v'], r['w'])
    # 낱말 목록: 칸(reg) → (글자, [(뒤 칸, 뒤 글자)])
    words = {}; sub = {}
    for a, ta, b, tb, mode in SHARED:
        words[reg(byo[a])] = (ta, tb); sub[reg(byo[b])] = reg(byo[a])
    for r in recs:
        if r['o'] in KR and r['h'] < 96 and reg(r) not in words and reg(r) not in sub:
            words[reg(r)] = (KR[r['o']], None)
    # 빈 칸 = 옮기는 칸들의 합(층별), 비우기
    free = {}
    for k in list(words) + list(sub):
        tp, sh, u, v, w, h = k
        ox, oy = at.page(tp)
        at.px[oy + v:oy + v + h, ox + u:ox + u + w] &= ~np.uint8(3 << sh)
        m = free.setdefault((tp, sh, h), np.zeros((256, 256), bool)); m[v:v + h, u:u + w] = True
    # 낱말 그림 만들기 — 이어 붙는 낱말(SHARED)은 «앞+뒤»를 한 번에 그리고 뒤 낱말 시작 열만 계산
    lv = {}
    for k, (ta, tb) in words.items():
        img, cut = render(ta + (tb or ''), H32, 256, k[5], XS, len(ta) if tb else None)
        lv[k] = (img, cut)
    # 배치(넓은 것부터, 같은 (tpage,층,높이)의 빈 칸에서 처음 맞는 곳)
    order = sorted(words, key=lambda k: -lv[k][0].shape[1])
    for k in order:
        tp, sh, u0, v0, w0, h = k
        img, cut = lv[k]; W = img.shape[1]
        m = free[(tp, sh, h)]; spot = None
        for v in sorted({vv for vv in range(256) if m[vv].any()}):
            if v + h > 256:
                continue
            col = m[v:v + h].all(0); run = 0
            for u in range(256):
                run = run + 1 if col[u] else 0
                if run >= W + 1:
                    spot = (u - run + 1, v); break
            if spot:
                break
        assert spot, '자리 없음 %s %s' % (k, words[k])
        u, v = spot; m[v:v + h, u:u + W + 1] = False
        ox, oy = at.page(tp)
        hh, ww = img.shape; top = (h - hh) // 2
        at.px[oy + v + top:oy + v + top + hh, ox + u:ox + u + ww] |= (img << sh).astype(np.uint8)
        for r in recs:
            if reg(r) == k:
                out[r['o']] = (u, v, W)
            elif sub.get(reg(r)) == k:
                out[r['o']] = (u + cut, v, W - cut)
    return recs, at, orig, out


SPACE = {0x1988A: 6, 0x198BA: 6, 0x196FA: 5, 0x1971A: 5}   # 낱말 사이 띄움(px): 메모리카드|슬롯 · 원핸들|타입


def relayout(recs, neww):
    """항목(같은 y, x 가 0 → 증가하는 연속 레코드) 안 조각 x = 원래 x − 앞 조각들이 줄어든 폭 합 (+ SPACE)"""
    out = {}
    for g in groups(recs):
        shrink = 0
        for k, r in enumerate(g):
            w = neww[r['o']][2] if r['o'] in neww else r['w']
            x = r['x'] - shrink + SPACE.get(r['o'], 0)
            out[r['o']] = (x, w)
            shrink += (r['w'] - w) - SPACE.get(r['o'], 0)
    return out


def patch_ov(ov, recs, lay, neww):
    """레코드 u·x·w 고쳐 쓰기 (v·y·h·tpage·clut 그대로)"""
    ov = bytearray(ov)
    for r in recs:
        if r['o'] in neww:
            ov[r['o']] = neww[r['o']][0]; ov[r['o'] + 1] = neww[r['o']][1]
        if r['o'] in lay:
            x, w = lay[r['o']]
            struct.pack_into('<h', ov, r['o'] + 2, x)
            struct.pack_into('<H', ov, r['o'] + 6, w)
    return bytes(ov)


def groups(recs):
    """relayout 과 같은 항목 묶음"""
    out = []; i = 0
    while i < len(recs):
        j = i + 1
        while j < len(recs) and recs[j]['x'] > recs[j - 1]['x'] and recs[j]['y'] == recs[i]['y'] and recs[j]['o'] - recs[j - 1]['o'] == 16:
            j += 1
        out.append(recs[i:j]); i = j
    return out


def preview():
    sys.path.insert(0, HERE)
    import vram as V
    pac = open(PAC, 'rb').read()
    recs, at, orig, neww = make(pac)
    lay = relayout(recs, neww)
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '3.sav'))

    def pal(c):
        cx, cy = (c & 0x3F) * 16, c >> 6; cc = v[cy, cx:cx + 16].astype(int)
        return np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8)

    def crop(px, r, w=None):
        ox, oy = at.page(r['tp'])
        a = px[oy + r['v']:oy + r['v'] + r['h'], ox + r['u']:ox + r['u'] + (w or r['w'])]
        im = pal(r['clut'])[a]; im[(a == 0) | (pal(r['clut'])[a].sum(-1) == 0)] = [40, 20, 50]
        return im

    def compose(px, g, geo):
        """항목 g 를 화면처럼 이어 그림 (geo: 레코드 → (u, x, w))"""
        W_ = max(geo[r['o']][2] + geo[r['o']][3] for r in g) - min(geo[r['o']][2] for r in g)
        H_ = max(r['h'] for r in g); x0 = min(geo[r['o']][2] for r in g)
        out = np.full((H_, max(W_, 1), 3), (40, 20, 50), np.uint8)
        for r in g:
            u, vv, x, w = geo[r['o']]
            rr = dict(r); rr['u'] = u; rr['v'] = vv
            im = crop(px, rr, w)
            m = im.sum(-1) != (40 + 20 + 50)
            out[:im.shape[0], x - x0:x - x0 + w][m] = im[m]
        return out

    old = {r['o']: (r['u'], r['v'], r['x'], r['w']) for r in recs}
    new = {r['o']: (neww[r['o']][0] if r['o'] in neww else r['u'], neww[r['o']][1] if r['o'] in neww else r['v'], lay[r['o']][0], lay[r['o']][1]) for r in recs}
    W = 1500; cv = Image.new('RGB', (W, 3000), (25, 25, 25)); d = ImageDraw.Draw(cv); X = Y = RH = 0
    for g in groups(recs):
        if not any(r['o'] in neww for r in g):
            continue
        a = compose(orig, g, old); b = compose(at.px, g, new)
        hh = max(a.shape[0], b.shape[0])
        pad = lambda m: np.concatenate([m, np.full((hh - m.shape[0], m.shape[1], 3), 25, np.uint8)], 0)
        pair = np.concatenate([pad(a), np.full((hh, 3, 3), 200, np.uint8), pad(b)], 1)
        I = Image.fromarray(pair).resize((pair.shape[1] * 2, pair.shape[0] * 2), Image.NEAREST)
        if X + I.width > W:
            X = 0; Y += RH + 16; RH = 0
        cv.paste(I, (X, Y + 14)); d.text((X, Y), '%X' % g[0]['o'], fill=(255, 255, 0))
        X += I.width + 12; RH = max(RH, I.height)
    p = os.path.join(ROOT, 'work', 'gfx'); os.makedirs(p, exist_ok=True)
    cv.crop((0, 0, W, Y + RH + 20)).save(os.path.join(p, 'prev_menu.png'))
    print('미리보기', os.path.join(p, 'prev_menu.png'))


if __name__ == '__main__':
    preview()
