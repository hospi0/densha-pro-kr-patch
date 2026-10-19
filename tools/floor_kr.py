# -*- coding: utf-8 -*-
r"""철도 자료관 화면 그림 글자 한글화 (2026-10-18)
  그림 = 메뉴 TXS 989A000 사각형 1 (VRAM 640,0 128×256 = tpage A·B), 데이터 PAC 0x98BA800. 조각 표 = 22번 파일 0x19D1A~0x1A6EA(16 B, menu_kr 와 같은 꼴).
  ★한 칸에 두 그림을 비트 층으로 겹침: CLUT 3EAE·3C6E = 위 2비트(i>>2), 3EEE·3CAE = 아래 2비트(i&3). 3EEF = 4비트 통째(음악·보이스·효과음, 세로 계조 5~13).
  - 플로어 이름 칸(tpB 80,80 80×64 / 160,80 96·80×64): 위 = 写真·モデルフロア, 아래 = 映像·音楽フロア. 0~43줄 일본어만 다시 그림(44줄~ 영어 부제 그대로). 머리 레코드(h44)도 같은 칸.
  - 지역 칸(tpA 96,0/48/96 104×48): 위 = 関西·関東·九州, 아래 = 東北・北陸·スライドショー·その他(안 쓰임).
    東北·北陸 은 글자 하나씩 두 조각(東 = 関東의 東, 北·陸 = 東北・北陸 그림) → 한국어는 104 칸 통째: 東北 = その他 자리, 北陸 = 東北・北陸 자리, 두 번째 조각은 첫 조각과 같게.
  python tools/floor_kr.py → work/gfx/prev_floor.png (원본|한글)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import btn_kr as BT  # noqa: E402
import menu_kr as MK  # noqa: E402

DATA = 0x98BA800                     # 사각형 1 데이터(128×256 halfword → 4bpp 512×256, tpA = u 0~255, tpB = 256~511)
RUNS = ((0x19D1A, 1), (0x19D3A, 127), (0x1A58A, 23))
HI, LO, FULL = 2, 0, None
LAYER = {0x3EAE: HI, 0x3C6E: HI, 0x3EEE: LO, 0x3CAE: LO}

# (tpage, u, v, w, 층, 다시 그릴 줄 수) → 한국어
FLOOR = {(0xB, 80, 80, 80, HI): '사진 플로어', (0xB, 80, 80, 80, LO): '영상 플로어',
         (0xB, 160, 80, 96, HI): '모델 플로어', (0xB, 160, 80, 80, LO): '음악 플로어'}
FLOOR_ROWS = 44
REGION = {(0xA, 96, 0, HI): '칸사이', (0xA, 96, 48, HI): '칸토', (0xA, 96, 96, HI): '큐슈',
          (0xA, 96, 48, LO): '슬라이드쇼', (0xA, 96, 0, LO): '호쿠리쿠', (0xA, 96, 96, LO): '토호쿠'}
# 東北(1A58A·1A59A) → その他 자리 아래층, 北陸(1A5AA·1A5BA) → 東北・北陸 자리 아래층. 두 번째 조각은 첫 조각과 같게(x 포함)
REMAP = {0x1A58A: (96, 96), 0x1A59A: (96, 96), 0x1A5AA: (96, 0), 0x1A5BA: (96, 0)}
SOUND = {(0xA, 32, 0): '음악', (0xA, 32, 48): '보이스', (0xA, 32, 96): '효과음'}   # 56×48, 3EEF


def records(ov):
    out = []
    for base, n in RUNS:
        for k in range(n):
            o = base + 16 * k; w = struct.unpack_from('<8H', ov, o)
            out.append(dict(o=o, u=w[0] & 0xFF, v=w[0] >> 8, x=MK.s16(w[1]), y=MK.s16(w[2]), w=w[3], h=w[4], tp=w[5], clut=w[6]))
    return out


def load_px(pac):
    raw = np.frombuffer(pac[DATA:DATA + 0x10000], np.uint8).reshape(256, 256)
    return np.stack([raw & 15, raw >> 4], -1).reshape(256, 512).copy()


def encode(px):
    p = px.reshape(256, 256, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def col0(tp):
    return 0 if tp == 0xA else 256


def fit(text, h, w, xmin=0.5):
    """높이 h 로 한 번에 그려 폭 w 안에(가로 비율 xmin 까지 좁힘) → 덮임 배열"""
    xs = 1.0
    while True:
        c, _ = BT.cover(text, h, xs)
        if c.shape[1] <= w:
            return c
        if xs <= xmin:                                     # 가장 좁혀도 넘치면 글자 높이를 줄임
            h -= 1; xs = xmin; continue
        xs = max(xmin, min(xs - 0.02, xs * w / c.shape[1]))


def uniform(texts, h, w, xmin):
    """여러 낱말을 «같은 높이·같은 가로 비율»로(가장 긴 낱말에 맞춤) → {낱말: 덮임}"""
    while True:
        nat = {t: BT.cover(t, h, 1.0)[0].shape[1] for t in texts}
        xs = min(1.0, min(w[t] / nat[t] for t in texts))
        if xs >= xmin:
            return {t: BT.cover(t, h, xs)[0] for t in texts}
        h -= 1


def put_layer(px, x0, y0, w, h, sh, cov, nlev, left=None):
    """칸(x0,y0,w,h)의 sh 층을 비우고 cov 를 가운데(left 주면 왼쪽 left px)에 그림(nlev = 2 → 1·2, 3 → 1·2·3)"""
    reg = px[y0:y0 + h, x0:x0 + w]
    reg &= ~np.uint8(3 << sh)
    if nlev == 2:
        lv = np.where(cov >= 0.55, 2, np.where(cov >= 0.2, 1, 0))
    else:
        lv = np.digitize(cov, [0.2, 0.45, 0.75])
    hh, ww = lv.shape
    ox = (w - ww) // 2 if left is None else left; oy = (h - hh) // 2
    reg[oy:oy + hh, ox:ox + ww] |= (lv << sh).astype(np.uint8)


def make(pac):
    """→ recs, 새 화소, 원본 화소, {레코드: (u, v, x)}"""
    ov = pac[MK.OV:MK.OV + 0x2F800]
    recs = records(ov); px = load_px(pac); orig = px.copy()
    # 플로어 이름: 0~43줄(일본어)만
    fc = uniform(list(FLOOR.values()), 30, {t: 76 for t in FLOOR.values()}, 0.5)      # 좁은 칸(80) 기준으로 넷 다 같게
    for (tp, u, v, w, sh), t in FLOOR.items():
        reg = px[v:v + FLOOR_ROWS, col0(tp) + u:col0(tp) + u + w]
        reg &= ~np.uint8(3 << sh)
        put_layer(px, col0(tp) + u, v + 2, w, FLOOR_ROWS - 4, sh, fc[t], 2, left=2)   # 왼쪽 맞춤 — 모델 칸만 96 이라 가운데면 선택(확대) 때 오른쪽으로 밀림(사용자 스샷)
    # 지역 이름(104×48) — 모두 같은 크기
    rc = uniform(list(REGION.values()), 36, {t: 100 for t in REGION.values()}, 0.6)
    for (tp, u, v, sh), t in REGION.items():
        put_layer(px, col0(tp) + u, v, 104, 48, sh, rc[t], 3)
    # 음악·보이스·효과음(4비트 통째, 줄마다 원본 계조값) — 셋 다 같은 크기
    sc = uniform(list(SOUND.values()), 34, {t: 54 for t in SOUND.values()}, 0.6)
    for (tp, u, v), t in SOUND.items():
        reg = px[v:v + 48, col0(tp) + u:col0(tp) + u + 56]
        o = reg.copy(); rows = [np.bincount(r[r > 0]).argmax() if (r > 0).any() else 0 for r in o]
        ys = [y for y in range(48) if rows[y]]
        reg[:] = 0
        c = sc[t]
        hh, ww = c.shape; oy = max(0, ys[0] + (ys[-1] - ys[0] + 1 - hh) // 2); ox = (56 - ww) // 2
        for y in range(hh):
            g = int(np.bincount(o[o > 0]).argmax())                       # 2026-10-19 사용자 «단색» — 줄별 계조 대신 최빈 색 하나
            reg[oy + y, ox:ox + ww][c[y] >= 0.4] = g
    moved = {}
    first = {}
    for r in recs:
        if r['o'] in REMAP:
            u, v = REMAP[r['o']]
            key = (u, v)
            if key not in first:
                first[key] = r['x'] if False else 0
            moved[r['o']] = (u, v, 0, 104, 0x3CAE)
    return recs, px, orig, moved


def patch_ov(ov, moved):
    ov = bytearray(ov)
    for o, (u, v, x, w, clut) in moved.items():
        ov[o] = u; ov[o + 1] = v
        struct.pack_into('<h', ov, o + 2, x)
        struct.pack_into('<H', ov, o + 6, w)
        struct.pack_into('<H', ov, o + 12, clut)
    return bytes(ov)


def apply(pac):
    recs, px, _, moved = make(bytes(pac))
    pac[MK.OV:MK.OV + 0x2F800] = patch_ov(bytes(pac[MK.OV:MK.OV + 0x2F800]), moved)
    enc = encode(px); pac[DATA:DATA + len(enc)] = enc
    return len(moved)


def preview():
    import vram as V
    pac = open(MK.PAC, 'rb').read()
    recs, px, orig, moved = make(pac)
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '8.sav'))

    def pal(c):
        cx, cy = (c & 0x3F) * 16, c >> 6; cc = v[cy, cx:cx + 16].astype(int)
        return np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8), cc == 0

    def crop(P, r, mv):
        u, vv, x, w, c = mv.get(r['o'], (r['u'], r['v'], r['x'], r['w'], r['clut']))
        a = P[vv:vv + r['h'], col0(r['tp']) + u:col0(r['tp']) + u + w]
        p, tr = pal(c); im = p[a]; im[tr[a]] = (40, 30, 55)
        return im, x

    def group(P, rs, mv):
        ims = [crop(P, r, mv) for r in rs]
        W = max(x + im.shape[1] for im, x in ims); H = max(r['y'] + im.shape[0] for (im, _), r in zip(ims, rs))
        cv = np.zeros((H, W, 3), np.uint8); cv[:] = (40, 30, 55)
        for (im, x), r in zip(ims, rs):
            m = im.sum(-1) != 125
            cv[r['y']:r['y'] + im.shape[0], x:x + im.shape[1]][m] = im[m]
        return cv

    byo = {r['o']: r for r in recs}
    sets = [[0x19D4A], [0x19D6A], [0x19D8A], [0x19DAA], [0x1A4BA], [0x1A4DA], [0x1A4FA], [0x1A51A],
            [0x19DBA], [0x19DCA], [0x19DDA], [0x1A58A, 0x1A59A], [0x1A5AA, 0x1A5BA], [0x1A5CA], [0x1A5DA], [0x1A5EA], [0x1A5FA]]
    W = 1400; cv = Image.new('RGB', (W, 1400), (25, 25, 25)); d = ImageDraw.Draw(cv); X = Y = RH = 0
    for s in sets:
        rs = [byo[o] for o in s]
        a = group(orig, rs, {}); b = group(px, rs, moved)
        H = max(a.shape[0], b.shape[0])
        pad = lambda m: np.pad(m, ((0, H - m.shape[0]), (0, 0), (0, 0)))
        pair = np.concatenate([pad(a), np.full((H, 3, 3), 200, np.uint8), pad(b)], 1)
        I = Image.fromarray(pair).resize((pair.shape[1] * 2, pair.shape[0] * 2), Image.NEAREST)
        if X + I.width > W:
            X = 0; Y += RH + 16; RH = 0
        cv.paste(I, (X, Y + 12)); d.text((X, Y), '%X' % s[0], fill=(255, 255, 0))
        X += I.width + 12; RH = max(RH, I.height + 12)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_floor.png')
    cv.crop((0, 0, W, Y + RH + 10)).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
