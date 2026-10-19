# -*- coding: utf-8 -*-
r"""운전 승무 시각표 역 이름 한글화 (2026-10-18)
  그림 = TXS 7E54800 (VRAM 640,0 256×256 → 4bpp 1024×256 = tpage A·B·C·D), 역 한자 칸 = tpage D 두 층(CLUT 38A8 위 2비트·38E8 아래 2비트, 값 1 = 획·2 = 옅은 가장자리).
  역 레코드 = 34번 파일 16 B(tt_stations.records/stations), 역 목록·번역 = work/text/tt_names.tsv(레코드 = 역 첫 레코드 위치).
  ★역마다 레코드 수가 정해져 있어(秋田 2개 → «아키타» 3음절) 음절 칸 조합은 안 됨 → 역 이름을 «한 조각»으로 그려
    빈 자리(tpage D 전부 + A·B·C 의 빈 24줄 띠, 층 둘)에 배치, 역 첫 레코드 = 그 조각(x0), 나머지 레코드 = 첫 레코드와 같게(겹쳐 그림).
  조각 폭 상한 = max(원래 역 폭, 80) — 넘으면 가로를 좁힘.
  python tools/timetable_kr.py → work/gfx/prev_timetable.png (원본|한글)
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
import tt_stations as TS  # noqa: E402

DATA = TS.TXS + 0x800
H_INK = 15
LAYERS = ((0x38A8, 2), (0x38E8, 0))
KEEP = [(0xD, 240, 216, 16, 24)]                 # 아래층 «/» 칸(다른 화면이 쓸 수 있어 보존)


def load_px(pac):
    raw = np.frombuffer(pac[DATA:DATA + 0x20000], np.uint8).reshape(256, 512)
    return np.stack([raw & 15, raw >> 4], -1).reshape(256, 1024).copy()


def encode(px):
    p = px.reshape(256, 512, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def names():
    out = {}
    for l in open(os.path.join(ROOT, 'work', 'text', 'tt_names.tsv'), encoding='utf-8').read().splitlines()[1:]:
        c = l.split('\t')
        out[int(c[0], 16)] = c[2] if len(c) > 2 else ''
    return out


def groups(recs, nm):
    """역 첫 레코드 위치(tt_names) 기준으로 tpage D 레코드를 역마다 묶음(다음 역 시작 전까지)"""
    starts = sorted(o for o, k in nm.items() if k)
    tpd = [r for r in recs if r['tp'] == 0xD and r['clut'] in dict(LAYERS)]
    out = {}
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else 1 << 30
        out[s] = [r for r in tpd if s <= r['o'] < e]
    return out


def used_mask(pac):
    """34번 파일 레코드가 쓰는 tpage A·B·C 자리 + CLUT 줄"""
    F = pac[TS.FILE34:TS.FILE34 + 0xE000]; m = np.zeros((256, 1024), bool)
    for o in range(0, len(F) - 16, 2):
        w = struct.unpack_from('<8H', F, o); tp = w[5]
        if 0xA <= tp <= 0xC and 0 < w[3] <= 256 and 0 < w[4] <= 128 and (w[6] >> 6) >= 200 and (w[0] & 255) + w[3] <= 256 and (w[0] >> 8) + w[4] <= 256:
            u, v = w[0] & 255, w[0] >> 8; x0 = (tp - 0xA) * 256
            m[v:v + w[4], x0 + u:x0 + u + w[3]] = True
    m[196:236, 0:64] = True                        # CLUT 들(640,203~227 …)
    return m


def strips(px, used):
    """빈 24줄 띠 목록 [(tpage, v, u0, u1)] — tpage D 는 통째(보존 칸 빼고), A·B·C 는 화소 0 이고 레코드가 안 쓰는 열 연속 16 이상"""
    out = []
    for v in range(0, 240, 24):
        out.append((0xD, v, 0, 256 if v != 216 else 240))
    free = (px == 0) & ~used
    for t in (0xA, 0xB, 0xC):
        x0 = (t - 0xA) * 256
        for v in range(0, 232, 24):
            col = free[v:v + 24, x0:x0 + 256].all(0)
            u = 0
            while u < 256:
                if col[u]:
                    e = u
                    while e < 256 and col[e]:
                        e += 1
                    if e - u >= 16:
                        out.append((t, v, u, e))
                    u = e
                else:
                    u += 1
    return out


def piece(text, wmax):
    """역 이름 한 조각 → 높이 24 칸 값(1 획·2 옅음)"""
    xs = 1.0
    while True:
        c, _ = BT.cover(text, H_INK, xs)
        if c.shape[1] <= wmax or xs <= 0.5:
            break
        xs = max(0.5, xs * wmax / c.shape[1] - 0.01)
    lv = np.where(c >= 0.5, 1, np.where(c >= 0.2, 2, 0)).astype(np.uint8)
    out = np.zeros((24, lv.shape[1]), np.uint8)
    y0 = 4 + (17 - lv.shape[0]) // 2
    out[y0:y0 + lv.shape[0]] = lv
    return out


def make(pac):
    """→ recs, 새 화소, 원본 화소, {레코드: (tp, u, v, x, w, clut)}"""
    recs = TS.records(pac); px = load_px(pac); orig = px.copy()
    nm = names(); G = groups(recs, nm)
    used = used_mask(pac)
    S = strips(px, used)
    # tpage D 비우기(보존 칸 아래층 빼고)
    keep = px[216:240, 768 + 240:1024] & 3
    px[:240, 768:] = 0
    px[216:240, 768 + 240:1024] |= keep
    imgs = {}
    for s, rs in G.items():
        span = max(r['x'] + r['w'] for r in rs) if rs else 64
        imgs[s] = piece(nm[s], max(80, span))          # 시각 칸(x≈98~)까지는 넘어가도 됨(원본 긴 이름 112~136)
    # 넓은 것부터 «남는 폭이 가장 작은» (층, 띠)에 넣음(최적 맞춤)
    cur = {(k, sh): u0 for k, (t, v, u0, u1) in enumerate(S) for _, sh in LAYERS}
    spot = {}
    for s in sorted(imgs, key=lambda s: -imgs[s].shape[1]):
        W = imgs[s].shape[1]
        fits = [(S[k][3] - cur[(k, sh)] - W, k, sh, clut) for clut, sh in LAYERS for k in range(len(S)) if cur[(k, sh)] + W <= S[k][3]]
        assert fits, '자리 모자람 %X %s' % (s, nm[s])
        _, k, sh, clut = min(fits)
        t, v, u0, u1 = S[k]; u = cur[(k, sh)]; x0 = (t - 0xA) * 256
        px[v:v + 24, x0 + u:x0 + u + W] |= (imgs[s] << sh).astype(np.uint8)
        spot[s] = (t, u, v, W, clut); cur[(k, sh)] = u + W + 1
    out = {}
    for s, rs in G.items():
        t, u, v, W, clut = spot[s]
        for r in rs:
            out[r['o']] = (t, u, v, 0, W, clut)
    return recs, px, orig, out, G, nm


def patch34(F, out):
    F = bytearray(F)
    for o, (t, u, v, x, w, clut) in out.items():
        F[o] = u; F[o + 1] = v
        struct.pack_into('<h', F, o + 2, x)
        struct.pack_into('<H', F, o + 6, w)
        struct.pack_into('<H', F, o + 8, 24)
        struct.pack_into('<H', F, o + 10, t)
        struct.pack_into('<H', F, o + 12, clut)
    return bytes(F)


def apply(pac):
    recs, px, _, out, _, _ = make(bytes(pac))
    a = TS.FILE34
    pac[a:a + 0xE000] = patch34(bytes(pac[a:a + 0xE000]), out)
    enc = encode(px); pac[DATA:DATA + len(enc)] = enc
    return len(out)


def preview():
    import vram as V
    pac = open(MK.PAC, 'rb').read()
    recs, px, orig, out, G, nm = make(pac)
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '16.sav'))

    def pal(c):
        cx, cy = (c & 0x3F) * 16, c >> 6; cc = v[cy, cx:cx + 16].astype(int)
        return np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8), cc == 0

    def comp(P, rs, geo):
        W = max(140, max(geo(r)[3] + geo(r)[4] for r in rs)); cv = np.full((24, W, 3), (200, 190, 160), np.uint8)
        for r in rs:
            t, u, vv, x, w, c = geo(r)
            a = P[vv:vv + 24, (t - 0xA) * 256 + u:(t - 0xA) * 256 + u + w]
            p, tr = pal(c); m = ~tr[a]
            cv[:, x:x + w][m] = p[a][m]
        return cv

    rows = []
    for s, rs in G.items():
        a = comp(orig, rs, lambda r: (r['tp'], r['u'], r['v'], r['x'], r['w'], r['clut']))
        b = comp(px, rs, lambda r: out[r['o']])
        W = 160
        a = np.pad(a, ((0, 0), (0, max(0, W - a.shape[1])), (0, 0)), constant_values=40)[:, :W]
        b = np.pad(b, ((0, 0), (0, max(0, W - b.shape[1])), (0, 0)), constant_values=40)[:, :W]
        rows.append(np.concatenate([a, np.full((24, 3, 3), 255, np.uint8), b], 1))
    cols = 3; per = (len(rows) + cols - 1) // cols
    colimgs = []
    for c in range(cols):
        part = rows[c * per:(c + 1) * per]
        part += [np.full_like(rows[0], 25)] * (per - len(part))
        colimgs.append(np.concatenate([np.concatenate([r, np.full((2, r.shape[1], 3), 25, np.uint8)], 0) for r in part], 0))
    im = np.concatenate([np.concatenate([c, np.full((c.shape[0], 10, 3), 25, np.uint8)], 1) for c in colimgs], 1)
    I = Image.fromarray(im).resize((im.shape[1] * 2, im.shape[0] * 2), Image.NEAREST)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_timetable.png'); I.save(p); print('미리보기', p, '역', len(G))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
