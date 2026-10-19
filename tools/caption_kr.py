# -*- coding: utf-8 -*-
r"""영상 플로어 제목(조각 글자 조합) 한글화 (2026-10-18)
  그림 = 메뉴 TXS 989A000 사각형 2 (VRAM 768,0 64×256 = tpage C), 데이터 PAC 0x98CA800. 글자 칸 높이 24, 값 3 = 획·2 = 오른쪽 그림자·1 = 옅은 가장자리 (CLUT 3D2F).
  조각 표 = 22번 파일 0x19F2A~0x1A3DA (CLUT 3D2F): 제목 18개 = 첫 줄 «노선 + 종류» 조각들(y0) + 둘째 줄 オープニング/エンディング(y24).
  ★한국어는 낱말 순서·길이가 달라 «낱말 단위»로 새로 그려 빈 칸에 다시 배치하고 레코드 u·v·x·w 를 고침. 첫 줄 낱말 수보다 레코드가 많으면 남는 레코드는 마지막 낱말을 겹쳐 그림.
  제목 문구 = 평문 04D320~04D514 번역과 같게. おまけ(お♥ま♥け) = 보♥너♥스(♥ 조각 그대로).
  python tools/caption_kr.py → work/gfx/prev_caption.png (원본|한글)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import btn_kr as BT  # noqa: E402
import floor_kr as FK  # noqa: E402
import menu_kr as MK  # noqa: E402

DATA = 0x98CA800
CLUT = 0x3D2F
H_INK = 15
FONT = r'C:\claude\utils\font\nanum-gothic\NanumGothicBold.ttf'   # 원본 글자가 가늘어 ExtraBold 는 무거움
GAP = 5                                        # 낱말 사이(전각 공백 자리)
# 제목 묶음 첫 레코드 → (첫 줄 낱말들, 둘째 줄 낱말)
TITLES = {
    0x19F2A: (['프로페셔널판'], '오프닝'), 0x19F4A: (['아키타・토호쿠신칸센'], '엔딩'), 0x19F8A: (['오우・타자와코선'], '엔딩'),
    0x19FDA: (['호쿠호쿠선', '특급'], '엔딩'), 0x1A01A: (['호쿠호쿠선', '보통・쾌속'], '엔딩'), 0x1A07A: (['야마노테선'], '엔딩'),
    0x1A0AA: (['케이힌토호쿠선・토카이도선'], '엔딩'), 0x1A11A: (['사가노선'], '엔딩'), 0x1A14A: (['쿄토선', '보통・쾌속・신쾌속'], '엔딩'),
    0x1A1CA: (['쿄토선', '특급'], '엔딩'), 0x1A20A: (['103계'], '엔딩'), 0x1A22A: (['오사카순환선'], '엔딩'),
    0x1A25A: (['카고시마본선', '보통・쾌속'], '엔딩'), 0x1A2AA: (['카고시마본선', '특급'], '엔딩'), 0x1A2DA: (['카고시마본선', '특급소닉'], '엔딩'),
    0x1A36A: (['오사카순환선', '103계'], '엔딩'), 0x1A3AA: (['오사카순환선', '쾌속'], '엔딩'),
}
OMAKE = (0x1A31A, ['보', None, '너', None, '스'])          # お ♥ ま ♥ け → None = ♥ 레코드 그대로
# 빈 칸(새 낱말 자리): (v, u0, u1) 높이 24 — 원래 글자 줄 전부(v80~175 전폭, v176 줄은 u0~143 — u123 v196 에 남의 화소 1개)
FREE = [(80, 0, 256), (104, 0, 256), (128, 0, 256), (152, 0, 256), (176, 0, 144)]
SYL = (200, 0, 48)
BAN = (200, 0, 40)                                         # 番 숫자판(CLUT 3FAF): 글자 a 흰·9 회색, 가름줄 18~21줄에선 4·9 → 줄 최빈값으로 지우고 «번»                                         # おまけ 의 お·ま·け 칸 줄(♥ 는 u48)


def load_px(pac):
    raw = np.frombuffer(pac[DATA:DATA + 0x8000], np.uint8).reshape(256, 128)
    return np.stack([raw & 15, raw >> 4], -1).reshape(256, 256).copy()


def encode(px):
    p = px.reshape(256, 128, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def glyph(cov):
    """덮임 → 높이 24 칸 값(3 획·1 옅음·2 오른쪽 그림자)"""
    hh, ww = cov.shape
    out = np.zeros((24, ww + 1), np.uint8)
    lv = np.where(cov >= 0.5, 3, np.where(cov >= 0.2, 1, 0)).astype(np.uint8)
    y0 = 4 + (17 - hh) // 2
    out[y0:y0 + hh, :ww] = lv
    sh = np.zeros_like(out); sh[:, 1:] = (out[:, :-1] == 3) & (out[:, 1:] == 0)
    out[sh.astype(bool)] = 2
    return out


def groups(recs):
    out = {}; cur = None
    for r in recs:
        if r['o'] in TITLES or r['o'] == OMAKE[0]:
            cur = r['o']; out[cur] = []
        if cur is not None and r['clut'] == CLUT:
            if out[cur] and r['o'] - out[cur][-1]['o'] != 16:
                cur = None; continue
            out[cur].append(r)
    return out


def make(pac):
    """→ recs, 새 화소, 원본 화소, {레코드: (u, v, x, w)}"""
    ov = pac[MK.OV:MK.OV + 0x2F800]
    recs = [r for r in FK.records(ov) if r['tp'] == 0xC]
    px = load_px(pac); orig = px.copy()
    words = sorted({w for a, b in TITLES.values() for w in a + [b]}, key=len, reverse=True)
    # 빈 칸 비우기
    for v, u0, u1 in FREE + [SYL]:
        reg = px[v:v + 24, u0:u1]
        # 값 4~ 는 남의 그림(지우지 않음)
        reg[reg <= 3] = 0
    # 같은 크기·같은 가로 비율로, 다 들어갈 때까지 비율을 줄임
    for xs in np.arange(1.0, 0.6, -0.02):
        imgs = {w: glyph(BT.cover(w, H_INK, xs, font=FONT)[0]) for w in words}
        spot = {}; cur = [u0 for _, u0, _ in FREE]
        ok = True
        for w in sorted(words, key=lambda w: -imgs[w].shape[1]):
            W = imgs[w].shape[1]
            k = next((i for i, (v, u0, u1) in enumerate(FREE) if cur[i] + W <= u1), None)
            if k is None:
                ok = False; break
            spot[w] = (cur[k], FREE[k][0]); cur[k] += W + 1
        if ok:
            break
    assert ok, '자리 모자람'
    for w, (u, v) in spot.items():
        img = imgs[w]; px[v:v + 24, u:u + img.shape[1]] |= img
    syl = {}
    for i, t in enumerate([t for t in OMAKE[1] if t]):
        img = glyph(BT.cover(t, H_INK, xs, font=FONT)[0])[:, :16]
        u = SYL[1] + 16 * i; ox = (16 - img.shape[1]) // 2                      # 칸 가운데
        px[SYL[0]:SYL[0] + 24, u + ox:u + ox + img.shape[1]] |= img
        syl[t] = (u, SYL[0], 16)
    # 番 → 번
    u0, v0, w0 = BAN
    plate = px[v0:v0 + w0, u0:u0 + w0]
    for y in range(w0):
        row = plate[y]; glyph_v = {0xA, 9} | ({4} if y in (18, 19) else set())
        bg = np.bincount([c for c in row if c not in glyph_v] or [1]).argmax()
        row[np.isin(row, list(glyph_v))] = bg
    c = BT.cover('번', 28)[0]
    hh, ww = c.shape; oy = 4 + (32 - hh) // 2; ox = (w0 - ww) // 2
    for y in range(hh):
        Y = oy + y; hi, lo = (4, 4) if Y in (18, 19) else (0xA, 9)
        r = plate[Y, ox:ox + ww]
        r[c[y] >= 0.5] = hi; r[(c[y] >= 0.2) & (c[y] < 0.5)] = lo
    out = {}
    G = groups(recs)
    for g0, (line1, line2) in TITLES.items():
        rs = G[g0]
        top = [r for r in rs if r['y'] == 0]; bot = [r for r in rs if r['y'] == 24]
        assert len(top) >= len(line1) and len(bot) == 1, ('%X' % g0, len(top), len(bot))
        x = 0
        for k, r in enumerate(top):
            w = line1[min(k, len(line1) - 1)]
            u, v = spot[w]; W = imgs[w].shape[1]
            if k < len(line1):
                out[r['o']] = (u, v, x, W); last = (u, v, x, W); x += W + GAP
            else:
                out[r['o']] = last
        u, v = spot[line2]; out[bot[0]['o']] = (u, v, 0, imgs[line2].shape[1])
    rs = G[OMAKE[0]]
    for r, t in zip(rs, OMAKE[1]):
        if t:
            u, v, w = syl[t]; out[r['o']] = (u, v, r['x'], w)
    return recs, px, orig, out, xs


def patch_ov(ov, out):
    ov = bytearray(ov)
    for o, (u, v, x, w) in out.items():
        ov[o] = u; ov[o + 1] = v
        struct.pack_into('<h', ov, o + 2, x)
        struct.pack_into('<H', ov, o + 6, w)
    return bytes(ov)


def apply(pac):
    recs, px, _, out, _ = make(bytes(pac))
    pac[MK.OV:MK.OV + 0x2F800] = patch_ov(bytes(pac[MK.OV:MK.OV + 0x2F800]), out)
    enc = encode(px); pac[DATA:DATA + len(enc)] = enc
    return len(out)


def preview():
    import vram as V
    pac = open(MK.PAC, 'rb').read()
    recs, px, orig, out, xs = make(pac)
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '13.sav'))
    cx, cy = (CLUT & 0x3F) * 16, CLUT >> 6; cc = v[cy, cx:cx + 16].astype(int)
    pal = np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8)
    pal[0] = (30, 50, 30)

    def comp(P, rs, geo):
        W = max(geo(r)[2] + geo(r)[3] for r in rs); H = 48
        cv = np.full((H, W, 3), (30, 50, 30), np.uint8)
        for r in rs:
            u, vv, x, w = geo(r)
            a = P[vv:vv + 24, u:u + w]; m = a > 0
            cv[r['y']:r['y'] + 24, x:x + w][m] = pal[a][m]
        return cv

    G = groups(recs)
    Wc = 1400; cvs = Image.new('RGB', (Wc, 2000), (25, 25, 25)); d = ImageDraw.Draw(cvs); X = Y = RH = 0
    for g0, rs in G.items():
        a = comp(orig, rs, lambda r: (r['u'], r['v'], r['x'], r['w']))
        b = comp(px, rs, lambda r: out.get(r['o'], (r['u'], r['v'], r['x'], r['w'])))
        pair = np.concatenate([a, np.full((48, 3, 3), 200, np.uint8), b], 1)
        I = Image.fromarray(pair).resize((pair.shape[1] * 2, 96), Image.NEAREST)
        if X + I.width > Wc:
            X = 0; Y += RH + 14; RH = 0
        cvs.paste(I, (X, Y + 12)); d.text((X, Y), '%X' % g0, fill=(255, 255, 0)); X += I.width + 12; RH = max(RH, 108)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_caption.png')
    cvs.crop((0, 0, Wc, Y + RH + 10)).save(p); print('미리보기', p, '가로 비율 %.2f' % xs)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
