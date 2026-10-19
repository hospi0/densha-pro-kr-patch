# -*- coding: utf-8 -*-
r"""설치본(F: 이전 시험본) 대비 «지금 코드»로 바뀌는 그림만 한 장에 (2026-10-19, 사용자 «바꾼 것만 하나로 모아서»)
  HUD = 레코드 묶음 조합(실제 CLUT 색) · 로고 = logo_kr.VIEW · 결과/보너스 = 바뀐 상자(CLUT 미상 → 결과 회색·보너스 임의 색) · 플로어 = floor_kr 레코드 묶음
  python tools/diff_sheet.py → work/gfx/diff_sheet.png
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bonus_kr as BK  # noqa: E402
import hud as H  # noqa: E402
import hud_kr as HK  # noqa: E402
import hud_words as W  # noqa: E402
import iso  # noqa: E402
import logo_kr as LG  # noqa: E402
import result_kr as RK  # noqa: E402

DISC = r'F:/hospi/roms/ps roms/Densha de Go Professional Shiyou (Japan)/Densha de Go! Professional Shiyou (Japan).bin'


def disc_reader():
    fl, f = iso.files(DISC)
    lba = [t[1] for t in fl if 'TRAIN.PAC' in t[0].upper()][0]
    exe = [iso.read(f, t[1], t[2]) for t in fl if 'SLPM' in t[0].upper()][0]
    raw = open(DISC, 'rb')

    def rd(off, size):
        out = bytearray(); s = off // 2048
        while len(out) < (off % 2048) + size:
            raw.seek((lba + s) * 2352 + 24); out += raw.read(2048); s += 1
        return bytes(out[off % 2048:off % 2048 + size])
    return rd, exe


def pair(a, b, label):
    H_ = max(a.shape[0], b.shape[0])
    pad = lambda m: np.pad(m, ((0, H_ - m.shape[0]), (0, 0), (0, 0)), constant_values=40)
    return label, np.concatenate([pad(a), np.full((H_, 3, 3), 255, np.uint8), pad(b)], 1)


def rgb(im):
    out = np.full(im.shape[:2] + (3,), 40, np.uint8); m = im[..., 3] > 0; out[m] = im[m][:, :3]; return out


def main():
    rd, exe_old = disc_reader()
    pac = open(H.PAC, 'rb').read(); exe0 = open(H.EXE, 'rb').read()
    p1 = bytearray(pac); HK.apply(p1); exe1, _ = W.apply(p1, exe0); LG.apply(p1)
    old_px, clut = H.load(bytes(0xD6000) + rd(0xD6000, 0x12000))
    new_px, _ = H.load(bytes(p1))
    items = []
    for a, b in zip(H.groups(H.records(exe_old)), H.groups(H.records(exe1))):
        if (a[0]['tp'] & 0xF) != 5:
            continue
        A = H.compose(old_px, clut, a); B = H.compose(new_px, clut, b)
        if A.shape != B.shape or (A != B).any():
            items.append(pair(rgb(A), rgb(B), 'HUD %X' % a[0]['o']))
    for (u0, u1, v0, v1), c in LG.VIEW:
        p = H.pal(clut, c)
        A = p[old_px[v0:v1, u0:u1]]; B = p[new_px[v0:v1, u0:u1]]
        if (A != B).any():
            items.append(pair(rgb(A), rgb(B), '로고 v%d' % v0))
    for offs, w, h, box, edge, text, sc in RK.ITEMS:
        a, px0 = RK.load(pac, offs[0], w, h)
        _, old = RK.load(bytes(offs[0]) + rd(offs[0], 0x800 + w * h * 2), offs[0], w, h)
        new = RK.make(px0, box, edge, text, sc)
        u0, u1, v0, v1 = box
        g = lambda m: np.repeat((m[v0:v1, u0:u1] * 17).astype(np.uint8)[..., None], 3, -1)
        if (old[v0:v1, u0:u1] != new[v0:v1, u0:u1]).any():
            items.append(pair(g(old), g(new), '결과 %s (회색)' % text))
    res = BK.build(pac)
    rng = np.random.RandomState(3); pal = rng.randint(40, 255, (16, 3)).astype(np.uint8); pal[0] = 0; pal[1] = [30, 30, 90]; pal[15] = [255, 255, 255]
    for (off, ri), (a, o, n) in res.items():
        hh, ww = o.shape
        oldb = np.frombuffer(rd(a, hh * ww // 2), np.uint8).reshape(hh, ww // 2)
        oldp = np.stack([oldb & 15, oldb >> 4], -1).reshape(hh, ww)
        d = oldp != n
        if not d.any():
            continue
        ys, xs = np.nonzero(d); y0, y1, x0, x1 = max(0, ys.min() - 4), min(hh, ys.max() + 5), max(0, xs.min() - 4), min(ww, xs.max() + 5)   # 둘레 줄은 안 바뀌어 잘려 보이던 것(준비 «위가 잘림») → 여백
        items.append(pair(pal[oldp[y0:y1, x0:x1]], pal[n[y0:y1, x0:x1]], '보너스 %X (임의 색)' % off))
    # 플로어(음악·보이스·효과음) — floor_kr 미리보기 묶음 중 바뀐 것
    import floor_kr as FK
    import vram as V
    recs, px, orig, moved = FK.make(pac)
    oldraw = np.frombuffer(rd(FK.DATA, px.size // 2), np.uint8).reshape(px.shape[0], px.shape[1] // 2)
    oldfx = np.stack([oldraw & 15, oldraw >> 4], -1).reshape(px.shape)
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '8.sav')); byo = {r['o']: r for r in recs}
    for o in (0x19DBA, 0x19DCA, 0x19DDA):
        r = byo[o]; u, vv, x, w, c = moved.get(o, (r['u'], r['v'], r['x'], r['w'], r['clut']))
        cx, cy = (c & 0x3F) * 16, c >> 6; cc = v[cy, cx:cx + 16].astype(int)
        pl = np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8)
        sl = (slice(vv, vv + r['h']), slice(FK.col0(r['tp']) + u, FK.col0(r['tp']) + u + w))
        if (oldfx[sl] != px[sl]).any():
            items.append(pair(pl[oldfx[sl]], pl[px[sl]], '자료관 %X' % o))
    Wd = 1400; cv = Image.new('RGB', (Wd, 4000), (25, 25, 25)); dr = ImageDraw.Draw(cv); X = Y = RH = 0
    for label, im in items:
        k = 3 if im.shape[1] < 200 else 2
        I = Image.fromarray(im).resize((im.shape[1] * k, im.shape[0] * k), Image.NEAREST)
        if X + I.width > Wd:
            X = 0; Y += RH + 18; RH = 0
        cv.paste(I, (X, Y + 14)); dr.text((X, Y), label, fill=(255, 255, 0))
        X += I.width + 14; RH = max(RH, I.height + 14)
    p = os.path.join(ROOT, 'work', 'gfx', 'diff_sheet.png')
    cv.crop((0, 0, Wd, Y + RH + 10)).save(p); print(len(items), '항목 →', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
