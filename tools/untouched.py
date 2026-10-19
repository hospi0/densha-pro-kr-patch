# -*- coding: utf-8 -*-
r"""스테이트 화면에서 «우리 빌드가 안 바꾼» 텍스처 조각 찾기 (2026-10-18)
  각 스테이트의 그리기 명령(FT4·GT4·SPRT, gpu_prims) → 텍스처 자리(VRAM) → 그 자리를 채운 TXS(스테이트 VRAM 과 바이트 일치) →
  빌드 bin 의 PAC 에서 그 TXS 화소가 원본과 같으면 «안 바뀐 조각». SPRT 는 tpage 가 명령에 없어 tpage 후보를 화면(프레임버퍼)과 대조해 고름.
  TXS 에 없는 자리(BIOS 글꼴로 즉석에 그린 평문 등)는 뺌.
  python tools/untouched.py [스테이트 번호…] → work/gfx/untouched_<n>.png (안 바뀐 조각, 같은 그림 한 번만)
"""
import json
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gpu_prims as GP  # noqa: E402
import sprites as SP  # noqa: E402
import vram as V  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
BIN = os.path.join(ROOT, 'work', 'build', 'Densha de Go! Professional Shiyou (Japan).bin')
PAC_LBA, SEC, USER = 175, 2352, 2048


def built_pac(n):
    out = bytearray(n)
    with open(BIN, 'rb') as f:
        for s in range((n + USER - 1) // USER):
            f.seek((PAC_LBA + s) * SEC + 24)
            out[s * USER:(s + 1) * USER] = f.read(USER)[:max(0, min(USER, n - s * USER))]
    return bytes(out)


def txs_rects():
    out = []
    for off, rects, _ in json.load(open(os.path.join(ROOT, 'work', 'txs_list.json'))):
        p = off + 0x800
        for x, y, w, h in rects:
            out.append((x, y, w, h, p)); p += w * h * 2
    return out


def source(pac, bpac, v, R, y0, y1, x0, x1):
    """VRAM 사각형을 «바이트 그대로» 채운 TXS 를 찾아 → None(출처 없음) / True(빌드에서 바뀜) / False(안 바뀜)"""
    found = None
    for x, y, w, h, p in R:
        if not (x <= x0 and x1 <= x + w and y <= y0 and y1 <= y + h):
            continue
        rows = []
        for yy in range(y0, y1):
            o = p + ((yy - y) * w + (x0 - x)) * 2
            rows.append(o)
        if all(pac[o:o + (x1 - x0) * 2] == v[yy, x0:x1].tobytes() for yy, o in zip(range(y0, y1), rows)):
            ch = any(pac[o:o + (x1 - x0) * 2] != bpac[o:o + (x1 - x0) * 2] for o in rows)
            if ch:
                return True
            found = False
    return found


def tex(v, tp, u, vv, w, h, clut):
    """4bpp/8bpp 텍스처 조각 → (색 배열 RGBA, VRAM halfword 사각형)"""
    tx, ty = (tp & 15) * 64, ((tp >> 4) & 1) * 256; bpp = (tp >> 7) & 3
    cx, cy = (clut & 0x3F) * 16, clut >> 6
    if bpp == 0:
        x0 = tx + u // 4; x1 = tx + (u + w + 3) // 4
        blk = v[ty + vv:ty + vv + h, x0:x1]
        px = np.stack([blk & 15, (blk >> 4) & 15, (blk >> 8) & 15, blk >> 12], -1).reshape(h, -1)[:, u % 4:u % 4 + w]
        pal = v[cy, cx:cx + 16]
    elif bpp == 1:
        x0 = tx + u // 2; x1 = tx + (u + w + 1) // 2
        blk = v[ty + vv:ty + vv + h, x0:x1]
        px = np.stack([blk & 255, blk >> 8], -1).reshape(h, -1)[:, u % 2:u % 2 + w]
        pal = v[cy, cx:cx + 256]
    else:
        return None
    if px.shape != (h, w) or px.max() >= len(pal):
        return None
    return pal[px], (ty + vv, ty + vv + h, x0, x1)


def rgb(c):
    c = c.astype(int)
    return np.stack([(c & 31) << 3, ((c >> 5) & 31) << 3, ((c >> 10) & 31) << 3], -1).astype(np.uint8)


def run(st, pac, bpac):
    sav = os.path.join(ROOT, 'work', 'shots', '%s.sav' % st)
    v = V.vram(sav); ram = V.ram(sav)
    R = txs_rects()
    tps = list(range(32))                                   # SPRT tpage 후보 = 4bpp 전 칸
    seen = set(); out = []
    loose = [('SPR', 0, x, y, w, h, u, vv, w, h, c, None) for (y, x, w, h, u, vv, c) in SP.sprites(ram)]   # 연결 목록 밖 SPRT 도
    for k, i, x, y, w, h, u, vv, uw, vh, clut, tp in GP.prims(ram) + loose:
        if uw < 0:
            u, uw = u + uw, -uw
        if vh < 0:
            vv, vh = vv + vh, -vh
        if uw < 6 or vh < 6 or u + uw > 256 or vv + vh > 256:
            continue
        cands = [tp] if tp is not None else tps
        best = None                                           # SPRT 는 tpage 가 없음 → TXS 출처가 바이트로 확인되는 후보 중 잉크가 가장 많은 것
        for t in cands:
            r = tex(v, t, u, vv, uw, vh, clut)
            if r is None:
                continue
            col, (y0, y1, x0, x1) = r
            ink = int((col != 0).sum())
            if ink < 8 or (best and ink <= best[0]):
                continue
            src = source(pac, bpac, v, R, y0, y1, x0, x1)
            if src is not None:
                best = (ink, t, col, src)
        if best is None or best[3]:
            continue
        _, t, col, _ = best
        key = (t, u, vv, uw, vh, clut)
        if key in seen:
            continue
        seen.add(key)
        out.append((key, col))
    W = 1400; cv = Image.new('RGB', (W, 6000), (40, 30, 50)); d = ImageDraw.Draw(cv); X = Y = RH = 0
    for key, col in sorted(out, key=lambda a: (a[0][0], a[0][2], a[0][1])):
        im = rgb(col); im[col == 0] = (40, 30, 50)
        I = Image.fromarray(im).resize((im.shape[1] * 2, im.shape[0] * 2), Image.NEAREST)
        if X + max(I.width, 90) > W:
            X = 0; Y += RH + 14; RH = 0
        d.text((X, Y), 't%X %d,%d' % (key[0], key[1], key[2]), fill=(255, 255, 0))
        cv.paste(I, (X, Y + 11)); X += max(I.width, 90) + 8; RH = max(RH, I.height + 11)
    p = os.path.join(ROOT, 'work', 'gfx', 'untouched_%s.png' % st)
    cv.crop((0, 0, W, min(6000, Y + RH + 12))).save(p)
    return len(out), p


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    pac = open(PAC, 'rb').read(); bpac = built_pac(len(pac))
    for st in sys.argv[1:] or [str(i) for i in range(1, 19)]:
        print(st, *run(st, pac, bpac))
