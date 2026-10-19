# -*- coding: utf-8 -*-
r"""게임 중 HUD 아틀라스 공용 (2026-10-19)
  아틀라스 = TXS D6000 (사각형 0 (320,0,64,512) = 4bpp 256×512, tpage 0x05(v<256)/0x15(v≥256) · 사각형 1 = CLUT 구역 (256,464,64,48))
    노선별 사본 E8800·FB000·10D800·120000·132800 + B894000 사각형 2·3 — 다 같게 써야 함(COPIES).
  조각 표 = 실행 파일 SLPM_863.78 0x3C294~0x40674, 12 B [u][v][w][h][x s16][y s16][tpage u16][clut u16]
    tpage 0x10 비트 = v+256, 0x8000 = 플래그. 12 B 연속 + 자리 안 겹침 = 화면 한 덩어리(groups()).
  python tools/hud.py groups → work/gfx/mf/hud_groups2.png (그룹 주소별 원본 조합)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
EXE = os.path.join(ROOT, 'work', 'disc', 'SLPM_863.78')
COPIES = [(0xD6000, 0), (0xE8800, 0), (0xFB000, 0), (0x10D800, 0), (0x120000, 0), (0x132800, 0), (0xB894000, 2)]
REC_LO, REC_HI = 0x3C294, 0x40674
CLUT_X, CLUT_Y = 256, 464


def rect_offsets(pac, off):
    n = struct.unpack_from('<I', pac, off + 4)[0]
    rs = [struct.unpack_from('<4H', pac, off + 8 + 8 * i) for i in range(n)]
    out, a = [], off + 0x800
    for r in rs:
        out.append((a, r)); a += r[2] * r[3] * 2
    return out


def load(pac, copy=COPIES[0]):
    """→ px (512,256) 4bpp 색인, clut (48,64) u16"""
    off, k = copy
    ro = rect_offsets(pac, off)
    a, r = ro[k]; assert r == (320, 0, 64, 512), r
    raw = np.frombuffer(pac[a:a + 0x10000], np.uint8).reshape(512, 128)
    px = np.stack([raw & 15, raw >> 4], -1).reshape(512, 256).copy()
    a, r = ro[k + 1]; assert r == (256, 464, 64, 48), r
    clut = np.frombuffer(pac[a:a + 64 * 48 * 2], '<u2').reshape(48, 64).copy()
    return px, clut


def store(pac, px, copies=COPIES):
    """★사본 7개는 똑같지 않다(노선·차종별 속도계 그림 u0~63 v384~447, B894000 은 더 여러 곳) — 11차 실기 속도계 깨짐.
    첫 사본(지금 PAC 상태) 대비 «바뀐 화소만» 각 사본에 씀, 나머지는 그 사본 고유 그림 그대로."""
    base, _ = load(bytes(pac), copies[0]); mask = px != base
    for c in copies:
        cur, _ = load(bytes(pac), c)
        cur[mask] = px[mask]
        p = cur.reshape(512, 128, 2); enc = (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()
        a, r = rect_offsets(pac, c[0])[c[1]]
        pac[a:a + len(enc)] = enc


def copies_differ(pac):
    """사본끼리 화소가 다른 자리(어느 한 사본이라도 첫 사본과 다르면 True) — 여기엔 쓰면 안 됨"""
    base, _ = load(pac, COPIES[0]); d = np.zeros(base.shape, bool)
    for c in COPIES[1:]:
        d |= load(pac, c)[0] != base
    return d


def pal(clut, c):
    """CLUT 낱말 → (16,4) RGBA"""
    x, y = (c & 0x3F) * 16 - CLUT_X, (c >> 6) - CLUT_Y
    if not (0 <= y < 48 and 0 <= x <= 48):               # 구역 밖 CLUT(다른 TXS) → 회색 계조
        g = np.arange(16) * 17; out = np.stack([g, g, g, np.where(g == 0, 0, 255)], -1); return out.astype(np.uint8)
    v = clut[y, x:x + 16].astype(int)
    rgb = np.stack([(v & 31) << 3, ((v >> 5) & 31) << 3, ((v >> 10) & 31) << 3], -1)
    a = np.where(v == 0, 0, 255)
    return np.concatenate([rgb, a[:, None]], -1).astype(np.uint8)


def records(exe):
    out = []
    for o in range(REC_LO, REC_HI, 12):
        u, v, w, h, x, y, tp, cl = struct.unpack_from('<4B2h2H', exe, o)
        out.append(dict(o=o, u=u, v=v + (256 if tp & 0x10 else 0), w=w, h=h, x=x, y=y, tp=tp, clut=cl))
    return out


def groups(recs):
    """연속 레코드 중 자리가 안 겹치는 것끼리 한 덩어리"""
    gs, cur = [], []
    for r in recs:
        if r['w'] == 0 or r['h'] == 0:
            if cur:
                gs.append(cur); cur = []
            continue
        hit = any(not (r['x'] >= q['x'] + q['w'] or q['x'] >= r['x'] + r['w'] or r['y'] >= q['y'] + q['h'] or q['y'] >= r['y'] + r['h']) for q in cur)
        if hit and cur:
            gs.append(cur); cur = []
        cur.append(r)
    if cur:
        gs.append(cur)
    return gs


def compose(px, clut, g, bg=(40, 40, 48, 255)):
    x0 = min(r['x'] for r in g); y0 = min(r['y'] for r in g)
    x1 = max(r['x'] + r['w'] for r in g); y1 = max(r['y'] + r['h'] for r in g)
    im = np.zeros((y1 - y0, x1 - x0, 4), np.uint8); im[:] = bg
    for r in g:
        p = pal(clut, r['clut'])[px[r['v']:r['v'] + r['h'], r['u']:r['u'] + r['w']]]
        sub = im[r['y'] - y0:r['y'] - y0 + r['h'], r['x'] - x0:r['x'] - x0 + r['w']]
        m = p[..., 3] > 0; sub[m] = p[m]
    return im


def sheet(pac, exe, path, scale=2):
    px, clut = load(pac); gs = groups(records(exe))
    tiles = []
    for g in gs:
        im = compose(px, clut, g)
        if im.shape[0] > 200 or im.shape[1] > 400:
            continue
        tiles.append(('%X' % g[0]['o'], im))
    W = 1400; x = y = rowh = 0; pos = []
    for name, im in tiles:
        w, h = im.shape[1] * scale, im.shape[0] * scale + 12
        if x + w > W:
            x = 0; y += rowh + 4; rowh = 0
        pos.append((x, y)); x += w + 6; rowh = max(rowh, h)
    out = Image.new('RGB', (W, y + rowh + 4), (0, 0, 0)); d = ImageDraw.Draw(out)
    for (name, im), (x, y) in zip(tiles, pos):
        out.paste(Image.fromarray(im).convert('RGB').resize((im.shape[1] * scale, im.shape[0] * scale), Image.NEAREST), (x, y + 12))
        d.text((x, y), name, fill=(255, 255, 0))
    out.save(path)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    pac = open(PAC, 'rb').read(); exe = open(EXE, 'rb').read()
    if sys.argv[1:] == ['groups']:
        p = os.path.join(ROOT, 'work', 'gfx', 'mf', 'hud_groups2.png'); sheet(pac, exe, p); print(p)
