# -*- coding: utf-8 -*-
r"""HUD 위 줄 노란 작은 역 이름(秋田▶大曲) 그림 판독·한글화 (2026-10-19)
  TXS (384,384,32,64) 8장 = 4bpp 128×64, 색인 비트 하나 = 역 이름 한 벌(1비트 그림), 칸 = u 12×k(k 0~7) × v 0~47, 이름은 칸 안 세로 가운데.
  비트 0 은 대개 전부 1(채움) → 글자 비트만(화소 수 0 < n < 전체) 판독.
  판독 = MS 고딕 12px 상관(1비트라 잘 맞음) → work/stn_small.json, 검수 그림 work/gfx/chk_stn_small.png.
  한글 = work/text/stn_board_kr.tsv(+ EXTRA), 갈무리11(1px 획) 12px 간격, 4음절 넘으면 간격 줄이고 작은 갈무리.
  python tools/stn_small.py read   → 판독표·검수 그림
  python tools/stn_small.py prev   → work/gfx/prev_stn_small.png (원본|한글)
"""
import json
import os
import sys
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pixfont as PF  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
TXS = [0x9378000, 0x937A000, 0x937C000, 0x937E000, 0x9380000, 0x9382000, 0x9384000, 0x9386000]
W, H = 128, 64
EXTRA = {'門司': '모지', '小森江': '코모리에', '門司港': '모지코', '代々木': '요요기', '新宿': '신주쿠', '原宿': '하라주쿠',
         '渋谷': '시부야', '梅小路': '우메코지'}


def load(pac, off):
    a = off + 0x800; raw = np.frombuffer(pac[a:a + W * H // 2], np.uint8).reshape(H, W // 2)
    return np.stack([raw & 15, raw >> 4], -1).reshape(H, W).copy()


def encode(px):
    p = px.reshape(H, W // 2, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def names_dict():
    d = {}
    for f in ('stations.tsv', 'stn_board_kr.tsv'):
        for l in open(os.path.join(ROOT, 'work', 'text', f), encoding='utf-8').read().split('\n'):
            a = l.split('\t')
            if len(a) >= 2 and a[0] and a[0] != 'jp':
                d[a[0]] = a[1]
    d.update(EXTRA)
    return d


F = ImageFont.truetype(r'C:\Windows\Fonts\msgothic.ttc', 12)


def rend(name):
    im = Image.new('L', (12, 12 * len(name))); dr = ImageDraw.Draw(im)
    for i, c in enumerate(name):
        dr.text((0, 12 * i), c, font=F, fill=255)
    return (np.array(im) > 100).astype(float)


def columns(px):
    """→ [(비트, 칸, 잉크 위, 잉크 아래, 1비트 그림)]"""
    out = []
    for b in range(4):
        m = ((px >> b) & 1)[:48, :96]
        if m.sum() == 0 or m.all():
            continue
        for k in range(8):
            c = m[:, 12 * k:12 * k + 12]
            ys = np.nonzero(c.any(1))[0]
            if len(ys):
                out.append((b, k, int(ys.min()), int(ys.max()) + 1, c))
    return out


def read():
    pac = open(PAC, 'rb').read(); D = names_dict(); res = []
    by_len = {}
    for nm in D:
        by_len.setdefault(len(nm), []).append(nm)
    for t, off in enumerate(TXS):
        for b, k, y0, y1, c in columns(load(pac, off)):
            seg = c[y0:y1].astype(float); best = (-1, '')
            n = max(1, round((y1 - y0) / 12))
            for L in (n - 1, n, n + 1):
                for nm in by_len.get(L, []):
                    r = rend(nm)
                    if r.shape[0] != seg.shape[0]:
                        r = np.array(Image.fromarray((r * 255).astype(np.uint8)).resize((12, seg.shape[0]))) / 255
                    for dx in (-1, 0, 1):
                        s = np.corrcoef(seg.ravel(), np.roll(r, dx, 1).ravel())[0, 1]
                        if s > best[0]:
                            best = (s, nm)
            res.append(dict(t=t, b=b, k=k, y0=y0, y1=y1, name=best[1], score=round(float(best[0]), 2)))
    json.dump(res, open(os.path.join(ROOT, 'work', 'stn_small.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    # 검수 그림: 칸 그림 + 판독 이름
    f = ImageFont.truetype(r'C:\Windows\Fonts\msgothic.ttc', 14)
    tiles = []
    for r in res:
        c = columns(load(pac, TXS[r['t']]))
        seg = [x for x in c if x[0] == r['b'] and x[1] == r['k']][0][4]
        t = Image.new('RGB', (52, 118), (255, 255, 255)); dr = ImageDraw.Draw(t)
        t.paste(Image.fromarray(np.where(seg, 0, 230).astype(np.uint8)).resize((24, 96), Image.NEAREST), (0, 20))
        dr.text((0, 0), '%d%d%d' % (r['t'], r['b'], r['k']), fill=(200, 0, 0))
        for i, ch in enumerate(r['name']):
            dr.text((30, 20 + 16 * i), ch, font=f, fill=(0, 0, 200) if r['score'] > 0.5 else (220, 0, 0))
        tiles.append(t)
    per = 26; rows = -(-len(tiles) // per)
    sh = Image.new('RGB', (per * 52, rows * 118), (255, 255, 255))
    for i, t in enumerate(tiles):
        sh.paste(t, ((i % per) * 52, (i // per) * 118))
    sh.save(os.path.join(ROOT, 'work', 'gfx', 'chk_stn_small.png'))
    print(len(res), '칸 · 낮은 점수', sum(1 for r in res if r['score'] <= 0.5))


FIX = {(0, 2, 7): '上越国際スキー場前', (0, 3, 4): '大池いこいの森', (0, 3, 7): 'ほくほく大島', (1, 1, 1): '黒崎', (1, 3, 7): '千早',
       (2, 2, 3): '福島', (2, 3, 4): '玉造', (2, 3, 0): '天王寺', (3, 0, 4): '天王寺', (3, 3, 0): '亀岡', (3, 3, 3): '嵯峨嵐山',
       (3, 3, 6): '二条', (5, 2, 4): '田町', (5, 2, 7): '五反田', (6, 2, 1): '田町', (7, 0, 7): '神代', (7, 1, 1): '田沢湖',
       (7, 1, 4): '雫石'}                       # (그림, 비트, 칸) → 일본어 이름(검수 그림 chk_stn_small.png 눈으로 고침)


def vertical(text):
    m = len(text)
    pitch = 12 if m <= 4 else 48 / m
    size = 11 if pitch >= 11 else 9 if pitch >= 9 else 7
    out = np.zeros((48, 12), bool)
    tot = pitch * m; top = (48 - tot) / 2
    for i, ch in enumerate(text):
        g = PF.cell(ch, size) > 0
        gh, gw = g.shape; gw = min(gw, 12); g = g[:, :gw]
        y = int(round(top + i * pitch + (pitch - gh) / 2)); x = (12 - gw) // 2
        y = max(0, min(48 - gh, y))
        out[y:y + gh, x:x + gw] |= g
    return out


def make_one(px, t, items, D):
    px = px.copy()
    for r in items:
        if r['t'] != t:
            continue
        nm = FIX.get((t, r['b'], r['k']), r['name'])
        b, k = r['b'], r['k']
        col = px[:48, 12 * k:12 * k + 12]
        col &= ~np.uint8(1 << b)                                 # 그 비트만 지움(다른 비트 = 다른 역 이름 묶음)
        col |= (vertical(D[nm]).astype(np.uint8) << b)
    return px


def build(pac):
    items = json.load(open(os.path.join(ROOT, 'work', 'stn_small.json'), encoding='utf-8')); D = names_dict()
    return [(off, load(pac, off), make_one(load(pac, off), t, items, D)) for t, off in enumerate(TXS)]


def apply(pac):
    for off, _, new in build(bytes(pac)):
        enc = encode(new); pac[off + 0x800:off + 0x800 + len(enc)] = enc
    return len(TXS)


def prev():
    pac = open(PAC, 'rb').read(); rows = []
    for off, a, b in build(pac):
        for bit in range(1, 4):
            ma = ((a >> bit) & 1)[:48, :96]; mb = ((b >> bit) & 1)[:48, :96]
            if ma.sum() == 0 or ma.all():
                continue
            rows.append(np.concatenate([np.where(ma, 0, 255), np.full((48, 4), 128), np.where(mb, 0, 255)], 1).astype(np.uint8))
            rows.append(np.full((4, 196), 128, np.uint8))
    sh = np.concatenate(rows, 0)
    half = sh.shape[0] // 2 // 52 * 52
    sh = np.concatenate([sh[:half], np.pad(sh[half:], ((0, max(0, 2 * half - sh.shape[0])), (0, 0)), constant_values=128)[:half]], 1)
    Image.fromarray(sh).resize((sh.shape[1] * 3, sh.shape[0] * 3), Image.NEAREST).save(os.path.join(ROOT, 'work', 'gfx', 'prev_stn_small.png'))
    print('미리보기 work/gfx/prev_stn_small.png')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    read() if sys.argv[1:] == ['read'] else prev()
