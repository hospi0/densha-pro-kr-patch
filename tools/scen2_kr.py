# -*- coding: utf-8 -*-
r"""풍경 이름표 2차 — 큐슈·오사카 순환·야마노테·케이힌 2 (2026-10-19, 사용자 스샷 東バイパス·多々良川·博多臨港線 «이런 표지 다 찾아봐»)
  1차 scen_kr 는 노선 TXS 사각형 1 만 봤음. 이 노선들은 사각형 0 에 있음 → 노선 모듈의 이름표 레코드 표(tools/scen_scan.py)로 정확한 자리:
    레코드 12 B = [u][v][w][h=16][0×4][tpage][CLUT]. 큐슈 PAC 0x375B4(18) · 오사카 0x4A820(19) · 야마노테 0x19F30 = 0x328D4(5) · 케이힌2 0x99A8(앞 10 = B02800 자리, 뒤 10 = BF4000 자리).
  이름표마다 CLUT 이 달라 비트 층을 나눠 쓰는 표가 있음 → CLUT 이 «읽는 비트»만 고침(다른 비트는 그대로), 바탕 = 원래 최빈 색, 글자 = 가장 어두운 색.
  글꼴 = scen_kr 과 같게 갈무리9(hud_px 'S', 넘치면 7).
  python tools/scen2_kr.py → work/gfx/prev_scen2.png (원본|한글, 실제 CLUT 색)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hud_px as HP  # noqa: E402
import scen_scan as SS  # noqa: E402

PAC = SS.PAC
KYUSHU = ['후쿠오카 도시고속 2호선', '동부 우회로', '타타라강', '하카타 임항선', '카시이선', '국도 3호', '온가강', '치쿠호 본선',
          '스페이스 월드', '신일본제철 전용선', '와카토 대교', '키타큐슈 도시고속 토바타 램프', '국도 199호', '키타큐슈 도시고속 3호선',
          '산요 신칸센', '히가시코쿠라 화물역', '모지 기관구', '칸몬 터널']
OSAKA = ['메이신 고속도로', '한큐 쿄토선', '킨키 자동차도', '오사카 모노레일', '부도 중앙 환상선', '오카와강', '한큐선', '한신 고속 이케다선',
         '한신 고속 코베선', '아지강', '지하철 츄오선', '한신 고속 오사카항선', '시리나시강', '사카이가와 신호장', '오사카 돔', '키즈강',
         '한신 고속 사카이선', '난카이 본선', '한신 고속 14호선']
YAMA = ['토요코선', '에비스 가든 플레이스', '수도 고속 메구로선', '제1 케이힌', '케이큐 본선']
KEIHIN2 = ['요코스카선', '칸파치 거리', '칸나나 거리', '타마강', '난부 지선', '힌카쿠 화물선', '츠루미강', '츠루미선', '요코하마선', '수도 고속 미츠자와선']
# (표 위치, 레코드 범위, 번역, 이 표를 쓰는 노선 TXS)
JOBS = [(0x375B4, slice(0, 18), KYUSHU, [0xFC8000]),
        (0x4A820, slice(0, 19), OSAKA, [0x125C000]),
        (0x19F30, slice(0, 5), YAMA, [0xEEF800]),
        (0x99A8, slice(0, 10), KEIHIN2, [0xB02800]),
        (0x99A8, slice(10, 20), KEIHIN2, [0xBF4000])]


def rects(pac, off):
    n = struct.unpack_from('<I', pac, off + 4)[0]; a = off + 0x800; out = []
    for k in range(n):
        x, y, w, h = struct.unpack_from('<4H', pac, off + 8 + 8 * k)
        out.append((a, x, y, w, h)); a += w * h * 2
    return out


def load(pac, off):
    V = np.zeros((512, 1024), np.uint16)
    for a, x, y, w, h in rects(pac, off):
        V[y:y + h, x:x + w] = np.frombuffer(pac[a:a + w * h * 2], '<u2').reshape(h, w)
    return V


def store(pac, off, V):
    for a, x, y, w, h in rects(pac, off):
        pac[a:a + w * h * 2] = V[y:y + h, x:x + w].astype('<u2').tobytes()


def recs(pac, s):
    out = []; o = s
    while SS.ok(SS.rec(pac, o)):
        out.append(SS.rec(pac, o)); o += 12
    return out


def pal(V, cl):
    cc = V[cl >> 6, (cl & 63) * 16:(cl & 63) * 16 + 16].astype(int)
    return np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3, (cc != 0) * 255], -1)


def get4(V, x, y, u, v, w):
    blk = V[y + v:y + v + 16, x + u // 4:x + (u + w + 3) // 4]
    return np.stack([(blk >> (4 * k)) & 15 for k in range(4)], -1).reshape(16, -1)[:, u % 4:u % 4 + w]


def put4(V, x, y, u, v, w, px):
    x0 = x + u // 4; x1 = x + (u + w + 3) // 4
    blk = V[y + v:y + v + 16, x0:x1]
    full = np.stack([(blk >> (4 * k)) & 15 for k in range(4)], -1).reshape(16, -1)
    full[:, u % 4:u % 4 + w] = px
    f = full.reshape(16, -1, 4).astype(np.uint16)
    V[y + v:y + v + 16, x0:x1] = f[..., 0] | (f[..., 1] << 4) | (f[..., 2] << 8) | (f[..., 3] << 12)


def draw(V, r, text):
    u, v, w, h, z, tp, cl = r
    x = (tp & 15) * 64; y = ((tp >> 4) & 1) * 256
    old = get4(V, x, y, u, v, w); P = pal(V, cl); idx = np.arange(16)
    mask = sum(1 << b for b in range(4) if (P[idx] != P[idx ^ (1 << b)]).any())
    col = P[old]
    keys = [tuple(c) for c in col.reshape(-1, 4)]
    from collections import Counter
    cnt = Counter(keys); bg = np.array(cnt.most_common(1)[0][0])
    dark = min((np.array(k) for k in cnt if k[3]), key=lambda c: c[:3].sum())
    lv = HP.word(text, w, 16, 'S', 'c')
    want = np.where((lv == 3)[..., None], dark, bg)
    new = old.copy()
    subs = [s for s in range(16) if s & (15 ^ mask) == 0]
    for yy in range(16):
        for xx in range(w):
            base = int(old[yy, xx]) & (15 ^ mask)
            best = min(subs, key=lambda s: np.abs(P[base | s] - want[yy, xx]).sum())
            new[yy, xx] = base | best
    put4(V, x, y, u, v, w, new)
    return old, new, P


def apply(pac, preview=None):
    n = 0
    for t, sl, texts, txs in JOBS:
        rs = recs(bytes(pac), t)[sl]
        assert len(rs) == len(texts), (hex(t), len(rs), len(texts))
        for off in txs:
            V = load(bytes(pac), off)
            for r, tx in zip(rs, texts):
                old, new, P = draw(V, r, tx); n += 1
                if preview is not None:
                    preview.append((tx, P[old], P[new]))
            store(pac, off, V)
    return n


def preview():
    pac = bytearray(open(PAC, 'rb').read()); items = []
    apply(pac, items)
    W = 1200; cv = Image.new('RGB', (W, 20 + 36 * ((len(items) + 1) // 2)), (40, 40, 48)); X = Y = 0
    for k, (tx, a, b) in enumerate(items):
        x0 = (k % 2) * 600; y0 = (k // 2) * 36
        for i, im in enumerate((a, b)):
            rgb = im[..., :3].astype(np.uint8).copy(); rgb[im[..., 3] == 0] = (255, 0, 255)
            I = Image.fromarray(rgb).resize((im.shape[1] * 2 // 1, 32), Image.NEAREST)
            cv.paste(I, (x0 + i * (I.width + 6), y0 + 2))
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_scen2.png'); cv.save(p); print(len(items), '→', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
