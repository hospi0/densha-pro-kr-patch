# -*- coding: utf-8 -*-
r"""로딩 화면 정차역 판(세로 역 이름) 한글화 (2026-10-19)
  그림 = 로딩 TXS 5장(loading_kr.COPIES, 노선별로 다름) v192~251 의 10×12(0·4번) / 12×12(1~3번) 한자 칸.
  표 = PAC 0x85098~ 16 B 레코드 [깃발 u16 (0x80 = 이름 끝)][u][v][x s16][y s16][w u16][h u16][tpage u16 6][CLUT u16 3EDB] — 한 글자 = 레코드 하나,
    판 안 y = 글자들을 고르게 벌림(0·34·68 등). 레코드 수(=한자 수)는 못 바꾸고 u·v·w·h·y 는 바꿀 수 있음.
  한글: 역마다 음절을 원래 y 범위(0 ~ 마지막 y)에 고르게 벌려 세로로 그리고, 레코드 수만큼 이어진 조각으로 나눠 가리킴.
    칸 자리(v192~251, u0~195)는 그 로딩 그림을 쓰는 역들 이름으로 다시 채움(u196~ 오른쪽 아래 그림은 그대로).
  묶음 → 로딩 그림·역 = 칸 한자 판독(MS 고딕) + 역 사전 맞춤 → work/plate_assign.json (read), 눈 검수 FIX.
  ★2026-10-19 묶음→노선 확정: 판 그리기 0x8006C010(25번 파일 RAM 0x80069000) 이 스프라이트 포인터 표 0x8006ECF8(파일 0x5CF8, 98칸,
    판 묶음 = 칸 4~78) 을 «노선 구조체 0x8006DAB0 + 16×[0x8006EE85]» 의 바이트 목록(파일 0x4A54~, 0xFF = 줄 끝)으로 가리킴.
    줄 9개 = 아키타(사본4)·호쿠호쿠(3)·야마노테·야마노테 북·토카이도(2)·사가노·쿄토선·환상선(1)·카고시마(0). 묶음 18·50 은 안 쓰임.
  python tools/plate_kr.py read → 판독표 / python tools/plate_kr.py → work/gfx/prev_plate.png
"""
import json
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import loading_kr as LK  # noqa: E402
import pixfont as PF  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
T0, T1 = 0x85098, 0x85BB8
ASSIGN = os.path.join(ROOT, 'work', 'plate_assign.json')
FIX = {}                                                     # 묶음 번호 → 일본어 역 이름(검수)


def recs(pac):
    out = []
    for o in range(T0, T1, 16):
        fl, u, v, x, y, w, h, tp, cl = struct.unpack_from('<H2B2h4H', pac, o)
        out.append(dict(o=o, fl=fl, u=u, v=v, x=x, y=y, w=w, h=h, tp=tp, cl=cl))
    return out


def groups(pac):
    gs, cur = [], []
    for r in recs(pac):
        cur.append(r)
        if r['fl'] & 0x80:
            gs.append(cur); cur = []
    return gs


def names_dict():
    d = {}
    for f in ('stations.tsv', 'stn_board_kr.tsv'):
        for l in open(os.path.join(ROOT, 'work', 'text', f), encoding='utf-8').read().split('\n')[1:]:
            a = l.split('\t')
            if len(a) >= 2 and a[0]:
                d[a[0]] = a[1]
    return d


_F = {}


def rend(ch, w):
    k = (ch, w)
    if k not in _F:
        f = ImageFont.truetype(r'C:\Windows\Fonts\msgothic.ttc', 12)
        im = Image.new('L', (12, 12)); ImageDraw.Draw(im).text((0, 0), ch, font=f, fill=255)
        a = np.array(im, float)
        if w != 12:
            a = np.array(Image.fromarray(a.astype(np.uint8)).resize((w, 12), Image.BILINEAR), float)
        _F[k] = a
    return _F[k]


def cell(px, u, v, w):
    return (px[v:v + 12, u:u + w] > 0).astype(float)


def read():
    pac = open(PAC, 'rb').read(); D = names_dict(); gs = groups(pac)
    PX = [LK.load(pac, off) for off in LK.COPIES]
    chars = sorted({c for n in D for c in n})
    res = []
    for gi, g in enumerate(gs):
        best = (-9, None, None)
        for ci, px in enumerate(PX):
            cells = []
            for r in g:
                for k in range(max(1, r['h'] // 12)):
                    cells.append(cell(px, r['u'], r['v'] + 12 * k, r['w']))
            if not all(c.any() for c in cells):
                continue
            for nm in D:
                if len(nm) != len(cells):
                    continue
                s = np.mean([np.corrcoef(c.ravel(), rend(ch, c.shape[1]).ravel())[0, 1] for c, ch in zip(cells, nm)])
                if s > best[0]:
                    best = (s, ci, nm)
        res.append(dict(g=gi, o=g[0]['o'], n=len(g), copy=best[1], name=best[2], score=round(float(best[0]), 2)))
        print(gi, hex(g[0]['o']), len(g), best[1], best[2], round(float(best[0]), 2))
    json.dump(res, open(ASSIGN, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)


LIST0, LIST1 = 0x80000 + 0x4A54, 0x80000 + 0x4AAE              # 줄 목록(스프라이트 칸 번호, 판 묶음 = 칸 - 4)
LCOPY = [4, 3, 2, 2, 2, 1, 1, 1, 0]                          # 줄 → 로딩 그림 사본(loading_kr.COPIES 번호)
JP = ['秋田 大曲 角館 田沢湖 雫石 盛岡 新花巻',                # 줄별 역(판독, 눈 검수 2026-10-19)
      '直江津 犀潟 くびき うらがわら 虫川大杉 ほくほく大島 まつだい 十日町 六日町 塩沢 大沢 石打 越後湯沢',
      '渋谷 恵比寿 目黒 五反田 大崎 品川 田町 浜松町 新橋 有楽町 東京',
      '品川 田町 浜松町 新橋 有楽町 東京 神田 秋葉原 御徒町 上野',
      '品川 大井町 大森 蒲田 川崎 鶴見 新子安 東神奈川 横浜',
      '亀岡 嵯峨 二条 京都',
      '京都 西大路 向日町 長岡京 山崎 高槻 茨木 新大阪 大阪',
      '大阪 西九条 弁天町 新今宮 天王寺',
      '博多 吉塚 香椎 古賀 福間 東郷 赤間 折尾 黒崎 八幡 枝光 戸畑 小倉']
SPAN = 80                                                    # 판 최대 높이(원본 2자 이름 y0·68 + 12)
INK = 1                                                      # CLUT 3EDB: 1 = 검정 잉크, 2~6 = 반투명 회색(원본 가장자리)


def lines(pac):
    out, cur = [], []
    for b in pac[LIST0:LIST1]:
        if b == 0xFF:
            out.append(cur); cur = []
        else:
            cur.append(b - 4)
    return out


def assign(pac):
    """묶음 번호 → (사본, 일본어, 한국어)"""
    D = names_dict(); out = {}
    for ln, ci, jp in zip(lines(pac), LCOPY, JP):
        jp = jp.split(); assert len(jp) == len(ln), (ln, jp)
        for gi, n in zip(ln, jp):
            assert out.get(gi, (ci, n))[:2] == (ci, n), (gi, n)
            out[gi] = (ci, n, D[n])
    return out


def layout(g, n, cw, cap=12):
    """레코드별 (첫 음절, 음절 수, 표시 y), 음절 간격, 글자 칸 높이, 글꼴 크기 — 레코드 수는 고정.
    조각 하나는 칸 띠(64줄) 안에 들어가야 함 → 넘치면 갈무리9·간격 9(호쿠호쿠오시마)"""
    size = 11 if cw == 12 else 9; r = len(g); ch = 12
    span = max(x['y'] + x['h'] for x in g)
    if n <= r:                                               # 레코드마다 한 음절, 원본처럼 판 높이에 고르게
        ys = [0] if n == 1 else [round(i * (span - 12) / (n - 1)) for i in range(n)]
        out = [(i, 1, ys[i]) for i in range(n)]
        return out + [out[-1]] * (r - n), 12, ch, size
    k_max = -(-n // r)                                       # 간격 = 판 높이에 고르게, 단 cap(VRAM 자리) 이하·조각 ≤ 64줄
    p = max(min(cap, (span - ch) // (n - 1), (64 - ch) // max(k_max - 1, 1)), min(size + 1, (SPAN - ch) // (n - 1)))
    if (k_max - 1) * p + ch > 64:
        size = ch = p = 9
    sizes = [n // r + (1 if k < n % r else 0) for k in range(r)]
    out, i = [], 0
    for k in sizes:
        out.append((i, k, i * p)); i += k
    return out, p, ch, size


def free_mask():
    m = np.zeros((256, 256), bool)
    m[192:250, :] = True; m[250:256, :196] = True            # u196~ v250~ 작은 그림은 보존
    return m


def make(pac):
    """사본별 새 그림, 레코드 고침 {표 위치: 16 B}"""
    A = assign(pac); gs = groups(pac); recs_new = {}; pxs = {}
    for ci, off in enumerate(LK.COPIES):
        mine = sorted(gi for gi, a in A.items() if a[0] == ci)
        if not mine:
            continue
        cw = 10 if ci in (0, 4) else 12
        for cap in range(34, 9, -1):                         # 자리 되는 만큼 넓게 벌림
            r = place(pac, gs, A, mine, ci, off, cw, cap)
            if r:
                break
        else:
            raise SystemExit('판 글자 자리 부족: 사본 %d' % ci)
        px, plan, strips, cols = r
        for gi, ko, lay, pk in plan:
            for rec, (s0, k, y) in zip(gs[gi], lay):
                u, v = strips[(ko[s0:s0 + k], pk)]
                b = bytearray(pac[rec['o']:rec['o'] + 16])
                struct.pack_into('<2B2h2H', b, 2, u, v, rec['x'], y, cw, (k - 1) * pk[0] + pk[1])
                recs_new[rec['o']] = bytes(b)
        pxs[off] = px
        print('사본 %d: 역 %d · 조각 %d · 쓴 열 %d · 간격 상한 %d' % (ci, len(mine), len(strips), len(cols), cap))
    return pxs, recs_new


def place(pac, gs, A, mine, ci, off, cw, cap):
    """간격 상한 cap 으로 사본 한 장 배치·그리기 — 자리 모자라면 None"""
    px = LK.load(pac, off); fm = free_mask(); px[fm] = 0
    strips = {}                                         # (음절들, 간격) → 위치
    plan = []
    for gi in mine:
        ko = A[gi][2]; lay, p, ch, size = layout(gs[gi], len(ko), cw, cap)
        plan.append((gi, ko, lay, (p, ch, size)))
        for s0, k, _ in lay:
            strips.setdefault((ko[s0:s0 + k], (p, ch, size)), None)
    cols = {}                                           # 열 u → 다음 빈 v
    for key in sorted(strips, key=lambda t: -len(t[0])):
        syl, (p, ch, size) = key; h = (len(syl) - 1) * p + ch
        for u in range(0, 256 - cw + 1, cw):
            v = cols.get(u, 192); bot = 250 if u + cw > 196 else 256
            if v + h <= bot:
                break
        else:
            return None
        cols[u] = v + h; strips[key] = (u, v)
        for j, c in enumerate(syl):
            gl = PF.cell(c, size) > 0; gy = v + j * p + (ch - gl.shape[0]) // 2; gx = u + (cw - gl.shape[1]) // 2
            px[gy:gy + gl.shape[0], gx:gx + gl.shape[1]][gl] = INK
    return px, plan, strips, cols


def apply(pac):
    pxs, recs_new = make(bytes(pac))
    for off, px in pxs.items():
        enc = LK.encode(px); pac[off + 0x800:off + 0x800 + len(enc)] = enc
    for o, b in recs_new.items():
        pac[o:o + 16] = b
    return len(recs_new)


def plates(pac, px_of, recs_of=None):
    """줄별 판 그림(회색 판 + 검정 글자) — 미리보기"""
    gs = groups(pac); rows = []
    for ln, ci in zip(lines(pac), LCOPY):
        px = px_of(ci); row = np.full((SPAN + 8, 4 + 18 * len(ln)), 40, np.uint8)
        for k, gi in enumerate(ln):
            g = [recs_of(r) if recs_of else r for r in gs[gi]]
            top = min(r['y'] for r in g); bot = max(r['y'] + r['h'] for r in g)
            x0 = 4 + 18 * k; y0 = 4 + (SPAN - (bot - top)) // 2
            row[y0 - 2:y0 + bot - top + 2, x0 - 1:x0 + 14] = 150
            for r in g:
                c = px[r['v']:r['v'] + r['h'], r['u']:r['u'] + r['w']]
                sub = row[y0 + r['y'] - top:y0 + r['y'] - top + r['h'], x0:x0 + r['w']]
                sub[c == 1] = 0; sub[(c >= 2) & (c <= 6)] = 90
        rows.append(row)
    W = max(r.shape[1] for r in rows)
    return np.concatenate([np.pad(r, ((0, 0), (0, W - r.shape[1])), constant_values=255) for r in rows], 0)


def preview():
    pac = open(PAC, 'rb').read(); pxs, recs_new = make(pac)
    org = {ci: LK.load(pac, o) for ci, o in enumerate(LK.COPIES)}
    a = plates(pac, lambda ci: org[ci])
    def rec(r):
        b = recs_new.get(r['o'])
        if not b:
            return r
        fl, u, v, x, y, w, h, tp, cl = struct.unpack_from('<H2B2h4H', b)
        return dict(r, u=u, v=v, y=y, w=w, h=h)
    b = plates(pac, lambda ci: pxs.get(LK.COPIES[ci], org[ci]), rec)
    sh = np.concatenate([a, np.full((a.shape[0], 8), 255, np.uint8), b], 1)           # 왼쪽 원본 | 오른쪽 한글
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_plate.png')
    Image.fromarray(sh).resize((sh.shape[1] * 3, sh.shape[0] * 3), Image.NEAREST).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if sys.argv[1:] == ['read']:
        read()
    else:
        preview()
