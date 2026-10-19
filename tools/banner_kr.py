# -*- coding: utf-8 -*-
r"""주행 중 노란 띠 큰 역 이름(«品川역 출발 진행») 한글화 (2026-10-19)
  그림 = 노선 TXS 사각형 2 (VRAM 0,464 256×48 halfword) 의 u0~255 × v0~47 = 역 이름 16px 3줄(4bpp).
    호쿠호쿠·쿄토·큐슈는 4비트를 2비트 두 층(L = 아래 2비트, H = 위 2비트)으로 나눠 이름 세트 두 개를 겹쳐 담음 — CLUT 이 층을 고름.
    CLUT 은 같은 사각형 안(halfword x≥64, 행 464~511)에 있음. 색인(층 안 2비트): 1 = 글자(노랑/초록), 2 = 모서리, 3 = 어두운 테두리.
  표시 = SPRT(E1 tpage 0x10) 레코드 12 B [u v w h x s16 y s16 tpage clut] — 노선 모듈(섹터 지도 2~8·21번) 안 표 + 실행 파일.
    다음 낱말(«역 출발 진행», HUD 조각)은 이름 폭 w 뒤에 이어 붙음 → 폭을 바꿔도 됨.
  한글: (그룹, 층)마다 이름을 갈무리11 Bold 로 12줄 높이 4줄에 다시 깔고, 레코드 u·v·w·h(12)·y(+2, 16줄 가운데) 고침.
    지우는 곳 = 원래 이름 칸만(그 층 비트). 칸 밖 작은 그림·CLUT 은 그대로.
  python tools/banner_kr.py → work/gfx/prev_banner.png (원본 | 한글, 그룹·층별)
"""
import json
import os
import struct
import sys
from functools import lru_cache

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pixfont as PF  # noqa: E402
import plate_kr as PK  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
EXE = os.path.join(ROOT, 'work', 'disc', 'SLPM_863.78')
RECT_OFF = 512 * 512 * 2 + 64 * 512 * 2                    # TXS 머리 0x800 뒤 사각형 0·1 다음
GROUPS = {                                                 # 이름 영역(u0~255)이 같은 사본끼리
    'akita': [0x145000, 0x21A000], 'akita_l': [0x2EF800], 'hoku': [0x3DA000],
    'keihin': [0x539800, 0x5D0800, 0x667800, 0x80B800, 0x8A2800, 0xB02800, 0xBF4000],
    'kyoto': [0xA17800, 0xCE9000], 'yama': [0xEEF800], 'kyushu': [0xFC8000], 'osaka': [0x125C000]}
NAMES = {                                                  # (그룹, 층) → 줄별 역 이름(띠 판독 2026-10-19)
    ('akita', 'L'): '秋田 四ッ小屋 和田 大張野 羽後境 大曲|峰吉川 刈和野 神宮寺 盛岡 新花巻',
    ('akita_l', 'L'): '北大曲 羽後四ツ屋 鑓見内 羽後長野|鶯野 角館 生田 神代 刺巻 田沢湖 赤渕|春木場 雫石 小岩井 大釜 盛岡',
    ('hoku', 'L'): '塩沢 上越国際スキー場前 大沢 石打 越後湯沢|六日町',
    ('hoku', 'H'): '直江津 黒井 犀潟 くびき 大池いこいの森|うらがわら 虫川大杉 ほくほく大島|まつだい 十日町 しんざ 美佐島 魚沼丘陵',
    ('keihin', 'L'): '横浜 東神奈川 新子安 鶴見 川崎 蒲田|大森 大井町 神田 秋葉原 御徒町 上野|品川 田町 浜松町 新橋 有楽町 東京',
    ('kyoto', 'L'): '吹田 東淀川 新大阪 西九条 弁天町 大正|芦原橋 今宮 新今宮 天王寺 梅小路',
    ('kyoto', 'H'): '亀岡 馬堀 保津峡 嵯峨嵐山 太秦 花園|二条 丹波口 京都 西大路 向日町 長岡京|山崎 高槻 摂津富田 茨木 千里丘 岸辺',
    ('yama', 'L'): '東京 有楽町 新橋 浜松町 田町 品川 大崎|五反田 目黒 恵比寿 渋谷 原宿 代々木|新宿',
    ('kyushu', 'L'): '八幡 枝光 戸畑 九州工大前 西小倉 小倉|門司 小森江 門司港',
    ('kyushu', 'H'): '博多 吉塚 箱崎 香椎 九産大前 筑前新宮|古賀 千鳥 福間 東福間 東郷 赤間|教育大前 海老津 遠賀川 水巻 折尾 黒崎',
    ('osaka', 'L'): '天王寺 寺田町 桃谷 鶴橋 玉造 森ノ宮|大阪城公園 京橋 桜ノ宮 天満 大阪 福島|野田 西九条 弁天町 大正 芦原橋 新今宮',
}
EXTRA_KO = {'四ッ小屋': '요츠코야', '原宿': '하라주쿠', '代々木': '요요기', '新宿': '신주쿠', '千鳥': '치도리',
            '上越国際スキー場前': '조에츠코쿠사이스키조마에', '今宮': '이마미야', '梅小路': '우메코지'}
SMALL = set('ッツヶ')
FIXW = {('kyoto', 'L', 0): [32, 48, 48, 48, 48, 32]}                # 칸 계산이 다른 그룹 레코드에 끌리는 줄(東淀川 56 아님)
OVERLAP = {('kyoto', 'H'): {(112, 0, 32): '嵯峨'}, ('kyoto', 'L'): {(96, 0, 32): '大阪'}, ('akita_l', 'L'): {(16, 0, 32): '大曲'}}   # 긴 이름 칸 일부만 쓰는 레코드(嵯峨嵐山→嵯峨, 北大曲→大曲)
LINE_H, LINES = 12, 4


def load(pac, off):
    a = off + 0x800 + RECT_OFF
    raw = np.frombuffer(pac[a:a + 256 * 48 * 2], np.uint8).reshape(48, 512)
    return np.stack([raw & 15, raw >> 4], -1).reshape(48, 1024).copy()


def store(pac, off, px):
    p = px.reshape(48, 512, 2); a = off + 0x800 + RECT_OFF
    pac[a:a + 256 * 48 * 2] = (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def palette(pac, off, clut):
    """띠 안 CLUT(halfword x = (clut&63)*16, 행 = (clut>>6)-464) → 16 u16"""
    a = off + 0x800 + RECT_OFF
    hw = np.frombuffer(pac[a:a + 256 * 48 * 2], '<u2').reshape(48, 256)
    x = (clut & 63) * 16; y = (clut >> 6) - 464
    return hw[y, x:x + 16] if 0 <= y < 48 and x + 16 <= 256 else np.zeros(16, '<u2')


def layer_of(pal):
    out = set()
    if any(pal[k] for k in (1, 2, 3)):
        out.add('L')
    if any(pal[k] for k in (4, 8, 12)):
        out.add('H')
    return out


def records():
    """띠를 가리키는 레코드 (출처, 위치, u, v-208, w, clut) — PAC 노선 모듈·실행 파일"""
    out = []
    for src, path in (('pac', PAC), ('exe', EXE)):
        d = open(path, 'rb').read(); b = np.frombuffer(d[:len(d) // 2 * 2], np.uint8)
        for o in np.nonzero((b[8:-4] == 0x10) & ((b[9:-3] == 0x80) | (b[9:-3] == 0)))[0]:
            o = int(o)
            u, v, w, h, x, y, tp, cl = struct.unpack_from('<4B2h2H', d, o)
            if h == 16 and v in (208, 224, 240) and w % 8 == 0 and 8 <= w <= 128 and u + w <= 256 and x == 0 and y == 0 and (cl >> 6) >= 464:
                out.append((src, o, u, v - 208, w, cl))
    return out


def est(n):
    return sum(8 if c in SMALL else (12 if '぀' <= c <= 'ヿ' else 16) for c in n)


def slots(RS):
    """(그룹, 층) → {(u, v, w): 이름} — 줄마다 u0 부터 이어지는 칸 사슬 중 레코드에 있고 예상 폭에 가까운 것"""
    out = {}
    for key, s in NAMES.items():
        sl = {}
        for r, line in enumerate(s.split('|')):
            names = line.split()

            @lru_cache(None)
            def best(i, u):
                if i == len(names):
                    return (0, ())
                res = (1e9, ())
                for w in range(8, 129, 8):
                    if u + w > 256:
                        break
                    pen = abs(w - est(names[i])) + (0 if (u, 16 * r, w) in RS else 40)
                    c, t = best(i + 1, u + w)
                    if c + pen < res[0]:
                        res = (c + pen, ((u, 16 * r, w),) + t)
                return res
            if (key[0], key[1], r) in FIXW:
                u = 0
                for w, n in zip(FIXW[(key[0], key[1], r)], names):
                    sl[(u, 16 * r, w)] = n; u += w
                continue
            for k, n in zip(best(0, 0)[1], names):
                sl[k] = n
        sl.update(OVERLAP.get(key, {}))
        out[key] = sl
    return out


def assign(pac):
    """레코드 → (그룹, 층, 이름). 같은 표(12 B 연속)는 한 노선·한 층 → 후보 교집합이 빌 때까지 한 구간.
    CLUT 이 그 그룹 띠 안에 있으면 그 CLUT 이 읽는 층만 허용(띠 밖 CLUT 은 제한 없음)"""
    R = sorted(records(), key=lambda r: (r[0], r[1])); RS = {(u, v, w) for _, _, u, v, w, _ in R}; SL = slots(RS)
    def cands(r):
        _, _, u, v, w, cl = r; out = set()
        for (g, ly), sl in SL.items():
            if (u, v, w) not in sl:
                continue
            lays = layer_of(palette(pac, GROUPS[g][0], cl))
            if not lays or ly in lays:
                out.add((g, ly))
        return out
    segs, cur, cs = [], [], None
    for r in R:
        c = cands(r)
        adj = cur and cur[-1][0] == r[0] and r[1] - cur[-1][1] <= 12 * 4
        if adj and cs is not None and cs & c:
            cur.append(r); cs &= c
        else:
            if cur:
                segs.append((cur, cs))
            cur, cs = [r], c
    segs.append((cur, cs))
    res, bad = [], []
    badrec = [r for rs, c in segs if not c for r in rs]
    def near_bad(rs):                                       # 표 안에 어느 띠에도 안 맞는 레코드가 섞이면(빈 띠 81D1800 노선 추정) 그 표는 손대지 않음
        return any(b[0] == r[0] and abs(b[1] - r[1]) <= 12 * 4 for r in rs for b in badrec)
    for rs, c in segs:
        if len(c) == 1 and near_bad(rs):
            bad.append((rs, c)); continue
        if len(c) == 1:
            g, ly = next(iter(c))
            res += [(r, g, ly, SL[(g, ly)][r[2:5]]) for r in rs]
        else:
            bad.append((rs, c))
    return res, bad, SL


def glyph10(ch):
    """갈무리11 Bold 한 글자에서 «위아래 줄과 똑같은 줄» 하나를 빼 10행으로(가운데에 가까운 것) — 위아래 테두리까지 12행(사용자: 위 잘림 → 살짝 줄이기)"""
    g = PF.cell(ch, 11, True) > 0
    h = g.shape[0]
    if h <= 10:
        return g
    dup = [r for r in range(h - 1) if (g[r] == g[r + 1]).all()]
    r = min(dup, key=lambda r: abs(r - h / 2)) if dup else int(np.argmin(g.sum(1)))
    return np.delete(g, r, 0)


def word(text):
    """10행 글자(1) + 둘레 1px(3), 높이 12"""
    cols = []
    for ch in text:
        if ch == ' ':
            cols.append(np.zeros((10, 3), bool)); continue
        cols.append(glyph10(ch)); cols.append(np.zeros((10, 1), bool))
    g = np.concatenate(cols[:-1], 1)
    xs = np.nonzero(g.any(0))[0]; g = g[:, :xs.max() + 1]
    h, w = g.shape
    ink = np.zeros((LINE_H, w + 2), bool); ink[1:1 + h, 1:1 + w] = g
    pd = np.pad(ink, 1); ring = np.zeros_like(pd)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ring |= np.roll(np.roll(pd, dy, 0), dx, 1)
    out = np.zeros(ink.shape, np.uint8); out[ring[1:-1, 1:-1]] = 3; out[ink] = 1
    return out


def make(pac):
    """→ {그룹: 새 띠(48×1024 4bpp 색인)}, {(출처, 위치): (u, v, w, h, y)}"""
    res, bad, SL = assign(pac)
    assert not bad, bad
    D = PK.names_dict(); D.update(EXTRA_KO)
    used = {}
    for r, g, ly, n in res:
        used.setdefault((g, ly), set()).add(n)
    bands, place = {}, {}
    for g in GROUPS:
        px = load(pac, GROUPS[g][0])
        for ly in 'LH':
            if (g, ly) not in SL:
                continue
            sh = 0 if ly == 'L' else 2
            lv = (px[:, :256] >> sh) & 3
            free = lv == 0
            for (u, v, w), n in SL[(g, ly)].items():                 # 이름이 있던 줄은 그 층 통째(칸 계산 밖 한자 끝 찌꺼기 — 梅小路)
                free[v:v + 16, :] = True
            sub = px[:, :256]
            keep = np.uint8(15 ^ (3 << sh))
            sub[free] = sub[free] & keep                                # 이름 칸·빈 곳 그 층 비움
            x = [0] * LINES
            for n in sorted(used.get((g, ly), ()), key=lambda n: (-word(D[n]).shape[1], n)):   # 결정적 순서(집합 순서는 실행마다 다름)
                wd = word(D[n]); w = wd.shape[1]
                for k in range(LINES):
                    v0 = k * LINE_H
                    while x[k] + w <= 256 and not free[v0:v0 + LINE_H, x[k]:x[k] + w].all():
                        x[k] += 1
                    if x[k] + w <= 256:
                        break
                else:
                    raise AssertionError('띠 자리 부족 %s %s %s' % (g, ly, n))
                u0 = x[k]; reg = sub[v0:v0 + LINE_H, u0:u0 + w]
                m = wd > 0; reg[m] = (reg[m] & keep) | (wd[m] << sh).astype(np.uint8)
                free[v0:v0 + LINE_H, u0:u0 + w] = False; x[k] += w
                place[(g, ly, n)] = (u0, v0, w)
        bands[g] = px
    recs = {}
    for r, g, ly, n in res:
        u, v, w = place[(g, ly, n)]
        recs[(r[0], r[1])] = (u, v + 208, w, LINE_H, 1)   # 글자 2~11행 = «역» 글자 윗줄과 맞춤
    return bands, recs, place


def apply(pac, exe):
    """pac(bytearray) 띠 그림·노선 모듈 레코드 고침 → 새 실행 파일(실행 파일 안 레코드)"""
    bands, recs, _ = make(bytes(pac)); exe = bytearray(exe)
    for g, px in bands.items():
        for off in GROUPS[g]:
            cur = load(bytes(pac), off); cur[:, :256] = px[:, :256]             # 사본마다 u256~(풍경·CLUT)은 그대로
            store(pac, off, cur)
    for (src, o), (u, v, w, h, y) in recs.items():
        d = pac if src == 'pac' else exe
        struct.pack_into('<4B', d, o, u, v, w, h); struct.pack_into('<h', d, o + 6, y)
    return bytes(exe), len(recs)


def preview():
    pac = open(PAC, 'rb').read(); bands, recs, place = make(pac)
    rows = []
    for g in GROUPS:
        a = load(pac, GROUPS[g][0])[:, :256]; b = bands[g][:, :256]
        for ly, sh in (('L', 0), ('H', 2)):
            if (g, ly) not in SLOTS_CACHE:
                continue
            pal = np.array([[20, 20, 60], [248, 248, 0], [144, 144, 0], [32, 32, 0]], np.uint8)
            rows.append(np.concatenate([pal[(a >> sh) & 3], np.full((48, 4, 3), 255, np.uint8), pal[(b >> sh) & 3]], 1))
            rows.append(np.full((4, 516, 3), 255, np.uint8))
    sh_ = np.concatenate(rows, 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_banner2.png')
    Image.fromarray(sh_).resize((sh_.shape[1] * 2, sh_.shape[0] * 2), Image.NEAREST).save(p); print('미리보기', p)


SLOTS_CACHE = NAMES


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    pac = open(PAC, 'rb').read()
    res, bad, SL = assign(pac)
    print('배정', len(res), '애매/없음 구간', len(bad))
    last = None
    for r, g, ly, n in res:
        if (g, ly) != last:
            print(); print(' %s %X %s/%s:' % (r[0], r[1], g, ly), end='')
            last = (g, ly)
        print('', n, end='')
    print()
    for rs, c in bad:
        print(' 애매', rs[0][0], hex(rs[0][1]), len(rs), [r[2:5] for r in rs][:6], c)
    D = PK.names_dict(); D.update(EXTRA_KO)
    print('사전에 없는 이름', sorted({n for sl in SL.values() for n in sl.values() if n not in D}))
    preview()
