# -*- coding: utf-8 -*-
r"""보너스 게임(연결 미니게임)·병합 안내·출발 점검 화면 그림 글자 한글화 (2026-10-19) — 스테이트 없음 → 원래 글자 칸 안 제자리
  묶음(GROUPS) = 기준 그림 좌표의 항목 목록 + 같은 그림이 들어 있는 곳(TXS, 사각형, 더할 x·y).
    BONUS = 9969800 사각형 1(576,0 64×160 = 4bpp 256×160) — 같은 그림이 사각형 0 의 u768~, 99B9800 두 곳, B8CB000 의 v+88(일부).
    HEIGO = B8CB000(384,256 64×256) あと·併合準備完了·併合完了·距離注意·東北.
    KENSA = B994800(384,0 64×256) 仕業検査·안내문 — 7E23800 사각형 0 의 (u+768, v+80) 에 v0~175 사본.
    KENSA7 = 7E23800 의 좁은 안내문(u700~767, 5줄).
  사본 쪽은 «항목 칸이 기준 원본과 똑같을 때만» 그림(잘린 사본·딴 그림 보호).
  항목 = dict(box=(u0,u1,v0,v1), t=한국어, f=(갈무리 크기, 굵게, 배율), fill=색인|'ramp', edge=둘레 색인(1px), sh=[(색인|'ramp', dx, dy)],
              bg='zero'|('const',i)|('repl',[색인들],i)|('period',표본 u), dots={바탕 색인: 획 색인}, al='c'|'l'|'r', ex=줄별 색 제외, fix={상자 열: 색인}, pitch=한글 간격)
    ramp = 원본 칸의 줄마다 최빈 색인(0·둘레·그림자·ex 제외) — 원본의 위→아래 무지개 줄 색을 그대로.
  python tools/bonus_kr.py → work/gfx/prev_bonus.png
"""
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pixfont as PF  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
G9X2, G9, G11B = (9, False, 2), (9, False, 1), (11, True, 1)   # 갈무리7 2배는 받침이 뭉개져 안 씀
BONUS = [
    dict(box=(4, 140, 9, 28), t='시간 안에 충격 없이', f=G11B, fill=6, sh=[(15, 1, 1)], bg=('repl', [6, 15], 1), al='l'),
    dict(box=(4, 140, 28, 47), t='열차를 연결하라', f=G11B, fill=6, sh=[(15, 1, 1)], bg=('repl', [6, 15], 1), al='l'),
    dict(box=(0, 160, 48, 72), t='보너스 게임', f=G9X2, fill='ramp', edge=15, ex={1, 2, 3, 4, 5},
         sh=[('ramp5', -1, 1), ('ramp5', -2, 2)], bg='zero'),
    dict(box=(0, 80, 73, 93), t='연결 성공', f=G9X2, fill='ramp', edge=8, bg='zero'),
    dict(box=(80, 162, 73, 93), t='연결 실패', f=G9X2, fill='ramp', edge=8, bg='zero'),
    dict(box=(184, 208, 2, 23), t='준', f=G9X2, fill='ramp', edge=8, bg='zero'),
    dict(box=(210, 235, 2, 23), t='비', f=G9X2, fill='ramp', edge=8, bg='zero'),
    dict(box=(184, 208, 26, 46), t='시', f=G9X2, fill='ramp', edge=8, bg='zero'),
    dict(box=(210, 235, 26, 46), t='작', f=G9X2, fill='ramp', edge=8, bg='zero'),
    dict(box=(166, 218, 54, 66), t='신에츠 본선', f=G9, fill=3, bg='zero', al='r'),
    dict(box=(6, 116, 106, 129), t='충격 미터', f=G9X2, fill=14, edge=10, bg=('period', 5), dots={2: 3}, fix={93: 2}),   # u99 = 칸막이 세로줄
    dict(box=(120, 166, 105, 115), t='합격 범위', f=G9, fill=11, bg=('period', 170), dots={2: 2}, al='l'),
    dict(box=(200, 232, 105, 115), t='위험', f=G9, fill=3, bg=('period', 170), dots={2: 2}),
]
HEIGO = [
    dict(box=(2, 26, 3, 16), t='남은', f=G9, fill=3, sh=[(1, 1, 1)], bg='zero'),
    dict(box=(130, 206, 2, 14), t='병합 준비 완료', f=G9, fill=0, sh=[(2, 1, 1)], bg=('const', 3)),
    dict(box=(178, 254, 18, 30), t='병합 완료', f=G9, fill=0, sh=[(2, 1, 1)], bg=('const', 3), pitch=14),
    dict(box=(186, 248, 34, 47), t='거리 주의', f=G9, fill=3, sh=[(1, 1, 1)], bg='zero', pitch=12),
    dict(box=(194, 228, 73, 86), t='토호쿠', f=G9, fill=3, sh=[(2, 1, 0)], bg='zero'),
]
KENSA = [
    dict(box=(166, 254, 63, 84), t='출발 점검', f=G9X2, fill='ramp', ex={1, 2}, sh=[(1, 1, 1)], bg='zero'),
    dict(box=(0, 148, 144, 161), t='각 레버를 시작 위치에', f=G11B, fill=4, sh=[(1, 1, 2)], bg='zero', al='l'),
] + [dict(box=(0, 60, y, y + 16), t=t, f=G11B, fill=4, sh=[(1, 1, 2)], bg='zero', al='l')
     for y, t in zip((162, 178, 194, 210, 226), ('각 레버를', '시작', '위치에', '맞춰', '주세요'))]
KENSA7 = [dict(box=(700, 768, y, y + 16), t=t, f=G11B, fill=4, sh=[(1, 1, 2)], bg='zero', al='l')
          for y, t in zip((80, 96, 112, 128, 144), ('각 레버를', '시작', '위치에', '맞춰', '주세요'))]
# (항목, 기준(TXS, 사각형), 사본들 [(TXS, 사각형, dx, dy)])
GROUPS = [(BONUS, (0x9969800, 1), [(0x9969800, 1, 0, 0), (0x9969800, 0, 768, 0), (0x99B9800, 1, 0, 0), (0x99B9800, 0, 768, 0),
                                   (0xB8CB000, 0, 0, 88)]),
          (HEIGO, (0xB8CB000, 0), [(0xB8CB000, 0, 0, 0)]),
          (KENSA, (0xB994800, 0), [(0xB994800, 0, 0, 0), (0x7E23800, 0, 768, 80)]),
          (KENSA7, (0x7E23800, 0), [(0x7E23800, 0, 0, 0)])]


def rect(pac, off, i):
    n = struct.unpack_from('<I', pac, off + 4)[0]; a = off + 0x800
    for k in range(n):
        x, y, w, h = struct.unpack_from('<4H', pac, off + 8 + 8 * k)
        if k == i:
            b = np.frombuffer(pac[a:a + w * h * 2], np.uint8).reshape(h, w * 2)
            return a, np.stack([b & 15, b >> 4], -1).reshape(h, w * 4).copy()
        a += w * h * 2


def encode(px):
    h, w = px.shape; p = px.reshape(h, w // 2, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


# 2026-10-19 사용자 «준비 위가 잘렸냐»: 갈무리9 «준» 은 받침 때문에 ㅈ 꼭지(세로 획)가 없어 2배에서 잘린 것처럼 보임 → ㅈ 꼭지 살리고 ㄴ 한 줄 줄인 손질 글자
FIX9 = {'준': ['.#######.', '....#....', '...#.#...', '.##...##.', '.........', '#########', '.#..#....', '.#.......', '.#######.']}


def glyph(ch, size, bold):
    if size == 9 and not bold and ch in FIX9:
        return np.array([[c == '#' for c in r] for r in FIX9[ch]], bool)
    return (PF.cell(ch, size, bold) if 0xAC00 <= ord(ch) <= 0xD7A3 else PF.text(ch, size, bold)) > 0


def render(t, font, w, pitch=None):
    size, bold, sc = font
    sp = {7: 3, 9: 3, 11: 4}[size]
    for gap in (1, 0):
        cols = []; x = 0
        for ch in t:
            if ch == ' ':
                x += sp * sc; continue
            g = np.kron(glyph(ch, size, bold), np.ones((sc, sc), bool))
            cols.append((x, g)); x += (pitch if pitch and ord(ch) >= 0xAC00 else g.shape[1] + gap * sc)
        total = cols[-1][0] + cols[-1][1].shape[1]
        if total <= w:
            break
    else:
        raise SystemExit('넘침 %r %d > %d' % (t, total, w))
    gh = max(g.shape[0] for _, g in cols); ink = np.zeros((gh, total), bool)
    for x, g in cols:
        ink[gh - g.shape[0]:, x:x + g.shape[1]] |= g
    return ink


def ramp(reg, skip):
    out = []
    for row in reg:
        v = row[~np.isin(row, list(skip))]
        out.append(int(np.bincount(v, minlength=16).argmax()) if len(v) else None)
    idx = [y for y, v in enumerate(out) if v is not None]
    full = [out[min(idx, key=lambda k: abs(k - y))] if idx else 0 for y in range(len(out))]
    v = reg[~np.isin(reg, list(skip))]                                   # 2026-10-19 사용자 «색 넣은 거 다 단색» — 줄별 무지개 대신 칸 전체 최빈 색 하나
    one = int(np.bincount(v, minlength=16).argmax()) if len(v) else full[0]
    return [one] * len(full)


def dil(m):
    p = np.pad(m, 1); r = p.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            r |= np.roll(np.roll(p, dy, 0), dx, 1)
    return r[1:-1, 1:-1]


def shift(m, dx, dy):
    o = np.zeros_like(m); h, w = m.shape
    o[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = m[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return o


def draw(px, it, ox=0, oy=0):
    u0, u1, v0, v1 = it['box']; u0 += ox; u1 += ox; v0 += oy; v1 += oy
    reg = px[v0:v1, u0:u1]; orig = reg.copy(); H, W = reg.shape
    edge = it.get('edge'); shs = it.get('sh', [])
    skip = {0} | ({edge} if edge is not None else set()) | {s[0] for s in shs if isinstance(s[0], int)} | it.get('ex', set())
    fr = ramp(orig, skip) if it['fill'] == 'ramp' else None
    sr = ramp(orig, set(range(16)) - {1, 2, 3, 4, 5}) if any(s[0] == 'ramp5' for s in shs) else None
    bg = it['bg']
    if bg == 'zero':
        base = np.zeros_like(orig)
    elif bg[0] == 'const':
        base = np.full_like(orig, bg[1])
    elif bg[0] == 'repl':
        base = orig.copy(); base[np.isin(base, bg[1])] = bg[2]
    else:                                                              # period: 같은 줄 표본 3칸(무늬 주기 3)을 이어 깔기
        sx = bg[1] + ox; smp = px[v0:v1, sx:sx + 3]
        base = np.stack([smp[:, (u0 + x - sx) % 3] for x in range(W)], 1)
    ink = render(it['t'], it['f'], W - (2 if edge is not None else 0) - max([abs(s[1]) for s in shs] or [0]), it.get('pitch'))
    m = np.zeros((H, W), bool); gh, gw = ink.shape
    al = it.get('al', 'c'); pad = 1 if edge is not None else 0
    x0 = pad if al == 'l' else (W - gw - pad if al == 'r' else (W - gw) // 2)
    y0 = max(0, (H - gh) // 2 - (1 if shs and any(s[2] > 0 for s in shs) else 0))
    assert y0 + gh <= H, ('높이 넘침', it['t'], gh, H)
    m[y0:y0 + gh, x0:x0 + gw] = ink
    ring = (dil(m) & ~m) if edge is not None else np.zeros_like(m)
    body = m | ring; out = base.copy()
    for s in shs:
        sm = shift(body, s[1], s[2]) & ~body
        for y in range(H):
            out[y][sm[y]] = sr[y] if s[0] == 'ramp5' else s[0]
    out[ring] = edge if edge is not None else 0
    for y in range(H):
        f = fr[y] if fr else it['fill']
        row = m[y]; out[y][row] = f
        for b, i in it.get('dots', {}).items():
            out[y][row & (base[y] == b)] = i
    for c, i in it.get('fix', {}).items():                         # 상자 기준 열 → 고정 색인(칸막이 줄 등, 글자 위에도)
        out[:, c][~body[:, c]] = i
    reg[:] = out


def build(pac):
    """→ {(TXS, 사각형): (주소, 원본, 새것)}"""
    res = {}
    for items, (sref, iref), targets in GROUPS:
        _, src = rect(pac, sref, iref)
        for off, ri, dx, dy in targets:
            if (off, ri) not in res:
                a, px = rect(pac, off, ri); res[(off, ri)] = (a, px.copy(), px)
            a, orig, px = res[(off, ri)]
            for it in items:
                u0, u1, v0, v1 = it['box']; su = it['bg'][1] if it['bg'][0] == 'period' else None
                if v1 + dy > px.shape[0] or u1 + dx > px.shape[1]:
                    continue
                cu0 = min(u0, su) if su is not None else u0; cu1 = max(u1, su + 3) if su is not None else u1
                if not (orig[v0 + dy:v1 + dy, cu0 + dx:cu1 + dx] == src[v0:v1, cu0:cu1]).all():
                    continue
                draw(px, it, dx, dy)
    return res


def apply(pac):
    res = build(bytes(pac))
    for a, _, px in res.values():
        enc = encode(px); pac[a:a + len(enc)] = enc
    return len(res)


def preview():
    pac = open(PAC, 'rb').read(); res = build(pac)
    rng = np.random.RandomState(3); pal = rng.randint(40, 255, (16, 3)).astype(np.uint8)
    pal[0] = 0; pal[1] = [30, 30, 90]; pal[15] = [255, 255, 255]
    views = [((0x9969800, 1), (0, 256, 0, 130)), ((0xB8CB000, 0), (0, 256, 0, 184)), ((0xB994800, 0), (0, 256, 60, 245)),
             ((0x7E23800, 0), (700, 1024, 76, 245))]
    rows = []
    for k, (u0, u1, v0, v1) in views:
        _, o, n = res[k]
        r = np.concatenate([pal[o[v0:v1, u0:u1]], np.full((v1 - v0, 4, 3), 255, np.uint8), pal[n[v0:v1, u0:u1]]], 1)
        rows.append(np.pad(r, ((0, 6), (0, 652 - r.shape[1]), (0, 0)), constant_values=90))
    sh = np.concatenate(rows, 0)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_bonus.png')
    Image.fromarray(sh).resize((sh.shape[1] * 2, sh.shape[0] * 2), Image.NEAREST).save(p); print('미리보기', p)
    for (off, ri), (a, o, n) in res.items():
        print('%X#%d 바뀐 화소 %d' % (off, ri, int((o != n).sum())))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
