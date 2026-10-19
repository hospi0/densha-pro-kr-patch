# -*- coding: utf-8 -*-
r"""HUD 아틀라스 낱말 조각 제자리 한글화 (2026-10-19) — hud_kr(위 줄)의 일반판
  HUD 문구 = 아틀라스 조각 이어 붙이기(出発+進+行). 일본어·한국어 어순이 같아 조각마다 옮김(한자어 = 음절 하나씩).
  조각 표 work/text/hud_pieces_kr.tsv (id = work/hud_pieces.json 번호, 정렬 l/c/r, 앞 공백 = 왼쪽 4px 띄움, \n = 여러 줄).
  색 구조 자동: 칸 테두리 최빈 단계 = 바탕(0 이면 투명), 나머지 최빈 = 글자, 다음 = 테두리(획 둘레 1px).
  층(H/L)은 그 조각 CLUT 가 보는 2비트만 다시 그림.
  글꼴: 한 줄 높이 ≥ 11 = 나눔고딕 ExtraBold, 작으면 갈무리 / 회색 여러 줄 문장 = 갈무리11(12px 줄).
  python tools/hud_words.py → work/gfx/prev_hud_words_N.png (문구 조합 원본|한글, 2배)
"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hud as H  # noqa: E402
import hud_kr as HK  # noqa: E402
import pixfont as PF  # noqa: E402

TSV = os.path.join(ROOT, 'work', 'text', 'hud_pieces_kr.tsv')


def table():
    out = {}
    for l in open(TSV, encoding='utf-8').read().split('\n')[1:]:
        if l:
            a = (l.split('\t') + [''] * 5)[:5]
            out[int(a[0])] = (a[2].replace('\\n', '\n'), a[3] or 'c')
    return out


def pieces():
    return {p[0]: p for p in json.load(open(os.path.join(ROOT, 'work', 'hud_pieces.json')))}


def levels_of(px, u, v, w, h, lay):
    a = px[v:v + h, u:u + w]
    return (a >> 2) if lay == 'H' else (a & 3)


def style(L):
    border = np.concatenate([L[0], L[-1], L[:, 0], L[:, -1]])
    bc = np.bincount(border, minlength=4)
    bg = 0 if bc[0] >= 0.2 * len(border) else int(bc.argmax())   # 글자 둘레선이 칸 가장자리에 닿아도 바탕으로 착각 안 하게
    cnt = np.bincount(L.ravel(), minlength=4).astype(float); cnt[bg] = -1
    order = [int(i) for i in np.argsort(-cnt) if cnt[i] > 0]
    cand = [k for k in order if cnt[k] > 0.15 * cnt[order[0]]][:2] if order else [3]
    def inner(k):                                               # 획(글자) = 바탕에 덜 닿는 단계, 둘레 = 바탕에 닿는 단계 — 개수로 고르면 둘레가 더 많아 뒤집힘
        m = L == k; nb = L == bg
        touch = m & (np.roll(nb, 1, 0) | np.roll(nb, -1, 0) | np.roll(nb, 1, 1) | np.roll(nb, -1, 1))
        return 1 - touch.sum() / max(1, m.sum())
    cand.sort(key=inner, reverse=True)
    fill = cand[0]; edge = cand[1] if len(cand) > 1 else None
    return bg, fill, edge


def render_line(text, hmax, wmax):
    """한 줄 → 덮임 배열. 높이 hmax 안, 폭 wmax 넘으면 70% 까지 좁히고 그래도 넘치면 높이를 줄임"""
    t = text.strip()
    if not t:
        return np.zeros((1, 1))
    h = hmax
    while h >= 6:
        if h >= 10:
            try:
                f = HK.cover(t, h, int(wmax / 0.7))
                c = f
            except AssertionError:
                h -= 1; continue
        else:
            c = PF.text(t, 7 if h < 9 else 9)
        if c.shape[1] > wmax:
            if c.shape[1] * 0.7 <= wmax:
                c = np.array(Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)).resize((wmax, c.shape[0]), Image.BOX)) / 255.0
            else:
                h -= 1; continue
        return c
    raise AssertionError('안 들어감 %r' % text)


def draw_piece(px, p, text, align, clut, spans=None, sty=None):
    """sty = (바탕, 글자, 둘레) — 조각 «전체» 원본에서 한 번만 정해 칸마다 같게(칸마다 따로 추정하면 음절 색이 달라짐)"""
    _, u, v, w, h, lay = p[:6]
    L = levels_of(px, u, v, w, h, lay)
    bg, fill, edge = sty or style(L)
    if spans:                                                   # 칸 나눠 그리기(그린 범위 밖은 원래 화소 그대로)
        for x0, x1, t, al in spans:
            q = list(p); q[1] = u + x0; q[3] = x1 - x0
            if t.strip():
                draw_piece(px, q, t, al, clut, None, (bg, fill, edge))
        return
    if '\n' not in text:                                        # 한 줄 = 갈무리11 Bold 비트맵(hud_px B안, 사용자 선택) — 축소·중간 단계 없음
        import hud_px as HP
        lv = HP.word(text, w, h, 'B' if h > 14 else 'S', align)  # 0 바탕 · 1 둘레 · 3 획 (높이 14 이하 = 주황 메뉴 글자, 갈무리9 로 고르게)
        new = np.full((h, w), bg, np.uint8)
        if edge is not None:
            new[lv == 1] = edge
        new[lv == 3] = fill
        reg = px[v:v + h, u:u + w]
        keep = (reg & 3) if lay == 'H' else (reg & 12)
        reg[:] = keep | ((new << 2) if lay == 'H' else new)
        return
    new = np.full((h, w), bg, np.uint8)
    multi = '\n' in text
    if multi:                                                    # 회색 문장: 원본 줄 위치(잉크 줄 띠)마다 한 줄, 갈무리(띠 높이에 맞춤)
        ink_rows = np.nonzero((L != bg).any(1))[0]
        bands = []
        for r in ink_rows:
            if bands and r - bands[-1][1] <= 2:
                bands[-1][1] = r
            else:
                bands.append([r, r])
        lines = text.split('\n')
        assert len(lines) <= len(bands), '줄 수 %d > 원본 %d %r' % (len(lines), len(bands), text)
        x_left = int(np.nonzero((L != bg).any(0))[0].min())
        for ln, (r0, r1) in zip(lines, bands):
            bh = r1 - r0 + 1
            c = PF.text(ln, 11 if bh >= 11 else 9) if ln.strip() else np.zeros((1, 1))
            if c.shape[1] > w - x_left - 1:
                c = np.array(Image.fromarray((c * 255).astype(np.uint8)).resize((w - x_left - 1, c.shape[0]), Image.BOX)) / 255.0
            hh, ww = c.shape
            x0 = x_left if align == 'l' else (w - ww) // 2
            y0 = max(0, r0 + (bh - hh) // 2)
            reg = new[y0:y0 + hh, x0:x0 + ww]; reg[c[:reg.shape[0], :reg.shape[1]] >= 0.5] = fill
    else:
        pad = 4 if text.startswith(' ') else 0
        ring = 1 if edge is not None else 0
        c = render_line(text, h - 2 * ring - (1 if h >= 14 else 0), w - 2 * ring - pad)
        hh, ww = c.shape
        if align == 'l':
            x0 = ring + pad
        elif align == 'r':
            x0 = w - ring - ww
        else:
            x0 = (w - ww) // 2 + pad // 2
        y0 = (h - hh) // 2
        ink = np.zeros((h, w), bool); ink[y0:y0 + hh, x0:x0 + ww] = c >= 0.5
        part = np.zeros((h, w), bool); part[y0:y0 + hh, x0:x0 + ww] = (c >= 0.25) & (c < 0.5)
        if edge is not None:
            ring_m = np.zeros_like(ink)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ring_m |= np.roll(np.roll(ink, dy, 0), dx, 1)
            new[ring_m & ~ink] = edge
        else:
            new[part] = fill
        new[ink] = fill
    reg = px[v:v + h, u:u + w]
    keep = (reg & 3) if lay == 'H' else (reg & 12)
    reg[:] = keep | ((new << 2) if lay == 'H' else new)


# 겹치는 조각: 큰 조각은 제자리, 그 안의 작은 조각은 빈 자리로 옮겨 그리고 레코드 u·v 고침(실행 파일)
RELOC = {96, 101, 102, 103, 109, 367, 391}     # 큰 조각 안에서 뜻이 부딪치는 작은 조각만(ス=수/스, さてい, め=행/이, 止=막/지)
ONLY = {}
# (조각, 한국어, 맞춤, 그 레코드들만) — 같은 조각을 문구마다 다르게: 戻 = 귀(귀환할까요?) / 뒤(뒤로), ますか? = 까요?(중지|할|까요?)
EXTRA = [(401, '귀', 'r', ['3E8A4']), (404, '까요?', 'l', ['3E868'])]   # 3E8A4 = 始発駅に「戻」(x-12) · 3E880 = 「戻」る(x11)                # 옮길 레코드 범위(그 밖 레코드는 원래 조각 = 회색 메뉴 標準に戻す 의 戻)
# 큰 조각은 작은 조각 경계(조각 안 x 범위)에 맞춰 나눠 그림 → 그 안의 작은 조각도 맞게 보임
SPANS = {
    50: [(0, 32, '속도', 'c'), (32, 64, ' 제한', 'l')],
    95: [(0, 16, '마', 'c'), (16, 32, '이', 'c'), (32, 48, '너', 'c'), (48, 64, '스', 'c')],
    105: [(0, 16, '경', 'c'), (16, 64, '적 울려', 'l')],
    107: [(0, 16, '열', 'c'), (16, 32, '차', 'c')],
    110: [(0, 16, '이', 'c'), (16, 48, ' 충돌', 'l')],
    114: [(0, 16, '감', 'c'), (16, 32, '점', 'c')],
    120: [(0, 16, '합', 'c'), (16, 32, '격', 'c')],
    336: [(0, 36, '운전', 'c'), (36, 72, ' 재개', 'l')],
    362: [(0, 16, '속', 'c'), (16, 32, '도', 'c'), (32, 48, ' 초', 'c'), (48, 64, '과', 'c')],
    366: [(0, 16, '정', 'c'), (16, 32, '지', 'c'), (32, 64, ' 위치', 'l')],
    368: [(0, 88, '오버런', 'c')],                                  # 마지막 8px = ! (合格! 와 같이 씀) 그대로
    375: [(0, 16, '급', 'c'), (16, 32, '제', 'c'), (32, 48, '동', 'c')],
    384: [(0, 32, '운전', 'c'), (32, 80, ' 계속?', 'l')],
    390: [(0, 112, '알림등 확인 실수', ('c', 64))],             # 빨간 상자(390+ス = 128px) 가운데 · 確認 조각(391)은 옮김
    392: [(0, 24, '운전', 'c'), (24, 48, ' 재개', 'l')],
    403: [(0, 12, '환', 'l'), (12, 60, '할까요?', 'l')],
    99: [(0, 64, '평지', ('c', 20))],                               # 표지 기둥(x≈40~) 왼쪽
}
SKIP = {221, 149, 182, 213, 472, 490, 195, 220, 234, 240, 241, 249, 250, 461}   # 220·234·240·241·249·250 = 4비트 색 조합 글자(운전 평가·보너스·완통) — 따로 손볼 것       # 221 = 220(글자 부분)과 같은 자리 · 회색 안내문과 화소를 나눠 쓰는 문장은 안내문 단계로


def full_pieces(P):
    """CLUT 가 다른 층 비트도 읽는(4비트) 조각 — 속도계 원판 등. 이 영역은 두 층 모두 빈 자리 아님(9차 실기: 속도계 깨짐)"""
    pac = open(H.PAC, 'rb').read(); exe = open(H.EXE, 'rb').read()
    _, clut = H.load(pac)
    recs = {('%X' % r['o']): r for r in H.records(exe)}
    out = set()
    for i, p in P.items():
        for o in p[6]:
            pal = H.pal(clut, recs[o]['clut'])
            other = [1, 2, 3] if p[5] == 'H' else [4, 8, 12]
            if any(pal[k][3] > 0 for k in other):
                out.add(i); break
    return out


DIFF = H.copies_differ(open(H.PAC, 'rb').read())
DIFF[256:384, 192:256] = True                                   # 실행 중 게임이 덮어쓰는 칸(VRAM x368~383 y256~383) — 18차 실기: 재래선은 v256~319 만 덮임(시발역으로·세·출 깨짐), 19차 실기: 신칸센은 v383 까지(«세» v336 깨짐, work/shots/end19)


def free_map(px, P):
    """층마다 «어느 조각도 안 쓰고 원래 0» 인 화소 — 4비트 조각 영역은 두 층 다 제외"""
    out = {}
    full = full_pieces(P)
    for lay in 'HL':
        used = np.zeros(px.shape, bool)
        for i, p in P.items():
            if p[5] == lay or i in full:
                used[p[2]:p[2] + p[4], p[1]:p[1] + p[3]] = True
        lv = (px >> 2) if lay == 'H' else (px & 3)
        out[lay] = ~used & (lv == 0) & ~DIFF                        # 사본끼리 다른 자리(차종별 속도계) 제외
    return out


def find_free(free, w, h):
    ii = np.zeros((513, 257), int); ii[1:, 1:] = (~free).cumsum(0).cumsum(1)
    for v in range(0, 512 - h + 1):
        if (v < 256) != (v + h - 1 < 256):                     # tpage 경계(v256) 를 넘으면 안 됨
            continue
        for u in range(0, 256 - w + 1):
            if ii[v + h, u + w] - ii[v, u + w] - ii[v + h, u] + ii[v, u] == 0:
                return u, v
    raise AssertionError('빈 자리 없음 %dx%d' % (w, h))


def find_fit(free, w, h):
    """층 단위 옮김 조각 자리 = 들어가는 자리 중 «둘레가 막힌 화소가 가장 많은» 곳(큰 빈 자리를 아낌).
    2026-10-19: 실행 중 덮이는 칸을 빼자 첫 자리(find_free) 방식으론 48×14 메뉴 조각 자리가 남지 않음"""
    ii = np.zeros((513, 257), int); ii[1:, 1:] = (~free).cumsum(0).cumsum(1)
    def box(v0, u0, hh, ww):                                    # 모든 (v0, u0) 에 대해 [v0:v0+hh, u0:u0+ww] 안 막힌 화소 수
        return ii[v0 + hh, u0 + ww] - ii[v0, u0 + ww] - ii[v0 + hh, u0] + ii[v0, u0]
    V, U = np.mgrid[0:512 - h + 1, 0:256 - w + 1]
    ok = box(V, U, h, w) == 0
    ok &= (V < 256) == (V + h - 1 < 256)                       # tpage 경계(v256) 를 넘으면 안 됨
    if not ok.any():
        raise AssertionError('빈 자리 없음 %dx%d' % (w, h))
    pad = np.ones((514, 258), bool); pad[1:-1, 1:-1] = ~free      # 바깥 = 막힘
    jj = np.zeros((515, 259), int); jj[1:, 1:] = pad.cumsum(0).cumsum(1)
    ring = (jj[V + h + 2, U + w + 2] - jj[V, U + w + 2] - jj[V + h + 2, U] + jj[V, U])   # (u-1..u+w, v-1..v+h) 막힌 수 = 둘레만(안은 0)
    score = np.where(ok, ring, -1)
    v, u = np.unravel_index(np.argmax(score), score.shape)
    return int(u), int(v)


def lpair(clut):
    """H 층 CLUT → 같은 색을 L 층으로 읽는 CLUT(있으면) — H 빈 자리가 잘게 쪼개져 있어 L 로 옮기기"""
    allc = [((464 + y) << 6) | ((256 + x) >> 4) for y in range(48) for x in range(0, 64, 16)]
    def lv(c, lay):
        pl = H.pal(clut, c); idx = [0, 4, 8, 12] if lay == 'H' else [0, 1, 2, 3]
        return [tuple(int(t) for t in pl[k]) for k in idx]
    out = {}
    for c in allc:
        want = lv(c, 'H')
        for d in allc:
            if lv(d, 'L') == want:
                out[c] = d; break
    return out


def alloc(free, lay, offs, w, h, recs, LP):
    """빈 자리 잡기: 원래 층 → 안 되면 같은 색을 다른 층으로 읽는 CLUT 짝이 있으면 그 층. → (층, u, v, CLUT 바꿈)"""
    HPm = {d: c for c, d in LP.items()}
    tries = [(lay, {})]
    pair = LP if lay == 'H' else HPm
    if all(recs[o]['clut'] in pair for o in offs):
        tries.append(('L' if lay == 'H' else 'H', {recs[o]['clut']: pair[recs[o]['clut']] for o in offs}))
    if lay == 'H':
        tries.reverse()                                         # H 빈 자리는 잘게 쪼개져 있어 L 먼저
    for ly, cm in tries:
        try:
            u, v = find_fit(free[ly], w, h)
        except AssertionError:
            continue
        free[ly][v:v + h, u:u + w] = False
        return ly, u, v, cm
    raise AssertionError('빈 자리 없음 %dx%d (%s 남은 H %d L %d)' % (w, h, offs, free['H'].sum(), free['L'].sum()))


GRAD = {255, 336, 338}
WIDTH = {}                                     # (340·341·139 = 아틀라스를 회색으로 비춰 보는 조각, 원본도 글자 조각이 보임 → 겹쳐도 무관)
BOTH = set()


def draw_grad(px, p, text, align, spans=None):
    """H 층 = 획(단계 2 주황 한 색), L 층 = 둘레 1(검붉은 32,0,0). 원본 7A50 색표"""
    import hud_px as HP
    if spans:
        for x0, x1, t, al in spans:
            q = list(p); q[1] = p[1] + x0; q[3] = x1 - x0
            if t.strip():
                draw_grad(px, q, t, al)
        return
    _, u, v, w, h, lay = p[:6]
    lv = HP.word(text, w, h, 'B', align)
    ink = lv == 3; ring = lv == 1
    na = np.zeros((h, w), np.uint8); na[ink] = 2                # 한 색(주황 224,80,0) — 위아래 색 나눔은 사용자가 빼라고 함
    nb = np.zeros((h, w), np.uint8); nb[ring] = 1
    px[v:v + h, u:u + w] = (na << 2) | nb


STYLE_FROM = {338: 336}                        # 運転終了 = 運転(336 왼쪽) + 終了(338) — 색 단계를 같은 조각에서
FULL = {254, 255, 336, 337, 338}   # 432(通過駅)는 다른 층에 信号·計·注意 조각이 있어 제외(판 = CLUT 가 칠함)   # CLUT 가 4비트 전체를 읽는 2층 글자(본문층 + 다른 층 그림자·둘레)


def draw_full(px, p, text, align, spans=None):
    """2층 글자: 본문층 = 위아래 그라데이션(원본 단계 순서), 다른 층 = 둘레선(원본에서 본문 밖 최빈 단계). 다른 층이 판이면 그대로"""
    if spans:
        for x0, x1, t, al in spans:
            q = list(p); q[1] = p[1] + x0; q[3] = x1 - x0
            if t.strip():
                draw_full(px, q, t, al)
        return
    _, u, v, w, h, lay = p[:6]
    oth = 'L' if lay == 'H' else 'H'
    A = levels_of(px, u, v, w, h, lay); B = levels_of(px, u, v, w, h, oth)
    lv = [k for k in (1, 2, 3) if (A == k).sum() > 0.05 * (A > 0).sum()]
    def inner(k):                                               # 획 안쪽 단계(그라데이션) / 바탕에 닿는 단계(테두리) 구분 — 合格!! 의 빨간 테두리를 그라데이션으로 읽던 것
        m = A == k; nb = A == 0
        t = m & (np.roll(nb, 1, 0) | np.roll(nb, -1, 0) | np.roll(nb, 1, 1) | np.roll(nb, -1, 1))
        return 1 - t.sum() / max(1, m.sum())
    edge_a = [k for k in lv if inner(k) < 0.35]
    fills = [k for k in lv if k not in edge_a] or lv
    rows = {k: np.nonzero(A == k)[0].mean() for k in fills}
    grad = [max(fills, key=lambda k: (A == k).sum())] if fills else [3]   # 2026-10-19 사용자 «조잡해, 단색» — 위아래 두 단계 대신 원본 최다 단계 하나
    a_edge = edge_a[0] if edge_a else None
    out_b = B[(A == 0) & (B > 0)]
    plate = (B > 0).mean() > 0.6
    ring_lv = int(np.bincount(out_b, minlength=4)[1:].argmax() + 1) if len(out_b) and not plate and a_edge is None else None
    in_b = B[A > 0]
    in_lv = int(np.bincount(in_b, minlength=4).argmax()) if len(in_b) else 0
    import hud_px as HP
    lvw = HP.word(text, w, h, 'B', align)                            # 갈무리 비트맵(사용자 B안)
    ink = lvw == 3
    ys = np.nonzero(ink.any(1))[0]; y0 = ys.min() if len(ys) else 0; hh = (ys.max() - y0 + 1) if len(ys) else 1
    na = np.zeros((h, w), np.uint8)
    if len(grad) == 2:
        cut = y0 + int(round(hh * 0.6))
        na[:cut][ink[:cut]] = grad[0]; na[cut:][ink[cut:]] = grad[1]
    else:
        na[ink] = grad[0]
    if a_edge is not None:                                      # 본문층 테두리(合格!! 빨강)
        pd = np.pad(ink, 1); rm = np.zeros_like(pd)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                rm |= np.roll(np.roll(pd, dy, 0), dx, 1)
        na[rm[1:-1, 1:-1] & ~ink] = a_edge
    reg = px[v:v + h, u:u + w]
    if plate:
        nb = B
    else:
        nb = np.zeros((h, w), np.uint8)
        if ring_lv:
            rm = np.zeros_like(ink)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    rm |= np.roll(np.roll(ink, dy, 0), dx, 1)
            nb[rm & ~ink] = ring_lv
        nb[ink] = 0                                              # 획 안 다른 층은 비움(원본 음영 흉내는 노이즈만 생김)
    hpart, lpart = (na, nb) if lay == 'H' else (nb, na)
    reg[:] = (hpart << 2) | lpart


# 여러 조각에 걸친 한 낱말: 64px 캔버스에 한 번 그려 조각마다 잘라 넣음(크기 고르게). 'in' = 제자리, 're' = 그 레코드만 빈 자리로
# 上り勾配 = 3DA64(上り 117 + 勾配 98) · 下り勾配 = 3DA88(下 119 + り 118) + 3DAAC(勾配 98) — 사용자: 오르막/내리막/평지
PHRASES = [
    ('오르막', 117, [('in', 117, 0, 32, None), ('re', 98, 32, 64, ['3DA70'])]),     # 勾配(98) 자리는 勾配なし(99) 왼쪽 반과 같은 화소 → 옮김          # 표지 기둥(x≈40~)이 가리지 않게 0~40 안 가운데
    ('내리막', 117, [('in', 119, 0, 16, None), ('re', 118, 16, 32, ['3DAA0']), ('re', 98, 32, 64, ['3DAAC'])]),
]


def draw_phrases(px, orig, P, free, recs, LP, moved):
    import hud_px as HP
    for text, sty_id, parts in PHRASES:
        sp = P[sty_id]
        bg, fill, edge = style(levels_of(orig, *sp[1:6]))
        W_ = max(p[3] for p in parts); H_ = sp[4]
        lv = HP.word(text, W_, H_, 'B', ('c', 20))
        new = np.full(lv.shape, bg, np.uint8); new[lv == 3] = fill
        if edge is not None:
            new[lv == 1] = edge
        for kind, i, x0, x1, offs in parts:
            _, u, v, w, h, lay = P[i][:6]
            seg = new[:h, x0:x1]
            if kind == 're':
                lay, nu, nv, cmap = alloc(free, lay, offs, x1 - x0, h, recs, LP)
                u, v = nu, nv
                moved[('ph', text, i, x0)] = (nu, nv, cmap, offs, x1 - x0)   # 낱말마다 따로(오르막·내리막 둘 다 勾配 98 을 옮김)
            reg = px[v:v + h, u:u + (x1 - x0)]
            keep = (reg & 3) if lay == 'H' else (reg & 12)
            reg[:] = keep | ((seg << 2) if lay == 'H' else seg)


def draw_flip(px, p, text):
    """숫자판(플립) 글자: 4비트 색인 그대로 — 판(1~7)·접힘 줄 유지, 글자(8~10) 지우고 갈무리11 Bold 를 10(밝은 노랑), 접힘 줄 위는 8·9"""
    import hud_px as HP
    _, u, v, w, h, lay = p[:6]
    reg = px[v:v + h, u:u + w]
    glyph = (reg >= 8) & (reg <= 10)
    for y in range(h):                                          # 줄마다 글자 아닌 화소의 최빈값으로 메움(접힘 줄 9·10행은 2·3)
        row = reg[y]; ok = row[~glyph[y]]
        if len(ok):
            row[glyph[y]] = np.bincount(ok, minlength=16).argmax()
    lv = HP.word(text, w - 2, h - 2, 'A', 'c')                 # 한 글자 큰 칸 = 갈무리7 2배(14px, 원본 秒 크기)
    ink = np.zeros((h, w), bool); ink[1:h - 1, 1:w - 1] = lv == 3
    split = [y for y in range(1, h - 1) if np.bincount(reg[y, 2:-2], minlength=16).argmax() in (2, 3)]   # 접힘 줄(판 최빈 2·3)
    for y in range(h):
        reg[y][ink[y]] = (8 if y == split[0] else 9) if y in split else 10


FLIP = {234: '초'}


def draw_custom(px):
    """時間切れ(397, L층 빨간 열차 창 21~92×7~20: 바탕 2 빨강·글자 3 노랑·둘레 1) · 車内信号(269~271 같은 자리 64,384 위 띠: 4비트 바탕 8 노랑·글자 3)"""
    import hud_px as HP
    u, v, w, h = 21, 64 + 7, 76, 14                               # 원본 れ 끝 점까지(21~96)
    lv = HP.word('시간 초과', w, h, 'B', 'c')
    reg = px[v:v + h, u:u + w]
    new = np.full((h, w), 2, np.uint8); new[lv == 1] = 1; new[lv == 3] = 3
    reg[:] = (reg & 12) | new
    u, v, w, h = 65, 385, 30, 10                                # 띠 안쪽(테두리 0·1 열/줄 제외)
    import pixfont as PF2                                       # 갈무리7 을 7px 간격(공백 2)으로 — 30px 띠에 «차내 신호»
    reg = px[v:v + h, u:u + w]; reg[:] = 8; x = 0
    for ch in '차내 신호':
        if ch == ' ':
            x += 2; continue
        g = PF2.cell(ch, 7) > 0; gh, gw = g.shape; gw = min(gw, 7)
        sub = reg[(h - gh) // 2:(h - gh) // 2 + gh, x:x + gw]; sub[g[:, :gw][:sub.shape[0], :sub.shape[1]]] = 3; x += 7                              # 숫자판 秒 (3E010)


# 문장 통째 회색 조각은 «한 문장 = 비트 하나»(2026-10-19 확인): u160 v224 96×32 = H 비트0 스타트 안내(+컨티뉴) · H 비트1 信号は守って下さい · L 層 素晴らしい運転でした
#   (u, v, w, h, 층, 비트, 줄들, 첫 줄 y, x) — 그 비트만 지우고 갈무리9(원본 9px 줄·10 간격)로. 로딩 화면은 3E94C 를 80×20 으로 잘라 씀(첫 두 줄 ≤ 76px)
BITTEXT = [                                     # (u, v, w, h, 층, 층 안 비트, 줄들, 첫 줄 y, x, 레코드, 글꼴 크기, 줄 간격, 정렬)
    (160, 224, 96, 32, 'H', 0, ['스타트 버튼을', '눌러 주세요.', '(컨티뉴 스타트)'], 1, 4, ['3E94C'], 9, 10, 'l'),
    # u160 v160~255 96×96 = 세로 3칸 × 4비트 문장 12개(2026-10-19 판독). 조각 표(149·182·213·461·472·490)는 같은 칸을 한 조각으로 묶어 둠
    (160, 160, 96, 32, 'H', 1, ['정차역은', '지켜 주세요.'], 2, 3, ['3E904'], 11, 16, 'l'),
    (160, 160, 96, 32, 'H', 0, ['타행 운전을', '해 주세요.'], 2, 3, ['3E928'], 11, 16, 'l'),
    (160, 160, 96, 32, 'L', 1, ['"알림등" 점등 전', '브레이크 해제 금지.'], 2, 3, ['3E958'], 11, 16, 'l'),
    (160, 160, 96, 32, 'L', 0, ['테츠짱', '감동입니다!'], 2, 3, ['3E97C'], 11, 16, 'c'),       # 鉄ちゃん = 안내원 이름(파이널)
    (160, 192, 96, 32, 'H', 1, ['브레이크가', '너무 강합니다.'], 2, 3, ['3E910'], 11, 16, 'l'),
    (160, 192, 96, 32, 'H', 0, ['건널목 강행 횡단은', '매우 위험합니다.', '안전 확인,일단 정지!'], 0, 2, ['3E940'], 9, 11, 'l'),
    (160, 192, 96, 32, 'L', 1, ['브레이크 조작은', '부드럽고', '정성스럽게...'], 0, 3, ['3E964'], 9, 11, 'l'),
    (160, 192, 96, 32, 'L', 0, ['정확한 운전을', '명심해 주세요.'], 2, 3, ['3E988'], 11, 16, 'l'),
    (160, 224, 96, 32, 'H', 1, ['신호는', '지켜 주세요.'], 2, 3, ['3E91C'], 11, 16, 'l'),
    (160, 224, 96, 32, 'L', 1, ['훌륭한', '운전이었습니다!'], 2, 3, ['3E970'], 11, 16, 'c'),
]


PLANE_CLUT = {0: 0x7652, 1: 0x7612, 2: 0x75D2, 3: 0x7592}                                   # 비트 → 그 비트만 읽는 회색(16,16,16)
BITCLUT = {0x7652: 0, 0x7612: 1, 0x75D2: 2, 0x7592: 3, 0x7752: 0, 0x7712: 1, 0x76D2: 2, 0x7692: 3}   # 비트 하나만 읽는 CLUT(회색·빨강)


def draw_bittext(px, free, recs, LP, moved):
    """BITTEXT 칸을 잘라 쓰는 다른 레코드(안내문 낱자 を·使·ブ·危険·ボタン 등)는 그 비트 그림을 빈 자리로 복사해 옮긴 뒤 칸을 다시 그림"""
    mine = {o for t in BITTEXT for o in t[9]} | (hint_offsets() if HINTS_ON else set())   # 안내문 레코드는 안내문 단계가 새로 그림
    boxes = {(t[0], t[1], t[2], t[3]) for t in BITTEXT}
    for o, r in sorted(recs.items()):
        if o in mine or not r['w'] or not r['h'] or ('rec', int(o, 16)) in moved or ('hint', int(o, 16)) in moved:
            continue
        if not any(bu <= r['u'] < bu + bw and bv <= r['v'] < bv + bh for bu, bv, bw, bh in boxes):
            continue
        bit = BITCLUT.get(r['clut'])
        assert bit is not None, ('BITTEXT 칸을 비트 하나 CLUT 아닌 레코드가 씀', o, hex(r['clut']))
        u, v, w, h = r['u'], r['v'], r['w'], r['h']
        src = (px[v:v + h, u:u + w] >> bit) & 1
        lay0 = 'H' if bit >= 2 else 'L'; bl = bit & 1
        lay, nu, nv, cmap = alloc(free, lay0, [o], w, h, recs, LP)
        reg = px[nv:nv + h, nu:nu + w]
        val = (src << bl).astype(np.uint8)
        reg[:] = ((reg & 3) | (val << 2)) if lay == 'H' else ((reg & 12) | val)
        moved[('rec', int(o, 16))] = ('rec', nu, nv, w, h, r['x'], r['y'], cmap.get(r['clut'], r['clut']))
    for u, v, w, h, lay, bit, lines, y0, x0, _, size, pitch, align in BITTEXT:
        b = 1 << (bit + (2 if lay == 'H' else 0))
        reg = px[v:v + h, u:u + w]; reg &= np.uint8(15 ^ b)
        if any(PF.text(t, size).shape[1] > w - x0 for t in lines):
            size = 9                                            # 한 줄이라도 넘치면 문장 전체 갈무리9(크기 통일)
        for k, t in enumerate(lines):
            g = PF.text(t, size) > 0; gh, gw = g.shape
            xx = (w - gw) // 2 if align == 'c' else x0
            assert xx + gw <= w and y0 + pitch * k + gh <= h, (t, gw, gh)
            sub = reg[y0 + pitch * k:y0 + pitch * k + gh, xx:xx + gw]; sub[g] |= np.uint8(b)


# 발착 창 発·着(4비트 흰 글자 7, 회색 4 = 가장자리) — 두 층 함께 지우고 갈무리11 을 7 로
SOLID4 = {307: '발', 308: '착'}


def draw_solid4(px, P):
    for i, t in SOLID4.items():
        _, u, v, w, h = P[i][:5]
        g = PF.cell(t, 11) > 0; gh, gw = g.shape
        reg = px[v:v + h, u:u + w]; reg[:] = 0
        sub = reg[(h - gh) // 2:(h - gh) // 2 + gh, (w - gw) // 2:(w - gw) // 2 + gw]; sub[g[:sub.shape[0], :sub.shape[1]]] = 7


# 일시정지·화면 위치 조정 메뉴(2026-10-19 미작업 스샷 1·2·5·6): 문구 하나 = 레코드 몇 개 → 한 캔버스(x0 부터 폭 합)에 한 번 그려
# 레코드마다 잘라 빈 자리에 놓고 레코드 통째(u·v·w·h·x) 고침. 낱자 조각(は·い·え·て·に·す…)은 회색 안내문과 같이 써서 제자리 금지.
#   (문구, [(레코드, 폭)], 왼쪽 x, 높이, 정렬) · ARROWS = 화살표 레코드 x 만 옮김
MENU = [
    ('시발역으로', [('3E874', 28), ('3E880', 28), ('3E88C', 28)], -42, 14, 'c'),   # 始発駅に戻る — 시발역으로|뒤|로 조각이 겹치던 것 · 18차 뒤 28×3 고르게(48px 빈 자리 없음)
    ('예', [('3E8C8', 12), ('3E8D4', 12)], -12, 12, 'c'),                             # はい
    ('아니요', [('3E8E0', 12), ('3E8EC', 12), ('3E8F8', 12)], -18, 12, 'c'),          # いいえ
    ('화면 조정', [('40338', 72)], -36, 12, 'c'),                                  # 画面位置調整(제목)
    ('위', [('402E4', 12)], 0, 12, 'l'),                                                # ↑上
    ('아래', [('402FC', 24)], -6, 12, 'l'),                                             # ↓下
    ('왼쪽', [('40314', 24)], -6, 12, 'l'),                                             # ←左
    ('오른쪽', [('4032C', 36)], -12, 12, 'l'),                                          # →右
    ('기본값으로', [('40344', 12), ('40350', 12), ('4035C', 12), ('40368', 12), ('40374', 12)], -30, 12, 'c'),   # 標準に戻す
    ('조정 종료', [('40380', 24), ('4038C', 24)], -24, 12, 'c'),                        # 調整終了
]
# 19차 실기(사용자): 일시정지 메뉴 커서는 x116~124 고정 → 글자는 126~205(79px) 안. «시발역으로 뒤로» 88px·«화면 위치 조정» 80px 는 커서에 닿음 → 문구 줄임(사용자 선택), 굵은 둘레 없이 다른 항목처럼
NORING = {'시발역으로', '화면 조정'}
ARROWS = {'402F0': -18, '40308': -18, '40320': -24}
ARROWS.update({'3E520': 4, '3E52C': 20, '3CC78': 12, '3CC84': 28})   # 通過: «통» 앞 공백 대신 [통][과] 레코드를 4px 오른쪽으로(공백 넣으면 16px 칸에서 글자가 눌림 — 사용자 «통 자만 작다»)                                  # ↓·←·→ 화살표(글자가 넓어진 만큼 왼쪽으로)


def draw_menu(px, orig, P, free, recs, LP, moved, clut):
    import struct
    exe = open(H.EXE, 'rb').read()
    byuv = {}
    for p in P.values():
        byuv.setdefault((p[1], p[2], p[3], p[4]), p)
    mine = {o for _, parts, _, _, _ in MENU for o, _ in parts}
    styles = {}
    for i, p in P.items():                                      # 메뉴 레코드만 쓰던 조각 — 비우기·빈 자리는 free_vacate(여기서 조각 통째로 비우면 겹친 제자리 조각 3E898 始発駅に 을 덮음)
        if p[6] and set(p[6]) <= mine:
            _, u, v, w, h, lay = p[:6]
            styles[(u, v, w, h, lay)] = style(levels_of(orig, u, v, w, h, lay))
    jobs = []
    for text, parts, x0, hh, align in MENU:
        import hud_px as HP                                         # 20차 실기(사용자 «크기 맞춰라»): 運転再開·運転中止 등 다른 HUD 메뉴 글자와 같은 갈무리9(hud_px 'S') — 예전 갈무리11 은 혼자 큼
        big = HP.word(text, 200, hh, 'S', 'l') > 0
        cw = int(np.nonzero(big.any(0))[0].max()) + 2
        W = sum(w for _, w in parts); extra = max(0, cw - W)
        if extra:                                                   # 넓으면 마지막 조각 폭을 늘리고 가운데 맞춤이면 왼쪽으로
            parts = parts[:-1] + [(parts[-1][0], parts[-1][1] + extra)]; W += extra
            if align == 'c':
                x0 -= extra // 2
        ink = HP.word(text, W, hh, 'S', align) == 3
        pd = np.pad(ink, 1); ring = np.zeros_like(pd)                  # 둘레(始発駅に 주황 글자의 검은 테두리) — 캔버스 전체로 한 번
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ring |= np.roll(np.roll(pd, dy, 0), dx, 1)
        ring = ring[1:-1, 1:-1] & ~ink & (text not in NORING)
        x = x0
        for o, w in parts:
            r = recs[o]; p = byuv[(r['u'], r['v'], r['w'], r['h'])]
            pal = H.pal(clut, r['clut'])                        # 층은 조각 표 표시가 아니라 레코드 CLUT 이 실제로 읽는 쪽(下·て 는 표와 다름)
            lay = 'H' if any(pal[k][3] for k in (4, 8, 12)) else 'L'
            op = [k for k in (1, 2, 3) if pal[k * 4 if lay == 'H' else k][3]]
            L = levels_of(orig, p[1], p[2], p[3], p[4], lay)
            bg, fill, edge = styles.get((p[1], p[2], p[3], p[4], lay)) or style(L)
            if fill not in op:                                  # 가는 낱자(い)는 투명 단계를 글자로 고름 → 불투명 단계 중 원본 최빈
                cnt = np.bincount(L.ravel(), minlength=4); fill = max(op, key=lambda k: cnt[k])
            if edge not in op or edge == fill:
                edge = None
            seg = ink[:, x - x0:x - x0 + w]
            new = np.full((hh, w), bg, np.uint8); new[seg] = fill
            if edge is not None:
                new[ring[:, x - x0:x - x0 + w]] = edge
            jobs.append((o, lay, new, x))
            x += w
    for o, lay, new, x in sorted(jobs, key=lambda j: -j[2].size):   # 넓은 조각부터 자리 잡기(제목 82px)
        hh, w = new.shape
        lay, nu, nv, cmap = alloc(free, lay, [o], w, hh, recs, LP)
        reg = px[nv:nv + hh, nu:nu + w]
        reg[:] = ((reg & 3) | (new << 2)) if lay == 'H' else ((reg & 12) | new)
        a = int(o, 16); _, _, _, _, _, y, _, cl = struct.unpack_from('<4B2h2H', exe, a)
        moved[('rec', a)] = ('rec', nu, nv, w, hh, x, y, cmap.get(cl, cl))
    for o, nx in ARROWS.items():
        r = recs[o]
        moved[('rec', int(o, 16))] = ('rec', r['u'], r['v'], r['w'], r['h'], nx, r['y'], r['clut'])


HINT_TSV = os.path.join(ROOT, 'work', 'text', 'hint_kr.tsv')
SMALL_TSV = os.path.join(ROOT, 'work', 'text', 'hud_small_jp.tsv')
TAB, NL = chr(9), chr(10)
HINTS_ON = True                                # 회색 안내문: 한국어 낱말 그림 폭 합 ≈3,840 > 빈 자리 ≈880(10px 선반) → 다른 텍스처 페이지 필요, 보류
HINT_FONT = 'G9'
MAXW = 24                                      # 안내문 낱말 조각 최대 폭(빈 자리가 12px 칸 단위로 쪼개져 있음)
WHOLE = {149, 182, 213, 461, 472, 490}                 # 문장 통째 조각(두 층에 문장 하나씩) — 낱자 조각 비우기에서 제외


def draw_hints(px, orig, P, free, recs, LP, moved):
    """회색 안내문: 줄을 낱말 조각 그림(갈무리9)으로 만들어 빈 자리에 두고, 그 문구의 글자 레코드가 가리키게. 남는 글자 레코드는 크기 0.
    다른 화면도 쓰는 낱자 조각은 그대로, 안내문만 쓰는 낱자 조각 자리는 비워 재활용."""
    import hud_px as HP
    small = {int(l.split(TAB)[0]) for l in open(SMALL_TSV, encoding='utf-8').read().split(NL)[1:] if l} - WHOLE
    lines = {}
    for l in open(HINT_TSV, encoding='utf-8').read().split(NL)[1:]:
        if l:
            g, y, x, t = l.split(TAB); lines.setdefault(int(g, 16), []).append((int(y), int(x), t))
    inv = {}
    for i, p in P.items():
        for o in p[6]:
            inv[int(o, 16)] = i
    groups = {g[0]['o']: g for g in H.groups(list(recs.values()) if False else H.records(open(H.EXE, 'rb').read()))}
    hint_offs = set()
    for go in lines:
        hint_offs |= {r['o'] for r in groups[go]}
    # 안내문만 쓰는 낱자 조각 → 비움
    users = {}
    for go, g in groups.items():
        for r in g:
            users.setdefault(inv.get(r['o']), set()).add(go)
    freeable = [i for i in small if users.get(i) and users[i] <= set(lines)]
    keep_mask = {'H': np.zeros(px.shape, bool), 'L': np.zeros(px.shape, bool)}
    for i, p in P.items():
        if i not in freeable and i not in (139, 340, 341):     # 139·340·341 = 아틀라스 회색 비춰 보기(지켜야 할 그림 아님)
            keep_mask[p[5]][p[2]:p[2] + p[4], p[1]:p[1] + p[3]] = True
    for i in freeable:
        _, u, v, w, h, lay = P[i][:6]
        m = np.zeros(px.shape, bool); m[v:v + h, u:u + w] = True; m &= ~keep_mask[lay]
        px[m] = (px[m] & 3) if lay == 'H' else (px[m] & 12)
        free[lay] |= m
    fill = style(levels_of(orig, *P[156][1:6]))[1]          # 회색 글자 단계(速度 조각)
    plans = {}; need = {}
    for go, ls in lines.items():
        g = groups[go]
        text_recs = [r for r in g if inv.get(r['o']) in small]
        icon_recs = {inv.get(r['o']): r for r in g if inv.get(r['o']) in (319, 299)}
        lay0 = P[inv[text_recs[0]['o']]][5]; clut0 = text_recs[0]['clut']
        plan = []
        for y, x, t in ls:
            cur = x
            for k, part in enumerate(t.split('@')):
                if k % 2 == 1:
                    ir = icon_recs[int(part)]; plan.append((None, cur, y, ir)); cur += ir['w'] + 2; continue
                for wd in part.split(' '):
                    if not wd:
                        cur += 3; continue
                    lv = HP.word(wd, 200, 12, HINT_FONT, 'l') == 3
                    xs = np.nonzero(lv.any(0))[0]; ys = np.nonzero(lv.any(1))[0]
                    img = lv[ys.min():ys.max() + 1, xs.min():xs.max() + 1]      # 잉크만(둘레 없이)
                    cuts = [0]                                  # 긴 낱말은 빈 열에서 48px 안팎으로 잘라 여러 조각(빈 자리가 잘게 쪼개져 있음)
                    while img.shape[1] - cuts[-1] > MAXW:
                        gaps = [c for c in range(cuts[-1] + 6, cuts[-1] + MAXW + 1) if not img[:, c].any()]
                        cuts.append(gaps[-1] if gaps else cuts[-1] + MAXW)
                    cuts.append(img.shape[1])
                    for c0, c1 in zip(cuts, cuts[1:]):
                        seg = img[:, c0:c1]
                        if not seg.any():
                            continue
                        key = (wd + '#%d' % c0, lay0, clut0); need[key] = (seg, '%X' % text_recs[0]['o'])
                        plan.append((key, cur + c0, y + 1, None))
                    cur += img.shape[1] + 3
        texts = [p_ for p_ in plan if p_[0] is not None]
        assert len(texts) <= len(text_recs), '%X 낱말 %d > 레코드 %d' % (go, len(texts), len(text_recs))
        plans[go] = (plan, text_recs)
    chunks = {}
    HPm = {d: c for c, d in LP.items()}
    shelves = []                                            # 선반: (층, v, 남은 빈 구간들) — 높이 10 줄 띠에서 10줄 모두 빈 열 구간
    for ly in ('L', 'H'):
        f = free[ly]
        for v in range(0, 512 - 10 + 1, 10):
            if (v < 256) != (v + 9 < 256):
                continue
            col = f[v:v + 10].all(0); runs = []; x = 0
            while x < 256:
                if col[x]:
                    s0 = x
                    while x < 256 and col[x]: x += 1
                    runs.append([s0, x])
                else:
                    x += 1
            if runs:
                shelves.append((ly, v, runs))
    def place(w_, h_, lay0, clut0):
        for ly in (lay0, 'L' if lay0 == 'H' else 'H'):                   # 1px 단위 빈 사각형(선반 10줄 고정은 자리를 버림)
            if ly != lay0 and clut0 not in (LP if lay0 == 'H' else HPm):
                continue
            try:
                u, v = find_free(free[ly], w_ + 1, h_ + 1)
            except AssertionError:
                continue
            free[ly][v:v + h_ + 1, u:u + w_ + 1] = False
            return ly, u, v, ({} if ly == lay0 else {clut0: (LP if lay0 == 'H' else HPm)[clut0]})
        raise AssertionError('안내문 자리 없음 %dx%d' % (w_, h_))
        for ly, v, runs in shelves:
            if ly != lay0:
                pair = LP if lay0 == 'H' else HPm
                if clut0 not in pair:
                    continue
            for run in runs:
                if run[1] - run[0] >= w_:
                    u = run[0]; run[0] += w_ + 1
                    free[ly][v:v + h_, u:u + w_] = False
                    return ly, u, v, ({} if ly == lay0 else {clut0: (LP if lay0 == 'H' else HPm)[clut0]})
        raise AssertionError('안내문 선반 자리 없음 %dx%d' % (w_, h_))
    # ★2026-10-19 비트 평면 배치: 회색 글자는 1비트 그림 → 4비트 화소의 비트 4장(L0·L1·H0·H1) 중 «켜도 기존 레코드 그림이 안 바뀌는» 곳에 둠,
    #   레코드 CLUT 은 그 비트만 읽는 회색(PLANE_CLUT). 자리 ≈ 4배(예전 층 단위 빈 자리로는 72조각 뒤 실패).
    hint_set = {r['o'] for go in plans for r in plans[go][1]} | {ir['o'] for go in plans for _, _, _, ir in plans[go][0] if ir is not None}
    cur_recs = []
    for o_, r in recs.items():
        a_ = int(o_, 16)
        if a_ in hint_set or not r['w'] or not r['h']:
            continue
        mvd = moved.get(('rec', a_))
        if mvd:
            _, nu_, nv_, w2, h2, _, _, cl2 = mvd; r = dict(r, u=nu_, v=nv_, w=w2, h=h2, clut=cl2)
        cur_recs.append(r)
    _, clut_ = H.load(open(H.PAC, 'rb').read())
    covered = np.zeros(px.shape, bool); bfree = [np.ones(px.shape, bool) for _ in range(4)]
    for r in cur_recs:
        pal = H.pal(clut_, r['clut']).astype(int); sl = (slice(r['v'], r['v'] + r['h']), slice(r['u'], r['u'] + r['w']))
        reg = px[sl].astype(int); covered[sl] = True
        for b in range(4):
            bfree[b][sl] &= (pal[reg] == pal[reg | (1 << b)]).all(-1)
    spd = np.zeros(px.shape, bool); spd[384:448, 0:64] = True                 # 속도계(코드가 직접 그림, 레코드 표 밖)
    for b in range(4):
        bfree[b] &= (((px >> b) & 1) == 0) & ~DIFF & ~spd
    for key in sorted(need, key=lambda k: -need[k][0].size):   # 넓은 것부터
        img, o = need[key]; h_, w_ = img.shape
        for b in (3, 2, 1, 0):
            try:
                nu, nv = find_free(bfree[b], w_ + 1, h_ + 1)
                break
            except AssertionError:
                continue
        else:
            raise AssertionError('안내문 비트 자리 없음 %dx%d (놓은 %d / %d, 남은 비트 화소 %s)' % (w_, h_, len(chunks), len(need), [int(f.sum()) for f in bfree]))
        bfree[b][nv:nv + h_ + 1, nu:nu + w_ + 1] = False
        cov = covered[nv:nv + h_, nu:nu + w_] & img                   # 기존 레코드 위에 켠 화소는 다른 비트로 또 켜지 않음(두 비트 겹침 검사 안 함)
        for b2 in range(4):
            if b2 != b:
                bfree[b2][nv:nv + h_, nu:nu + w_] &= ~cov
        px[nv:nv + h_, nu:nu + w_] |= (img.astype(np.uint8) << b)
        chunks[key] = (nu, nv, w_, h_, PLANE_CLUT[b])
    n = 0
    for go, (plan, text_recs) in plans.items():
        texts = [p_ for p_ in plan if p_[0] is not None]
        for k, r in enumerate(text_recs):
            if k < len(texts):
                key, x, y, _ = texts[k]
                nu, nv, w_, h_, cl = chunks[key]
                moved[('hint', r['o'])] = ('rec', nu, nv, w_, h_, x, y, cl)
            else:
                moved[('hint', r['o'])] = ('rec', 0, 0, 0, 0, r['x'], r['y'], r['clut'])   # 안 쓰는 글자 레코드 = 크기 0
            n += 1
        for _, x, y, ir in [p_ for p_ in plan if p_[0] is None]:
            moved[('hint', ir['o'])] = ('rec', ir['u'], ir['v'] & 0xFF if False else ir['v'], ir['w'], ir['h'], x, ir['y'], ir['clut'])
    return n


def flag_groups():
    """실행 파일 문장(조각 묶음) = 레코드 tpage 0x8000 끝 표시까지 — 포인터 표 0x40AE0~ 와 같은 단위(겹침 기준 H.groups 는 문장을 쪼갬)"""
    out, cur = {}, []
    for r in H.records(open(H.EXE, 'rb').read()):
        cur.append(r)
        if r['tp'] & 0x8000:
            out[cur[0]['o']] = cur; cur = []
    return out


def hint_offsets():
    """안내문(HINT_TSV) 묶음의 레코드 위치(hud_check 제외용)"""
    gs = flag_groups(); P = pieces()
    small = {int(l.split(TAB)[0]) for l in open(SMALL_TSV, encoding='utf-8').read().split(NL)[1:] if l} - WHOLE
    inv = {int(o, 16): i for i, p in P.items() for o in p[6]}
    out = set()
    for l in open(HINT_TSV, encoding='utf-8').read().split(NL)[1:]:
        if l and int(l.split(TAB)[0], 16) in gs:
            out |= {'%X' % r['o'] for r in gs[int(l.split(TAB)[0], 16)] if inv.get(r['o']) in small}   # 글자 레코드만(아이콘은 검사)
    return out


def free_vacate(px, P, recs):
    """층 빈 자리 = «그대로 남는 레코드가 읽지 않는 층 비트». 옮겨 가는 레코드(낱말·EXTRA·RELOC·메뉴·안내문 글자)만 쓰던 화소는 그 층을 비움(px 고침).
    2026-10-19 18차 실기: 실행 중 덮이는 칸(DIFF, u192~255 v256~319)을 빼자 조각 표 기준 free_map 으론 자리가 모자라 바꿈.
    레코드가 비트를 «읽는다» = 그 비트를 뒤집으면 색이 바뀌는 색인이 있음(CLUT 16색 실제 비교)."""
    _, clut = H.load(open(H.PAC, 'rb').read())
    moving = {o for _, _, parts in PHRASES for kind, _, _, _, offs in parts if kind == 're' for o in offs}
    moving |= {o for _, _, _, offs in EXTRA for o in offs} | {o for i in RELOC for o in P[i][6]}
    moving |= {o for _, parts, _, _, _ in MENU for o, _ in parts}
    if HINTS_ON:
        moving |= {'%X' % o for o in hint_text_offsets(P)}
    reads = np.zeros((4,) + px.shape, bool)
    for o, r in recs.items():
        if o in moving or not r['w'] or not r['h'] or (r['tp'] & 0xF) != 5:
            continue
        pl = H.pal(clut, r['clut']).astype(int); idx = np.arange(16)
        sl = (slice(r['v'], r['v'] + r['h']), slice(r['u'], r['u'] + r['w']))
        for b in range(4):
            if (pl[idx] != pl[idx ^ (1 << b)]).any():
                reads[b][sl] = True
    spd = np.zeros(px.shape, bool); spd[384:448, 0:64] = True        # 속도계(코드가 직접 그림, 레코드 표 밖)
    keep = np.zeros(px.shape, bool)                                 # 제자리에 그리는 조각(레코드가 다 옮겨 가도 make 가 그 자리에 또 그림 — る→«로» 가 귀 자리를 덮던 것)
    T = {k for k in table() if k not in SKIP}
    for i in (T - RELOC - BOTH) | set(FLIP) | set(SOLID4):
        _, u, v, w, h = P[i][:5]; keep[v:v + h, u:u + w] = True
    for u, v, w, h in [(21, 71, 76, 14), (64, 384, 32, 12)] + [t[:4] for t in BITTEXT]:   # draw_custom·통째 문장 칸
        keep[v:v + h, u:u + w] = True
    out = {}
    for lay, bs in (('H', (2, 3)), ('L', (0, 1))):
        f = ~reads[bs[0]] & ~reads[bs[1]] & ~DIFF & ~spd & ~keep
        m = f & (((px >> 2) if lay == 'H' else (px & 3)) != 0)       # 옮겨 가는 레코드만 쓰던 화소 → 비움
        px[m] &= (3 if lay == 'H' else 12)
        out[lay] = f
    return out


def hint_text_offsets(P):
    """안내문 글자 레코드(낱자 조각을 가리키는 것만) 주소 — 아이콘·車内信号 는 제외"""
    small = {int(l.split(TAB)[0]) for l in open(SMALL_TSV, encoding='utf-8').read().split(NL)[1:] if l} - WHOLE
    lines = set()
    for l in open(HINT_TSV, encoding='utf-8').read().split(NL)[1:]:
        if l:
            lines.add(int(l.split(TAB)[0], 16))
    inv = {}
    for i, p in P.items():
        for o in p[6]:
            inv[int(o, 16)] = i
    groups = flag_groups()
    return {r['o'] for go in lines & set(groups) for r in groups[go] if inv.get(r['o']) in small}


def draw_hints_bits(px, orig, P, recs, moved, clut):
    """회색 안내문(낱자 조각 조합 문장) — 2026-10-19 비트 평면판.
    음절 하나 = 조각 하나(레코드 수가 모자라면 같은 낱말 안 이웃 음절을 묶음, 다른 문장에도 나오는 묶음 우선), 같은 조각 그림은 공유.
    조각은 4비트 화소의 비트 하나(PLANE_CLUT 회색)에 «켜도 기존 레코드 그림이 안 바뀌는» 곳에 둠. 글꼴 갈무리9(hud_px.syllable G9)."""
    import re
    import hud_px as HP
    small = {int(l.split(TAB)[0]) for l in open(SMALL_TSV, encoding='utf-8').read().split(NL)[1:] if l} - WHOLE
    lines = {}
    for l in open(HINT_TSV, encoding='utf-8').read().split(NL)[1:]:
        if l:
            g, y, x, t = l.split(TAB); lines.setdefault(int(g, 16), []).append((int(y), int(x), t))
    inv = {}
    for i, p in P.items():
        for o in p[6]:
            inv[int(o, 16)] = i
    groups = flag_groups()
    lines = {go: ls for go, ls in lines.items() if go in groups}          # 앞 단계가 옮긴 낱자 레코드도 여기서 덮어씀(patch_exe 는 나중 항목이 이김)
    G = {}
    def glyph(ch):
        if ch not in G:
            G[ch] = HP.syllable(ch, 'G9') > 0
        return G[ch]
    GH = 9
    # 1) 문장별 음절(위치) → 묶음
    units, recsof, icons = {}, {}, {}
    for go, ls in lines.items():
        g = groups[go]
        recsof[go] = [r for r in g if inv.get(r['o']) in small]
        ic = {inv.get(r['o']): r for r in g if inv.get(r['o']) in (319, 299)}
        us = []; icons[go] = []
        for y, x, t in ls:
            cur = x
            for k, part in enumerate(re.split(r'@(\d+)@', t)):
                if k % 2 == 1:
                    ir = ic[int(part)]; icons[go].append((ir, cur)); cur += ir['w'] + 2; continue
                for wi, wd in enumerate(part.split(' ')):
                    if wi:
                        cur += 3
                    word = []
                    for ch in wd:
                        word.append([ch, cur, y]); cur += glyph(ch).shape[1] + 1
                    if word:
                        us.append(word)
        units[go] = us
    freq = {}
    for go, us in units.items():
        for wd in us:
            for a, b in zip(wd, wd[1:]):
                freq[a[0] + b[0]] = freq.get(a[0] + b[0], 0) + 1
    chunks = {}
    for go, us in units.items():
        ws = [[[c[0], c[1], c[2]] for c in wd] for wd in us]         # 묶음 = [글자열, x, y]
        while sum(len(w) for w in ws) > len(recsof[go]):
            best = None
            for wi, w in enumerate(ws):
                for k in range(len(w) - 1):
                    s2 = w[k][0] + w[k + 1][0]
                    key = (-freq.get(s2, 0) if len(s2) == 2 else 0, len(s2))
                    if best is None or key < best[0]:
                        best = (key, wi, k)
            assert best, '%X 묶을 음절 없음' % go
            _, wi, k = best; w = ws[wi]; w[k][0] += w[k + 1][0]; del w[k + 1]
        chunks[go] = [c for w in ws for c in w]
    # 2) 조각 그림(공유)
    imgs = {}
    for go, cs in chunks.items():
        for t, _, _ in cs:
            if t not in imgs:
                cols = [glyph(ch) for ch in t]
                w = sum(c.shape[1] for c in cols) + len(cols) - 1
                im = np.zeros((GH, w), bool); x = 0
                for c in cols:
                    im[GH - c.shape[0]:, x:x + c.shape[1]] |= c; x += c.shape[1] + 1
                imgs[t] = im
    # 3) 비트 평면 빈 자리
    hint_offs = {r['o'] for go in lines for r in recsof[go]}            # 글자 레코드만(아이콘 299·319·車内信号 271 은 그대로 지켜야 함)
    # (안내문만 쓰던 낱자 조각 자리 비우기는 make 첫머리 free_vacate — 층 단위 옮김 조각이 그 자리를 쓰므로 여기서 또 지우면 안 됨)
    covered = np.zeros(px.shape, bool); bfree = [np.ones(px.shape, bool) for _ in range(4)]
    for o_, r in recs.items():
        a_ = int(o_, 16)
        if a_ in hint_offs or not r['w'] or not r['h']:
            continue
        mvd = moved.get(('rec', a_))
        if mvd:
            _, nu_, nv_, w2, h2, _, _, cl2 = mvd; r = dict(r, u=nu_, v=nv_ & 0x1FF, w=w2, h=h2, clut=cl2)
        pal = H.pal(clut, r['clut']).astype(int); sl = (slice(r['v'], r['v'] + r['h']), slice(r['u'], r['u'] + r['w']))
        reg = px[sl].astype(int); covered[sl] = True
        for b in range(4):
            bfree[b][sl] &= (pal[reg] == pal[reg | (1 << b)]).all(-1)
    spd = np.zeros(px.shape, bool); spd[384:448, 0:64] = True        # 속도계(코드가 직접 그림)
    for b in range(4):
        bfree[b] &= (((px >> b) & 1) == 0) & ~DIFF & ~spd
    place = {}
    for t in sorted(imgs, key=lambda t: -imgs[t].size):
        im = imgs[t]; h_, w_ = im.shape
        for b in (3, 2, 1, 0):
            try:
                nu, nv = find_free(bfree[b], w_ + 1, h_ + 1); break
            except AssertionError:
                continue
        else:
            raise AssertionError('안내문 비트 자리 없음 %r %dx%d (놓은 %d / %d)' % (t, w_, h_, len(place), len(imgs)))
        bfree[b][nv:nv + h_ + 1, nu:nu + w_ + 1] = False
        cov = covered[nv:nv + h_, nu:nu + w_] & im
        for b2 in range(4):
            if b2 != b:
                bfree[b2][nv:nv + h_, nu:nu + w_] &= ~cov
        px[nv:nv + h_, nu:nu + w_] |= (im.astype(np.uint8) << b)
        place[t] = (nu, nv, w_, h_, PLANE_CLUT[b])
    # 4) 레코드
    n = 0
    for go, cs in chunks.items():
        for k, r in enumerate(recsof[go]):
            if k < len(cs):
                t, x, y = cs[k]; nu, nv, w_, h_, cl = place[t]
                moved[('hint', r['o'])] = ('rec', nu, nv, w_, h_, x, y + 1, cl)
            else:
                moved[('hint', r['o'])] = ('rec', 0, 0, 0, 0, r['x'], r['y'], r['clut'])   # 안 쓰는 글자 레코드 = 크기 0
            n += 1
        for ir, x in icons[go]:
            moved[('hint', ir['o'])] = ('rec', ir['u'], ir['v'], ir['w'], ir['h'], x, ir['y'], ir['clut'])
    return n, len(imgs), sum(im.size for im in imgs.values())


def make(pac):
    px, clut = H.load(pac); orig = px.copy()
    P = pieces(); T = {k: v for k, v in table().items() if k not in SKIP}
    recs = {('%X' % r['o']): r for r in H.records(open(H.EXE, 'rb').read())}
    free = free_vacate(px, P, recs); moved = {}
    LP = lpair(clut)
    draw_menu(px, orig, P, free, recs, LP, moved, clut)       # 메뉴 제목(82·48px)이 가장 넓음 — 빈 자리 먼저(18차 뒤: 실행 중 덮이는 칸 제외로 자리 빠듯)
    draw_phrases(px, orig, P, free, recs, LP, moved)           # 큰 것(32px)부터 빈 자리 잡기
    for k, (i, text, align, offs) in enumerate(EXTRA):
        _, u, v, w, h, lay = P[i][:6]
        src = levels_of(orig, u, v, w, h, lay)
        sty = style(src)
        if align == 'l':                                        # 글자 폭만큼만 옮김(까요? 48→ 잉크 폭) — 18차 뒤 48px 빈 자리 없음
            import hud_px as HP
            cols = np.nonzero((HP.word(text, w, h, 'B' if h > 14 else 'S', align) > 0).any(0))[0]
            w = int(cols.max()) + 2; src = src[:, :w]
        lay, nu, nv, cmap = alloc(free, lay, offs, w, h, recs, LP)
        reg = px[nv:nv + h, nu:nu + w]
        reg[:] = ((reg & 3) | (src << 2)) if lay == 'H' else ((reg & 12) | src)
        q = list(P[i]); q[1], q[2], q[3], q[5] = nu, nv, w, lay
        draw_piece(px, q, text, align, clut, None, sty); moved[('ex', k)] = (nu, nv, cmap, offs, w)
    order = sorted(T, key=lambda i: -(P[i][3] * P[i][4]) if i in RELOC else 0)
    for i in order:
        text, align = T[i]
        p = P[i]
        if i in BOTH:
            _, u, v, w, h, lay = p[:6]
            both = free['H'] & free['L']
            nu, nv = find_free(both, w, h)
            free['H'][nv:nv + h, nu:nu + w] = False; free['L'][nv:nv + h, nu:nu + w] = False
            q = list(p); q[1], q[2] = nu, nv
            draw_grad(px, q, text, align); moved[i] = (nu, nv, {}, list(p[6]))
        elif i in RELOC:
            _, u, v, w, h, lay = p[:6]
            cmap = {}
            offs = [o for o in p[6] if i not in ONLY or ONLY[i][0] <= int(o, 16) < ONLY[i][1]]
            src = levels_of(orig, u, v, w, h, lay)
            lay, nu, nv, cmap = alloc(free, lay, offs, w, h, recs, LP)
            reg = px[nv:nv + h, nu:nu + w]                          # 원본 꼴(바탕·글자·테두리 단계)을 새 자리에 먼저 복사해 style() 이 읽게
            reg[:] = ((reg & 3) | (src << 2)) if lay == 'H' else ((reg & 12) | src)
            q = list(p); q[1], q[2], q[5] = nu, nv, lay
            draw_piece(px, q, text, align, clut); moved[i] = (nu, nv, cmap, offs)
        elif i in GRAD:                                         # 運転再開·運転終了·セクション通過(CLUT 7A50): 원본처럼 위 노랑→아래 주황 + 검붉은 둘레, 셋 다 같은 규칙
            draw_grad(px, p, text, align, SPANS.get(i))
        elif i in FULL:                                         # 2층 글자: 본문층만 글자색·테두리로 깔끔하게, 다른 층(음영)은 칸 안 비움
            _, u0, v0, w0, h0, lay0 = p[:6]
            sty = style(levels_of(orig, *P[STYLE_FROM.get(i, i)][1:6]))   # 한 문구 조각들은 같은 색(運転|終了)
            reg = px[v0:v0 + h0, u0:u0 + w0]
            reg[:] = (reg & 12) if lay0 == 'H' else (reg & 3)
            draw_piece(px, p, text, align, clut, SPANS.get(i), sty)
        else:
            draw_piece(px, p, text, align, clut, SPANS.get(i))
    for i, t in FLIP.items():
        draw_flip(px, P[i], t)
    draw_custom(px)
    draw_solid4(px, P)
    draw_bittext(px, free, recs, LP, moved)
    if HINTS_ON:
        draw_hints_bits(px, orig, P, recs, moved, clut)
    return px, orig, clut, moved


def patch_exe(exe, moved):
    """옮긴 조각을 가리키는 레코드의 u·v(+tpage v256 비트)"""
    import struct
    P = pieces(); exe = bytearray(exe); n = 0
    for i, mv in moved.items():
        if mv[0] == 'rec':                                      # 안내문: 레코드 통째(u, v, w, h, x, y, CLUT)
            _, nu, nv, w, h, x, y, cl = mv; a = i[1]
            u0, v0, w0, h0, x0, y0, tp, cl0 = struct.unpack_from('<4B2h2H', exe, a)
            tp = (tp & ~0x10) | (0x10 if nv >= 256 else 0)
            struct.pack_into('<4B2h2H', exe, a, nu, nv & 0xFF, w, h, x, y, tp, cl); n += 1
            continue
        nu, nv, cmap, offs = mv[:4]
        for o in offs:
            a = int(o, 16)
            u, v, w, h, x, y, tp, cl = struct.unpack_from('<4B2h2H', exe, a)
            if len(mv) > 4:
                w = mv[4]                                       # 낱말 캔버스 조각 폭(り 16 → 16, 勾配 32)
            tp = (tp & ~0x10) | (0x10 if nv >= 256 else 0)
            cl = cmap.get(cl, cl)
            struct.pack_into('<4B2h2H', exe, a, nu, nv & 0xFF, w, h, x, y, tp, cl); n += 1
    for i, w in WIDTH.items():
        for o in P[i][6]:
            exe[int(o, 16) + 2] = w; n += 1
    return bytes(exe), n


def apply(pac, exe):
    """hud_kr 다음에(같은 아틀라스) → (새 실행 파일, 바꾼 레코드 수)"""
    px, _, _, moved = make(bytes(pac)); H.store(pac, px)
    return patch_exe(exe, moved)


def preview():
    pac = open(H.PAC, 'rb').read(); exe = open(H.EXE, 'rb').read()
    px, orig, clut, moved = make(pac)
    T = table(); P = pieces()
    exe = patch_exe(exe, moved)[0]
    want = {k for k in T}
    recs = {('%X' % r['o']): r for r in H.records(exe)}
    G = json.load(open(os.path.join(ROOT, 'work', 'hud_groups.json')))
    exe0 = open(H.EXE, 'rb').read()
    gs0 = H.groups(H.records(exe0)); gs1 = H.groups(H.records(exe))   # 원본 = 원래 레코드, 한글 = 고친 레코드
    rows = []
    for g, g1 in zip(gs0, gs1):
        ids = [t[0] for o, l in G if o == '%X' % g[0]['o'] for t in l]
        if not want & set(ids):
            continue
        a = H.compose(orig, clut, g); b = H.compose(px, clut, g1)
        if a.shape[1] > 300:
            continue
        rows.append(np.concatenate([a, np.full((a.shape[0], 3, 4), 255, np.uint8), b], 1))
    W = 1300; x = y = rh = 0; pos = []
    for r in rows:
        if x + r.shape[1] * 2 > W:
            x = 0; y += rh + 6; rh = 0
        pos.append((x, y)); x += r.shape[1] * 2 + 10; rh = max(rh, r.shape[0] * 2)
    out = Image.new('RGB', (W, y + rh + 4), (0, 0, 0))
    for r, (x, y) in zip(rows, pos):
        out.paste(Image.fromarray(r).convert('RGB').resize((r.shape[1] * 2, r.shape[0] * 2), Image.NEAREST), (x, y))
    for k in range(0, out.size[1], 900):
        out.crop((0, k, W, min(out.size[1], k + 900))).save(os.path.join(ROOT, 'work', 'gfx', 'prev_hud_words_g%d.png' % (k // 900)))
    print('미리보기 work/gfx/prev_hud_words_g0~%d.png' % ((out.size[1] - 1) // 900))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
