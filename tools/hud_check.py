# -*- coding: utf-8 -*-
r"""HUD 아틀라스 검산 (2026-10-19, 9차 속도계 깨짐 뒤): 번역 대상이 아닌 레코드는 «실제 색으로» 한 화소도 안 바뀌어야 함
  레코드마다 원본(원래 실행 파일·원래 아틀라스) / 새것(고친 실행 파일·새 아틀라스)을 그 CLUT 16색으로 그려 비교.
  번역 조각을 가리키는 레코드는 제외(바뀌어야 정상). python tools/hud_check.py → 바뀐 레코드 목록, 없으면 «OK»
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hud as H  # noqa: E402
import hud_kr as HK  # noqa: E402
import hud_words as W  # noqa: E402
import logo_kr as LG  # noqa: E402


def render(px, clut, r):
    return H.pal(clut, r['clut'])[px[r['v']:r['v'] + r['h'], r['u']:r['u'] + r['w']]]


def check(pac):
    exe0 = open(H.EXE, 'rb').read()
    px0, clut = H.load(pac)
    pac1 = bytearray(pac); HK.apply(pac1)
    exe1, _ = W.apply(pac1, exe0); LG.apply(pac1)
    px1, _ = H.load(bytes(pac1))
    P = W.pieces()
    trans = set(k for k in W.table() if k not in W.SKIP) | {117, 118, 119, 98, 99, 337, 112, 234, 221, 397, 269, 270, 271, 220, 240, 249, 219, 196, 208}   # 337 = 336 칸 그림 안 · 112 秒 = hud_kr · 220·240·249 = logo_kr
    hk = {('%X' % o) for o, *_ in HK.ITEMS} | {'%X' % HK.FULL[0]}
    skip_offs = set(hk) | {o for _, _, _, offs in W.EXTRA for o in offs}
    skip_offs |= {o for _, parts, _, _, _ in W.MENU for o, _ in parts} | set(W.ARROWS) | {o for i in W.SOLID4 for o in P[i][6]} | {o for t in W.BITTEXT for o in t[9]} | (W.hint_offsets() if W.HINTS_ON else set())   # 일시정지·화면 조정 메뉴(통째로 옮김)
    for i in trans:
        skip_offs |= set(P[i][6])
    r0 = {r['o']: r for r in H.records(exe0)}; r1 = {r['o']: r for r in H.records(exe1)}
    bad = []
    IGNORE = {(64, 481, 32, 31), (96, 481, 32, 31), (0, 160, 240, 47)}   # 회색 비춰 보기 조각(340·341·139)
    for o, a in r0.items():
        so = '%X' % o
        if so in skip_offs or a['w'] == 0 or a['h'] == 0:
            continue
        if any(so in P[i][6] for i in trans):
            continue
        if (a['u'], a['v'], a['w'], a['h']) in IGNORE or (a['tp'] & 0xF) != 5:   # tpage x ≠ 320 = 아틀라스 아닌 VRAM 을 읽음(4029C·402CC tp 0x13)
            continue
        b = r1[o]
        A = render(px0, clut, a); B = render(px1, clut, b)
        if A.shape == B.shape and np.abs(A.astype(int) - B.astype(int)).max() <= 8:
            continue                                            # 빨간 상자 248↔240 처럼 눈에 안 띄는 차이
        if A.shape != B.shape or (A != B).any():
            n = int((A != B).any(-1).sum()) if A.shape == B.shape else -1
            bad.append((so, a['u'], a['v'], a['w'], a['h'], '%04X' % a['clut'], n, so in hk))
    D = H.copies_differ(pac)                                   # ★사본별 고유 그림(차종별 속도계)은 한 화소도 안 바뀌어야(11차 실기 깨짐)
    for c in H.COPIES:
        n = int(((H.load(pac, c)[0] != H.load(bytes(pac1), c)[0]) & D).sum())
        if n:
            bad.append(('사본 %X 고유 그림' % c[0], 0, 0, 0, 0, '-', n, False))
    if W.HINTS_ON:                                              # 통째 문장 비트(BITTEXT)는 안내문 배치 뒤에도 그대로여야(18차 스타트 안내 깨짐)
        a, _, _, _ = W.make(bytes(pac)); W.HINTS_ON = False
        b, _, _, _ = W.make(bytes(pac)); W.HINTS_ON = True
        for t in W.BITTEXT:
            u, v, w, h, lay, bit = t[:6]; sh = bit + (2 if lay == 'H' else 0)
            n = int((((a[v:v + h, u:u + w] >> sh) & 1) != ((b[v:v + h, u:u + w] >> sh) & 1)).sum())
            if n:
                bad.append(('통째 문장 ' + t[6][0], u, v, w, h, '-', n, False))
    return bad


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    bad = check(open(H.PAC, 'rb').read())
    for b in bad:
        print('바뀜', *b)
    print('OK' if not bad else '바뀐 레코드 %d' % len(bad))
