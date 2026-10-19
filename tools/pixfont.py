# -*- coding: utf-8 -*-
r"""갈무리 비트맵 글꼴(BDF) 한 줄 그리기 (2026-10-19) — 8~9px 작은 글자용(사용자: 나눔은 8px 에서 뭉개짐 → 갈무리)
  text(문장, 크기 7/9/11) → 0/1 배열(잉크 위·아래·오른쪽 끝까지 잘라냄)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bdf  # noqa: E402

DIR = r'C:\claude\utils\font\Galmuri-v2.40.3'
_cache = {}


def font(size, bold=False):
    k = (size, bold)
    if k not in _cache:
        _cache[k] = bdf.load(os.path.join(DIR, 'Galmuri%d%s.bdf' % (size, '-Bold' if bold else '')))
    return _cache[k]


def text(s, size, bold=False):
    """bold = 갈무리11 Bold 만 있음"""
    g, asc = font(size, bold)
    h = asc + 3
    cols = []
    for ch in s:
        a, adv = bdf.render(g, asc, ord(ch), max(adv_of(g, ch), 1), h, asc)
        cols.append(np.array(a, float)[:, :adv] if adv else np.zeros((h, 0)))
    a = np.concatenate(cols, 1)
    ys = np.nonzero(a.max(1))[0]; xs = np.nonzero(a.max(0))[0]
    return a[ys.min():ys.max() + 1, :xs.max() + 1]


def adv_of(g, ch):
    v = g.get(ord(ch))
    assert v is not None, '갈무리에 없는 글자 %r' % ch
    return max(v[4], v[0] + max(0, v[2]))


def cell(ch, size, bold=False):
    """글자 하나를 «글꼴 고정 칸»으로(잉크로 자르지 않음) — 세로쓰기에서 글자마다 가운데가 달라 비뚤어지는 것 막기.
    가로 = 전진 폭, 세로 = 한글 '한·가·각' 잉크 줄 범위(글꼴마다 같음)"""
    g, asc = font(size, bold)
    h = asc + 3
    if (size, bold) not in _rows:
        rr = []
        for c in '한가각':
            a, adv = bdf.render(g, asc, ord(c), adv_of(g, c), h, asc)
            ys = np.nonzero(np.array(a).max(1))[0]; rr += [ys.min(), ys.max()]
        _rows[(size, bold)] = (min(rr), max(rr) + 1)
    y0, y1 = _rows[(size, bold)]
    a, adv = bdf.render(g, asc, ord(ch), adv_of(g, ch), h, asc)
    a = np.array(a, float)[y0:y1, :adv]
    xs = np.nonzero(np.array(bdf.render(g, asc, ord('한'), adv_of(g, '한'), h, asc)[0]).max(0))[0]
    return a[:, :max(xs.max() + 1, 1)] if a.shape[1] > xs.max() + 1 and not a[:, xs.max() + 1:].any() else a


_rows = {}
