# -*- coding: utf-8 -*-
r"""번역 TSV 검사 (2026-10-18) — 쓰는 순간 막기
  python tools/check_tr.py [ko_NN.tsv …]   (기본 work/text/tsv/*.tsv)
  막는 것: 바이트 > 예산 · 제어 토큰(@1 @2 @3 @r ?0 ?1) 개수·순서 다름 · 반각 글자(@·? 토큰 밖) · 탭/개행
           · 줄(@r 사이) 칸 수 > 그 행 원문 최장 줄(상자 글) · 줄 수 다름 
  인코딩 = 한글 2 B · 그 밖 cp932 (전각 2 B) · 토큰 2 B
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = re.compile(r'@[123r]|\?[01]')
PUNCT = ',.!?:;)]}\'"~、。，．！？：；）］｝」』】〉》”’…‥・·～〜♪♥'
# 포인터 표로 다시 배치하는 문자열 영역 (PAC 위치 [시작, 끝), 적재 주소, 포인터를 찾을 파일 범위) — 문장별 예산 대신 영역 합으로 검사(build.py)
POOLS = {'25': (0x80000, 0x81870, 0x80069000, (0x80000, 0x86800)),
         '34': (0xBA830, 0xBCF28, 0x8006F000, (0xBA800, 0xC8800))}


def in_pool(pos):
    o = int(pos, 16)
    return any(a <= o < b for a, b, _, _ in POOLS.values())
SQ = re.compile('([' + re.escape(PUNCT) + '])[ 　](?![ 　])')


def squeeze(t):
    """문장부호 뒤 공백 1칸 지움(2칸 이상은 칸 맞춤이라 둠)"""
    return SQ.sub(r'\1', t)


def nbytes(s):
    n = 0
    for part in TOK.split(s):
        for c in part:
            if '가' <= c <= '힣':
                n += 2
            else:
                n += len(c.encode('cp932'))
    return n + 2 * len(TOK.findall(s))


def lines(s):
    return [len(x) for x in TOK.sub(lambda m: '\n' if m.group() == '@r' else '', s).split('\n')]


def check(row):
    pos, fi, bud, src, kr = row
    errs = []
    if not kr:
        return errs
    kr2 = squeeze(kr)
    if kr2 != kr:
        errs.append('부호 뒤 공백')
    try:
        b = nbytes(kr)
    except UnicodeEncodeError as e:
        return ['인코딩 불가 %r' % e.object[e.start:e.end]]
    if b > int(bud) and not in_pool(pos):
        errs.append('바이트 %d > %s' % (b, bud))
    if TOK.findall(src) != TOK.findall(kr):
        errs.append('토큰 %s ≠ %s' % (TOK.findall(src), TOK.findall(kr)))
    rest = TOK.sub('', kr)
    if any(ord(c) < 0x80 for c in rest):
        errs.append('반각 %r' % [c for c in rest if ord(c) < 0x80])
    if '@r' in src.rstrip('@r'):
        sl, kl = lines(src), lines(kr)
        w = max(sl)                                     # 줄 예산 = 그 행 원문 가장 긴 줄(상자 폭 이하)
        for i, k in enumerate(kl):
            if k > w:
                errs.append('%d줄 %d칸 > 원문 최장 %d' % (i + 1, k, w))
        if len(kl) != len(sl):
            errs.append('줄 수 %d ≠ %d' % (len(kl), len(sl)))
    return errs


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, 'work', 'text', 'tsv', '*.tsv')))
    bad = done = tot = 0
    for p in files:
        for ln in open(p, encoding='utf-8').read().split('\n')[1:]:
            if not ln:
                continue
            row = (ln.split('\t') + [''] * 5)[:5]
            tot += 1; done += bool(row[4])
            e = check(row)
            if e:
                bad += 1
                print('%s %s %s | %s | %s' % (os.path.basename(p), row[0], e, row[3][:30], row[4]))
    print('검사 %d줄 · 번역 %d · 오류 %d' % (tot, done, bad))
