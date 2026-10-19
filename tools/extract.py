# -*- coding: utf-8 -*-
r"""평문 SJIS 문자열 뽑기 → work/text/src.tsv (2026-10-18)
  대상 = 섹터 지도 22(메뉴·게임 오버레이)·25(시각표)·27(다이아 이름)·34(열차 소개) 파일 안, 4바이트 정렬에서 시작해 0 으로 끝나는 문자열.
  제어 = @1 @2 @3(색) @r(줄 끝) — 본문에 그대로 둔다(번역문도 같은 순서로).
  예산 = 원문 바이트 수(0 앞까지). 번역문은 SJIS·한글 코드 2바이트/전각, ASCII 1바이트.
  python tools/extract.py → work/text/src.tsv (번호 PAC위치 · 파일 · 예산 · 원문)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import secmap as M  # noqa: E402

FILES = (22, 25, 27, 34)
# 오버레이(22) 안 텍스트 구간 — 코드·그림 구간의 거짓 양성 제외(손으로 확인한 범위)
RANGES = {22: [(0x4B800, 0x4EC10)]}   # 2026-10-18: 0x4E600~ 설정 설명·자동 로드 메시지 누락 → 넓힘
JUNK = re.compile(r'^[!#$<&+.@A-Za-z0-9]?[^\x00-\x7f]{1,2}$')


def scan(d, lo, hi):
    out = []; i = lo
    while i < hi:
        if i % 4 or (i > lo and d[i - 1] != 0):
            i += 1; continue
        j = i; wide = 0; ok = True
        while j < hi:
            b = d[j]
            if (0x81 <= b <= 0x9f or 0xe0 <= b <= 0xef) and j + 1 < hi and 0x40 <= d[j + 1] <= 0xfc and d[j + 1] != 0x7f:
                j += 2; wide += 1
            elif 0x20 <= b < 0x7f:
                j += 1
            else:
                break
        s = None
        if j < hi and d[j] == 0 and wide >= 1:
            try:
                s = d[i:j].decode('cp932')
            except UnicodeDecodeError:          # 미정의 코드 = 글자가 아님
                pass
        if s is not None:
            out.append((i, j - i, s))
            i = j + 1
        else:
            i += 4 - i % 4 if i % 4 else 4
    return out


def rows():
    pac = open(os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC'), 'rb').read()
    T = {i: (lba, n) for i, lba, n, t in M.table()}
    out = []
    for fi in FILES:
        lba, n = T[fi]; lo = (lba - M.PAC_LBA) * 2048; hi = lo + n * 2048
        for a, b in RANGES.get(fi, [(lo, hi)]):
            for o, ln, s in scan(pac, a, b):
                if JUNK.match(s) and fi == 22:
                    continue
                if ':/' in s or ':\\' in s:                 # 개발 경로(E:/cddata/…)
                    continue
                if len(s) <= 3 and sum(ord(c) > 0x7f for c in s) <= 1:   # 그림·코드 자리 거짓 양성(誓·!話·I≡ 등)
                    continue
                out.append((o, fi, ln, s))
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    r = rows()
    os.makedirs(os.path.join(ROOT, 'work', 'text'), exist_ok=True)
    with open(os.path.join(ROOT, 'work', 'text', 'src.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('위치\t파일\t예산\t원문\n')
        for o, fi, ln, s in r:
            f.write('%06X\t%d\t%d\t%s\n' % (o, fi, ln, s))
    import collections
    print('문장', len(r), '바이트', sum(x[2] for x in r), collections.Counter(x[1] for x in r))
