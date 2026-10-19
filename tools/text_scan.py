# -*- coding: utf-8 -*-
r"""TRAIN.PAC 평문 SJIS 문자열 후보 (2026-10-18)
  0 으로 끝나는 SJIS(전각 + 반각 ASCII 허용) 문자열 중 가나 비율·길이로 거짓 양성(그림 데이터) 거름.
  python tools/text_scan.py → work/text/scan_pac.tsv (위치·바이트 수·문자열)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strings(d):
    out = []; i = 0; n = len(d)
    while i < n:
        j = i; chars = []
        while j < n - 1:
            b = d[j]
            if (0x81 <= b <= 0x9f or 0xe0 <= b <= 0xef) and 0x40 <= d[j + 1] <= 0xfc and d[j + 1] != 0x7f:
                chars.append(d[j:j + 2].decode('cp932', 'replace')); j += 2
            elif 0x20 <= b < 0x7f or b in (0x0a,):
                chars.append(chr(b)); j += 1
            else:
                break
        if j < n and d[j] == 0 and len(chars) >= 2:
            s = ''.join(chars)
            kana = len(re.findall(r'[ぁ-んァ-ヶー]', s)); wide = sum(1 for c in s if ord(c) > 0x7f)
            if wide >= 2 and (kana >= max(1, wide * 0.25) or wide <= 6 and kana >= 1) and '�' not in s:
                out.append((i, j - i, s))
            i = j + 1
        else:
            i = max(j, i + 1)
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    d = open(os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC'), 'rb').read()
    r = strings(d)
    os.makedirs(os.path.join(ROOT, 'work', 'text'), exist_ok=True)
    with open(os.path.join(ROOT, 'work', 'text', 'scan_pac.tsv'), 'w', encoding='utf-8') as f:
        f.write('위치\t바이트\t원문\n')
        for o, n, s in r:
            f.write('%08X\t%d\t%s\n' % (o, n, s.replace('\n', '\n')))
    print('문자열', len(r), '바이트', sum(n for _, n, _ in r))
    import collections
    c = collections.Counter(o >> 20 for o, _, _ in r)
    print(sorted(c.items()))
