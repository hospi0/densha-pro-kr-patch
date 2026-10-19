# -*- coding: utf-8 -*-
r"""번역(위치→번역) 파일들을 TSV 의 번역 열에 채운다 (2026-10-18)
  python tools/merge_tr.py   ← work/text/kr/*.txt (위치<TAB>번역) → work/text/tsv/ko_NN.tsv
  ⚠️문장부호 뒤 공백은 여기서 지운다(squeeze). 원문에 없는 위치는 오류.
  {역・역} = 역 이름 사전으로 자동 변환.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from check_tr import squeeze  # noqa: E402

D = dict(l.split('\t') for l in open(os.path.join(ROOT, 'work', 'text', 'stations.tsv'), encoding='utf-8').read().split('\n')[1:] if l)


def stations(v):
    """{역・역…} → 역 이름 사전(work/text/stations.tsv)으로 바꿈. 사전에 없으면 오류"""
    return re.sub(r'\{([^}]*)\}', lambda m: '・'.join(D[n] for n in m.group(1).split('・')), v)

if __name__ == '__main__':
    tr = {}
    for p in sorted(glob.glob(os.path.join(ROOT, 'work', 'text', 'kr', '*.txt'))):
        for ln in open(p, encoding='utf-8').read().split('\n'):
            if ln.strip():
                k, v = ln.split('\t', 1)
                tr[k] = squeeze(stations(v))
    seen = set()
    for p in sorted(glob.glob(os.path.join(ROOT, 'work', 'text', 'tsv', '*.tsv'))):
        L = open(p, encoding='utf-8').read().split('\n')
        out = [L[0]]
        for ln in L[1:]:
            if not ln:
                continue
            a = (ln.split('\t') + [''] * 5)[:5]
            if a[0] in tr:
                a[4] = tr[a[0]]; seen.add(a[0])
            out.append('\t'.join(a))
        open(p, 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')
    miss = set(tr) - seen
    assert not miss, '원문에 없는 위치 %s' % sorted(miss)[:5]
    print('채움', len(seen))
