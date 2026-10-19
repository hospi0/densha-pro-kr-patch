# -*- coding: utf-8 -*-
r"""번역용 TSV 나누기 — work/text/src.tsv → work/text/tsv/ko_NN.tsv (20KB 단위, 자투리는 앞 묶음에 합침) (2026-10-18)
  열: 위치 · 파일 · 예산(바이트) · 원문 · 번역(빈칸)
  ⚠️번역문 규칙: @1 @2 @3 @r ?0 ?1 은 원문과 같은 개수·순서로 · 예산 = 원문 바이트(한글 1자 2바이트, 반각 1바이트)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = 20 * 1024

if __name__ == '__main__':
    lines = open(os.path.join(ROOT, 'work', 'text', 'src.tsv'), encoding='utf-8').read().split('\n')
    head = lines[0] + '\t번역'; rows = [l + '\t' for l in lines[1:] if l]
    chunks = [[]]; size = 0
    for r in rows:
        n = len((r + '\n').encode('utf-8'))
        if size + n > LIMIT and chunks[-1]:
            chunks.append([]); size = 0
        chunks[-1].append(r); size += n
    if len(chunks) > 1 and sum(len(r.encode('utf-8')) for r in chunks[-1]) < LIMIT // 4:
        chunks[-2] += chunks.pop()
    d = os.path.join(ROOT, 'work', 'text', 'tsv'); os.makedirs(d, exist_ok=True)
    for i, c in enumerate(chunks, 1):
        p = os.path.join(d, 'ko_%02d.tsv' % i)
        open(p, 'w', encoding='utf-8', newline='\n').write(head + '\n' + '\n'.join(c) + '\n')
        print(p, len(c), os.path.getsize(p))
