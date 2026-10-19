# -*- coding: utf-8 -*-
"""실행파일에서 **도달 불가능한 코드 구역**을 찾는다 (글리프 자리 확보용)

  python tools/deadcode.py

## 방법
진입점 집합에서 출발해 흐름을 따라가며 «닿은 명령»을 표시하고, 남는 구역을 낸다.
진입점 = ① `PC`(헤더) ② 모든 `jal`/`j` 목적지 ③ 모든 분기 목적지
        ④ **데이터 안 u32 상수 중 코드 범위를 가리키는 것**(함수 포인터)
지연슬롯 때문에 `jr`/`j` 뒤 한 명령은 실행되므로 그것까지 표시한다.

⚠️**간접 호출(레지스터로 계산한 주소)은 원리상 안 보인다.**
  ⇒ 여기서 나온 「죽은 구역」은 **후보**일 뿐이다. 반드시 실기로 하나씩 확인할 것.
  → [[feedback_mips_injection_traps]] 의 «자리는 실증으로만» 원칙
"""
import os, struct, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
EXE = os.path.join(ROOT, 'work', 'disc', 'SLPM_863.78')
BASE = 0x8000F800
LO, HI = 0x800, 0x04A800                 # 코드로 볼 파일 범위
BR = {0x04, 0x05, 0x06, 0x07}            # beq bne blez bgtz


def main():
    d = open(EXE, 'rb').read()
    n = (HI - LO) // 4
    seen = bytearray(n)
    ent = set()

    def add(a):
        o = a - BASE
        if LO <= o < HI and o % 4 == 0:
            ent.add(o)

    add(struct.unpack_from('<I', d, 0x10)[0])                     # PC
    for i in range(0, len(d) - 4, 4):                             # 데이터 안 포인터 상수
        v = struct.unpack_from('<I', d, i)[0]
        if 0x80000000 <= v < 0x80200000:
            add(v)
    for i in range(LO, HI, 4):                                    # 모든 분기·호출 목적지
        w = struct.unpack_from('<I', d, i)[0]
        op = w >> 26
        if op in (0x02, 0x03):
            add(((i + BASE) & 0xF0000000) | ((w & 0x3FFFFFF) << 2))
        elif op in BR or (op == 0x01):
            s = (w & 0xFFFF) - (0x10000 if w & 0x8000 else 0)
            add(i + BASE + 4 + s * 4)

    work = list(ent)
    while work:
        o = work.pop()
        while LO <= o < HI:
            k = (o - LO) // 4
            if seen[k]:
                break
            seen[k] = 1
            w = struct.unpack_from('<I', d, o)[0]
            op, fn = w >> 26, w & 63
            if op == 0 and fn in (0x08, 0x09):                    # jr / jalr
                if (o - LO) // 4 + 1 < n:
                    seen[(o - LO) // 4 + 1] = 1                   # 지연슬롯
                break
            if op == 0x02:                                        # j
                if (o - LO) // 4 + 1 < n:
                    seen[(o - LO) // 4 + 1] = 1
                t = ((o + BASE) & 0xF0000000) | ((w & 0x3FFFFFF) << 2) - BASE
                break
            o += 4

    runs, i = [], 0
    while i < n:
        if not seen[i]:
            j = i
            while j < n and not seen[j]:
                j += 1
            if (j - i) * 4 >= 128:
                runs.append((LO + i * 4, LO + j * 4))
            i = j
        else:
            i += 1
    tot = sum(e - s for s, e in runs)
    print('코드 범위 0x%06X~0x%06X (%d KB)' % (LO, HI, (HI - LO) // 1024))
    print('닿지 않은 128 B 이상 구역 **%d개 / 합계 %d B (%.1f KB)**' % (len(runs), tot, tot / 1024))
    runs.sort(key=lambda t: -(t[1] - t[0]))
    for s, e in runs[:20]:
        print('   0x%06X ~ 0x%06X  %6d B  (메모리 0x%08X)' % (s, e, e - s, s + BASE))


if __name__ == '__main__':
    main()
