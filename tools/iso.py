# -*- coding: utf-8 -*-
"""전차로 GO! 프로페셔널 사양 (PS1) `MODE2/2352` 이미지에서 ISO9660 파일 목록을 읽고 뽑는다

  python tools/iso.py list                  # 파일 목록
  python tools/iso.py dump <이미지> <출력폴더>   # 전 파일 추출

## 섹터 구조 (PS1 은 MODE 2)
```
0   12 B  SYNC  00 FF*10 00
12   4 B  헤더  (분,초,프레임,모드=2)
16   8 B  서브헤더 (파일,채널,서브모드,코딩  × 2벌)
─ Form 1 (서브모드 bit5=0) : 24 + **2048 B 데이터** + EDC 4 + ECC 276
─ Form 2 (서브모드 bit5=1) : 24 + **2324 B 데이터** + EDC 4      ← 동영상·XA 음성
```
⚠️**데이터 트랙 안에 Form 2 섹터가 섞여 있다** — 파일을 읽을 땐 Form 1 만 이어 붙이면
  안 되고, «그 파일의 LBA 부터 크기만큼» 읽되 섹터 종류를 봐야 한다.
  (일반 파일은 전부 Form 1 이므로 2048 씩 읽으면 된다)
"""
import io, json, os, struct, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SEC, USER = 2352, 2048
JP = "C:/claude/roms/ps/Densha de Go Professional Shiyou (Japan)/Densha de Go! Professional Shiyou (Japan).bin"
JP_SHA1 = None


def user(f, lba):
    f.seek(lba * SEC)
    s = f.read(SEC)
    return s[24:24 + USER]


def read(f, lba, size):
    out = bytearray()
    for i in range(-(-size // USER)):
        out += user(f, lba + i)
    return bytes(out[:size])


def walk(f, lba, size, prefix='', out=None, depth=0):
    """디렉터리 레코드를 재귀로 훑는다"""
    if out is None:
        out = []
    d = read(f, lba, size)
    p = 0
    while p < len(d):
        n = d[p]
        if n == 0:
            p = (p // USER + 1) * USER          # 섹터 경계로 넘어간다
            if p >= len(d):
                break
            continue
        flag = d[p + 25]
        nl = d[p + 32]
        nm = d[p + 33:p + 33 + nl]
        clba = struct.unpack_from('<I', d, p + 2)[0]
        csz = struct.unpack_from('<I', d, p + 10)[0]
        if nl == 1 and nm in (b'\x00', b'\x01'):
            pass
        elif flag & 2:
            if depth < 8:
                walk(f, clba, csz, prefix + '/' + nm.decode('ascii', 'replace'), out, depth + 1)
        else:
            out.append((prefix + '/' + nm.split(b';')[0].decode('ascii', 'replace'), clba, csz))
        p += n
    return out


def files(path=JP):
    f = open(path, 'rb')
    pvd = user(f, 16)
    assert pvd[1:6] == b'CD001', 'PVD 가 아니다'
    rec = pvd[156:156 + 34]
    lba = struct.unpack_from('<I', rec, 2)[0]
    size = struct.unpack_from('<I', rec, 10)[0]
    return sorted(walk(f, lba, size), key=lambda t: t[1]), f


def main(argv):
    if not argv or argv[0] == 'list':
        fl, f = files(argv[1] if len(argv) > 1 else JP)
        tot = 0
        for n, l, s in fl:
            print('%-28s LBA %7d  %12d B' % (n, l, s))
            tot += s
        print('\n파일 %d개 / 합계 %d B' % (len(fl), tot))
        json.dump(fl, io.open(os.path.join(ROOT, 'work', 'jp_files.json'), 'w',
                              encoding='utf-8'), ensure_ascii=False)
        return
    if argv[0] == 'dump':
        img, out = argv[1], argv[2]
        fl, f = files(img)
        os.makedirs(out, exist_ok=True)
        for n, l, s in fl:
            p = os.path.join(out, n.lstrip('/').replace('/', '_'))
            open(p, 'wb').write(read(f, l, s))
        print('%d개 -> %s' % (len(fl), out))


if __name__ == '__main__':
    main(sys.argv[1:])
