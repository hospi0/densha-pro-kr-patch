# -*- coding: utf-8 -*-
r"""MDEC BS 정지 영상 풀기·굽기 (2026-10-19)
  PAC 안 BS 프레임 = 머리 [u16 코드 수/2][u16 0x3800][u16 양자화][u16 판(3)] + 2바이트 단위 비트열, 섹터(0x800) 정렬.
  풀기 = 2016 B 씩 STR 섹터(2352 B, 머리 0x80010160·폭·높이)로 감싸 ffmpeg psxstr/mdec 로 (사용자가 준 gist: ffmpeg MDEC 디코더로 BS 를 푼다).
  굽기 = psxavenc -t sbs -v v3 -s WxH -a 크기 (원래 자리 크기 안).
  python tools/bs.py list                → PAC 안 BS 프레임 목록(위치·코드 수·양자화)
  python tools/bs.py dump <위치hex> W H  → work/gfx/bs/<위치>.png
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
FF = r'C:/claude/utils/ps1/ffmpeg.exe'
PSXAVENC = r'C:/claude/utils/ps1/psxavenc.exe'


def frames(pac):
    out = []
    for o in range(0, len(pac) - 8, 0x800):
        a, b, q, ver = struct.unpack_from('<HHHH', pac, o)
        if b == 0x3800 and ver in (2, 3) and 0 < q < 64:
            out.append((o, a, q, ver))
    return out


def wrap_str(bs, w, h):
    """BS 한 프레임 → STR 섹터들(2352 B, ffmpeg psxstr 가 읽는 꼴)"""
    n = (len(bs) + 2015) // 2016; out = bytearray()
    for k in range(n):
        sec = bytearray(2352)
        sec[0:12] = b'\x00' + b'\xff' * 10 + b'\x00'
        sec[15] = 2                                       # mode 2
        sub = bytes([1, 1, 0x48, 0])                       # 파일 1·채널 1·데이터+실시간
        sec[16:20] = sub; sec[20:24] = sub
        struct.pack_into('<IHHIIHH', sec, 0x18, 0x80010160, k, n, 1, len(bs), w, h)
        chunk = bs[k * 2016:(k + 1) * 2016]
        sec[0x38:0x38 + len(chunk)] = chunk
        out += sec
    return bytes(out)


def decode(bs, w, h):
    with tempfile.TemporaryDirectory() as d:
        s = os.path.join(d, 'f.str'); p = os.path.join(d, 'f.png')
        open(s, 'wb').write(wrap_str(bs, w, h) * 2)        # 프레임 두 번(디먹서가 첫 프레임을 버리는 경우 대비)
        subprocess.run([FF, '-hide_banner', '-loglevel', 'error', '-f', 'psxstr', '-i', s, '-frames:v', '1', '-y', p], check=True)
        return np.array(Image.open(p).convert('RGB'))


def encode(img, w, h, size, ver='v3'):
    """RGB 배열 → BS 프레임(크기 size 이하)"""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'in.png'); i = os.path.join(d, 'in.avi'); o = os.path.join(d, 'out.sbs')
        Image.fromarray(img).save(p)                        # psxavenc 는 PNG 한 장을 못 엶 → 1프레임 무손실 AVI
        subprocess.run([FF, '-hide_banner', '-loglevel', 'error', '-y', '-i', p, '-c:v', 'ffv1', '-r', '15', i], check=True)
        subprocess.run([PSXAVENC, '-q', '-t', 'sbs', '-v', ver, '-s', '%dx%d' % (w, h), '-I', '-a', str(size), i, o], check=True)
        return open(o, 'rb').read()


def frame_bytes(pac, o, nxt):
    return pac[o:nxt]


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    pac = open(PAC, 'rb').read()
    if sys.argv[1] == 'list':
        F = frames(pac)
        for k, (o, a, q, v) in enumerate(F):
            nxt = F[k + 1][0] if k + 1 < len(F) else o + 0x20000
            print('%X\t%d\t%d\tv%d\t자리 %d' % (o, a, q, v, nxt - o))
    elif sys.argv[1] == 'dump':
        o = int(sys.argv[2], 16); w, h = int(sys.argv[3]), int(sys.argv[4])
        F = [f[0] for f in frames(pac)]; nxt = next((x for x in F if x > o), o + 0x20000)
        im = decode(pac[o:nxt], w, h)
        d = os.path.join(ROOT, 'work', 'gfx', 'bs'); os.makedirs(d, exist_ok=True)
        Image.fromarray(im).save(os.path.join(d, '%X.png' % o)); print(os.path.join(d, '%X.png' % o), im.shape)
