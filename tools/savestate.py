# -*- coding: utf-8 -*-
"""덕스테이션 세이브스테이트에서 «VRAM» 을 꺼낸다

  python tools/savestate.py <상태파일> [나온파일.png]

## 꼴
`DUCC` + u32 판번호 + 문자열들, 그 뒤로 **zstd 프레임**이 이어 붙는다.
프레임 1 = 화면 미리보기(256x192 RGBA) · 프레임 2 = 기계 상태(RAM·VRAM·레지스터)

기계 상태 안에서 VRAM(1024x512 u16 = 1 MB)은 **가로 이웃 차이가 가장 작은 곳**으로 찾는다
(그림이라 가로로 매끈하고, 그 옆의 RAM·레지스터는 난삽하다) → [[reference_duckstation_savestate_vram]]
"""
import os, struct, sys

MAGIC = b'\xb5\x2f\xfd'          # zstd 프레임 머리 (0xFD2FB528 리틀엔디안)
VW, VH = 1024, 512


def frames(path):
    d = open(path, 'rb').read()
    out, i = [], 0
    import zstandard as zstd
    while True:
        i = d.find(b'\x28\xb5\x2f\xfd', i)
        if i < 0:
            break
        try:
            out.append(zstd.ZstdDecompressor().decompress(d[i:], max_output_size=1 << 28))
        except Exception:
            pass
        i += 4
    return out


def vram(path):
    """(512,1024) u16 배열"""
    import numpy as np
    fs = frames(path)
    st = max(fs, key=len)
    need = VW * VH * 2
    best, bo = None, None
    for off in range(0, len(st) - need, 2048):
        a = np.frombuffer(st[off:off + need], dtype='<u2').reshape(VH, VW).astype(np.int32)
        s = int(np.abs(np.diff(a[::8, ::1], axis=1)).mean())
        if best is None or s < best:
            best, bo = s, off
    a = np.frombuffer(st[bo:bo + need], dtype='<u2').reshape(VH, VW)
    print('기계상태 %d B · VRAM 오프셋 0x%X (가로차 %d)' % (len(st), bo, best))
    return a


def png(a, out):
    import numpy as np
    from PIL import Image
    r = ((a & 0x1F) << 3).astype(np.uint8)
    g = (((a >> 5) & 0x1F) << 3).astype(np.uint8)
    b = (((a >> 10) & 0x1F) << 3).astype(np.uint8)
    Image.fromarray(np.dstack([r, g, b])).save(out)
    print('-> %s' % out)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    v = vram(sys.argv[1])
    import numpy as np
    np.save(os.path.splitext(sys.argv[2] if len(sys.argv) > 2 else 'vram')[0] + '.npy', v)
    png(v, sys.argv[2] if len(sys.argv) > 2 else 'vram.png')
