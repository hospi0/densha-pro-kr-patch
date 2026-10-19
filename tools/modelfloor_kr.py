# -*- coding: utf-8 -*-
r"""모델 플로어 조작 안내(12px 글자) 한글화 (2026-10-18)
  그림 = TXS 9B91800(사각형 (768,384,64,128) → tpage 0x1C 의 v 128~255), 데이터 +0x800.
  조각 표 = 22번 파일 레코드 0x1BEC6~0x1C0F6(menu_kr.RUNS 끝 두 구간, 16 B [v<<8|u][x][y][w][h][tpage][clut][flag]).
  글자 = CLUT 7F70(값 1 흰·2 회색), 줄 v171·183·195 높이 12 → 세 줄(u56~255)을 비우고 낱말을 다시 배치, 버튼 안내와 같은 나눔고딕 ExtraBold(btn_kr.cover, 잉크 높이 10) → 진한 곳 1·옅은 곳 2, 왼쪽 1px 여백(▶ 와 띄움).
  (Galmuri11-Bold 비트맵판은 사용자가 버튼 글꼴로 바꾸라 함)
  セレクト/スタート = CLUT 7FB0(회색 계조, 8 흰) 32×24 칸(u36 v211) → 같은 글꼴 1~8 계조, ✕ 아이콘(v235~) 그대로.
  python tools/modelfloor_kr.py → work/gfx/prev_modelfloor.png (원본|한글, 화면 배치 두 벌)
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import btn_kr as BT
import menu_kr as MK

TXS = 0x9B91800
V0 = 128                                   # 사각형이 tpage 안 v 128 부터
H_INK = 10                                 # 원본 글자 잉크 높이 10(칸 12)
ROWS = (171, 183, 195)                     # 글자 줄(높이 12)
U0 = 56                                    # 비우는 범위 u56~255 (LR12·前進後退 조각은 안 쓰임)
GAP = 2

# 원래 칸 (u, v, w) → 한국어. 拡大縮小(88,171,48)는 拡大·縮小 칸을 겹쳐 쓰지만 한국어는 따로 그림
KR = {
    (184, 171, 24): '회전', (106, 195, 84): '원래 위치로', (112, 171, 24): '축소', (88, 171, 24): '확대',
    (136, 183, 60): '표시 끄기', (56, 195, 48): '다음 차량', (58, 183, 68): '메뉴로 가기',
    (196, 183, 12): '세로', (196, 195, 12): '가로', (88, 171, 48): '확대/축소',
}
PAIRS = ((0x1C036, 0x1C046, 'c'), (0x1C086, 0x1C096, 'l'))   # 縦+回転(패드 아래 가운데 맞춤) · 横+回転(왼쪽 맞춤): 뒤 조각 x = 앞 x + 앞 w + 2
SEL = (36, 211, 32)                        # セレクト(v211)·スタート(v223) 칸
SELKR = ('셀렉트', '스타트')


def recs_mf(ov):
    return [r for r in MK.records(ov) if r['o'] >= 0x1BEC6]


def load_px(pac):
    raw = np.frombuffer(pac[TXS + 0x800:TXS + 0x800 + 0x4000], np.uint8).reshape(128, 128)
    px = np.zeros((256, 256), np.uint8)
    px[V0:] = np.stack([raw & 15, raw >> 4], -1).reshape(128, 256)
    return px


def encode(px):
    p = px[V0:].reshape(128, 128, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def text_lv(text, w=256):
    """한 번에 그림 → 높이 12 칸 값 배열(0 투명·1 흰·2 회색), 폭 넘치면 가로 85% 까지 좁힘"""
    for xsc in (1.0, 0.95, 0.9, 0.85):
        c, _ = BT.cover(text, H_INK, xsc)
        if c.shape[1] <= w:
            break
    assert c.shape[1] <= w, text
    out = np.zeros((12, c.shape[1]), np.uint8)
    out[1:1 + H_INK] = np.where(c >= 0.5, 1, np.where(c >= 0.18, 2, 0))
    return out, c


def make(pac):
    """→ recs, 새 화소, 원본 화소, {레코드: (u, v, w)}, {레코드: x}"""
    ov = pac[MK.OV:MK.OV + 0x2F800]
    recs = recs_mf(ov); px = load_px(pac); orig = px.copy()
    # 글자 세 줄 비우기(값 1·2 만 — 다른 그림 칸은 이 범위에 없음)
    for v in ROWS:
        reg = px[v:v + 12, U0:]
        assert set(np.unique(reg)) <= {0, 1, 2}, v
        reg[:] = 0
    # 넓은 것부터 줄 순서대로 채우기
    imgs = {k: np.pad(text_lv(t)[0], ((0, 0), (1, 0))) for k, t in KR.items()}
    pos = {}; cur = {v: U0 for v in ROWS}
    for k in sorted(imgs, key=lambda k: -imgs[k].shape[1]):
        w = imgs[k].shape[1]
        v = next(v for v in ROWS if cur[v] + w <= 256)
        u = cur[v]; cur[v] = u + w + GAP
        px[v:v + 12, u:u + w] = imgs[k]
        pos[k] = (u, v, w)
    out = {}
    for r in recs:
        k = (r['u'], r['v'], r['w'])
        if r['clut'] == 0x7F70 and k in pos:
            out[r['o']] = pos[k]
    # セレクト/スタート(회색 계조 칸 1~8)
    u, v, w = SEL
    for i, t in enumerate(SELKR):
        reg = px[v + 12 * i:v + 12 * i + 12, u:u + w]
        reg[:] = 0
        c = text_lv(t, w)[1]
        x0 = (w - c.shape[1]) // 2
        reg[1:1 + H_INK, x0:x0 + c.shape[1]] = BT.levels(np.minimum(1, c * 1.6))   # 그대로면 회색으로 흐림
    # x: 단독 항목은 그대로, 두 조각 항목은 이어 붙임
    newx = {}
    byo = {r['o']: r for r in recs}
    for a, b, al in PAIRS:
        ra, rb = byo[a], byo[b]
        wa, wb = out[a][2], out[b][2]
        if al == 'c':
            mid = ra['x'] + (ra['w'] + rb['w']) / 2
            xa = int(round(mid - (wa + 2 + wb) / 2))
        else:
            xa = ra['x']
        newx[a] = xa; newx[b] = xa + wa + 2
    return recs, px, orig, out, newx


def patch_ov(ov, recs, out, newx):
    ov = bytearray(ov)
    for r in recs:
        o = r['o']
        if o in out:
            u, v, w = out[o]
            ov[o] = u; ov[o + 1] = v
            struct.pack_into('<H', ov, o + 6, w)
        if o in newx:
            struct.pack_into('<h', ov, o + 2, newx[o])
    return bytes(ov)


def apply(pac):
    """pac(bytearray) 에 그림·레코드 적용 → 바꾼 조각 수"""
    recs, px, _, out, newx = make(bytes(pac))
    pac[MK.OV:MK.OV + 0x2F800] = patch_ov(bytes(pac[MK.OV:MK.OV + 0x2F800]), recs, out, newx)
    enc = encode(px); a = TXS + 0x800
    pac[a:a + len(enc)] = enc
    return len(out)


def preview():
    import vram as V
    pac = open(MK.PAC, 'rb').read()
    recs, px, orig, out, newx = make(pac)
    v = V.vram(os.path.join(ROOT, 'work', 'shots', '12.sav'))

    def pal(c):
        cx, cy = (c & 0x3F) * 16, c >> 6; cc = v[cy, cx:cx + 16].astype(int)
        p = np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8)
        return p, cc == 0

    def screen(P, geo):
        """레코드 묶음 두 벌(0x1BEC6~, 0x1C016~)을 화면처럼 그림"""
        cvs = []
        for lo, hi in ((0x1BEC6, 0x1C016), (0x1C016, 0x1C106)):
            cv = np.zeros((50, 300, 3), np.uint8); cv[:] = (12, 22, 40)
            for r in recs:
                if not lo <= r['o'] < hi:
                    continue
                u, vv, x, w = geo(r)
                a = P[vv:vv + r['h'], u:u + w]
                p, tr = pal(r['clut']); im = p[a]; m = ~tr[a]
                X = x + 10; Y = r['y'] + 2
                sub = cv[Y:Y + a.shape[0], X:X + w]; sub[m[:sub.shape[0], :sub.shape[1]]] = im[m][:sub[m[:sub.shape[0], :sub.shape[1]]].shape[0]]
            cvs.append(cv)
        return np.concatenate(cvs, 0)

    old = screen(orig, lambda r: (r['u'], r['v'], r['x'], r['w']))
    new = screen(px, lambda r: (out[r['o']][0] if r['o'] in out else r['u'], out[r['o']][1] if r['o'] in out else r['v'],
                                newx.get(r['o'], r['x']), out[r['o']][2] if r['o'] in out else r['w']))
    both = np.concatenate([old, np.full((old.shape[0], 4, 3), 200, np.uint8), new], 1)
    I = Image.fromarray(both).resize((both.shape[1] * 3, both.shape[0] * 3), Image.NEAREST)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_modelfloor.png'); I.save(p)
    print('미리보기', p)
    for r in recs:
        if r['o'] in out or r['o'] in newx:
            print('%X' % r['o'], (r['u'], r['v'], r['x'], r['w']), '→', out.get(r['o']), newx.get(r['o']))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
