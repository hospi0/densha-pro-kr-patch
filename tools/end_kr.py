# -*- coding: utf-8 -*-
r"""운행 종료 화면 한글화 (2026-10-19, 미작업 «운행종료» 스샷 2장 — work/shots/end_eval.sav · end_record.sav)
  ① 평가표 TXS 9798800 (384,0 64×272: 4bpp 256×256 + CLUT 두 줄) — 사본 없음.
     CLUT 줄 256 = 회색 계조(0 진함 … F 흰), 줄 257 = 도장 빨강 계조(0 투명, 1 연함 … F 진함).
     표 선: 가로 v46·47·78·79·110·111·142·143·174·175·206·207, 세로 u9·147·164·246. 글자 = 나눔고딕 ExtraBold 계조(원본도 계조 글꼴).
     도장 優·良·可·不可(u 16k, 16×32) → 우·양·가·불가(세로 두 줄), 테두리 타원 다시 그림.
  ② 승무 기록 화면 TXS 97A2000 사각형 0 (512,0 256×256 → 4bpp 1024 폭, 글자는 u0~511) — 사본 없음.
     持ち時間(C 채움·E 둘레, CLUT 3CE8 초록) · 乗務記録 판(3 바탕·4 글자·1 그림자) · 選択/決定 안내(CLUT 3D28 1~4 계조) · 秒 칸(1 바탕·A 글자) ·
     이름 입력 칸 キハ…系·形·終(1 바탕·7 글자) · 노선 이름 5줄·차종 系/形(4 바탕·6 글자·1 그림자) · 次に進む/メニューに戻る(8 글자·4 그림자).
  사용자(2026-10-19): 그림 글자는 «깔끔하게 단색» — 계조 줄무늬 안 씀.
  python tools/end_kr.py → work/gfx/prev_end.png (원본|한글, 스테이트 CLUT 실제 색)
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import bonus_kr as BK  # noqa: E402
import btn_kr as BT  # noqa: E402

PAC = BK.PAC
EVAL = 0x9798800
REC = 0x97A2000
G11B, G9 = (11, True, 1), (9, False, 1)


def load4(pac, off, w, h):
    a = off + 0x800; b = np.frombuffer(pac[a:a + w * h * 2], np.uint8).reshape(h, w * 2)
    return a, np.stack([b & 15, b >> 4], -1).reshape(h, w * 4).copy()


def ink(t, font, wide=1):
    m = BK.render(t, font, 1000)
    return np.repeat(m, wide, 1) if wide > 1 else m


def put(reg, m, fill, shadow=None, ring=None, al='c', dy=0):
    """reg 가운데(또는 왼쪽)에 1비트 글자 m — 그림자(+1,+1)·둘레 1px 단색"""
    H, W = reg.shape; gh, gw = m.shape
    pad = 1 if ring is not None else 0
    x0 = pad if al == 'l' else (W - gw - (1 if shadow is not None else 0)) // 2
    y0 = (H - gh - (1 if shadow is not None else 0)) // 2 + dy
    assert 0 <= x0 and x0 + gw <= W and 0 <= y0 and y0 + gh <= H, ('넘침', gw, gh, W, H)
    big = np.zeros((H, W), bool); big[y0:y0 + gh, x0:x0 + gw] = m
    if ring is not None:
        r = BK.dil(big) & ~big; reg[r] = ring
    if shadow is not None:
        s = BK.shift(big, 1, 1) & ~big; reg[s] = shadow
    reg[big] = fill


def cov(t, h_ink, w):
    for xsc in (1.0, 0.95, 0.9, 0.85, 0.8):
        c, _ = BT.cover(t, h_ink, xsc)
        if c.shape[1] <= w:
            return c
    raise SystemExit('넘침 %r %d > %d' % (t, c.shape[1], w))


def put_cov(reg, c, idx):
    """계조 덮임 c → idx(덮임) 함수 값으로, 가운데"""
    H, W = reg.shape; gh, gw = c.shape
    y0 = (H - gh) // 2; x0 = (W - gw) // 2
    sub = reg[y0:y0 + gh, x0:x0 + gw]; m = c > 0.06
    sub[m] = idx(c[m])


# ── ① 평가표
ROWS = [(48, 78, '다이아 정확성'), (80, 110, '브레이크(승차감)'), (112, 142, '정지 위치 정확성'), (144, 174, '신호·표지 준수'), (176, 206, '안전 배려')]
STAMPS = ['우', '양', '가', '불가']


def stamp(text):
    SS = 8; W, H = 16 * SS, 32 * SS
    im = Image.new('L', (W, H)); d = ImageDraw.Draw(im)
    d.ellipse((int(0.6 * SS), int(0.6 * SS), int(15.4 * SS), int(31.4 * SS)), outline=255, width=int(1.5 * SS))
    lines = list(text) if len(text) > 1 else [text]
    f = ImageFont.truetype(BT.FONT, (11 if len(lines) == 1 else 9) * SS)
    hs = (10 if len(lines) == 1 else 11) * SS; y = (H - hs * len(lines)) // 2
    for ch in lines:
        l, t, r, b = f.getbbox(ch)
        d.text(((W - (r - l)) // 2 - l, y + (hs - (b - t)) // 2 - t), ch, font=f, fill=255); y += hs
    return np.array(im.resize((16, 32), Image.BOX)).astype(float) / 255


def make_eval(px):
    px = px.copy()
    gray = lambda c: (15 - np.rint(c * 15)).astype(np.uint8)          # 0 진함 … F 흰
    for k, t in enumerate(STAMPS):
        c = stamp(t); reg = px[0:32, 16 * k:16 * k + 16]
        reg[:] = 0; m = c > 0.06; reg[m] = np.clip(np.rint(c[m] * 15), 1, 15).astype(np.uint8)
    reg = px[0:42, 64:224]; reg[:] = 15                                 # 제목(원본 v7~38)
    put_cov(reg, cov('운전 평가', 24, 150), gray)
    for y0, y1, t in ROWS:
        reg = px[y0:y1, 12:146]; reg[:] = 15                             # u9 세로선 계조·둥근 모서리가 u10~11, u147 선 계조가 u146 → 칠하지 않음(사용자 «테두리 모서리 깨짐»)
        put_cov(reg, cov(t, 17, 130), gray)
    reg = px[210:254, 60:176]; reg[:] = 15; put_cov(reg, cov('종합 평가', 22, 112), gray)
    reg = px[210:254, 196:250]; reg[:] = 15; put_cov(reg, cov('점', 24, 50), gray)
    return px


# ── ② 승무 기록
TILES = [((0, 96), '키하'), ((32, 96), '사하'), ((64, 96), '쿠하'), ((96, 96), '모하'), ((128, 96), '쿠로'), ((160, 96), '키로'),
         ((192, 96), '사로'), ((224, 96), '계'), ((0, 112), '형'), ((64, 112), '끝')]
GUIDE = [((171, 182), (3, 36), '선택'), ((171, 182), (50, 112), '브레이크'),
         ((187, 198), (3, 36), '결정'), ((187, 198), (68, 116), '버튼'),
         ((203, 214), (3, 36), '선택'), ((203, 214), (76, 126), '방향키'),
         ((219, 230), (3, 36), '결정'), ((219, 230), (68, 116), '버튼')]
ROUTES = ['케이힌토호쿠선', '호쿠호쿠선', '아키타 신칸센', '오사카 순환선', '카고시마 본선']
SUFFIX = {80: '계', 96: '형', 112: '계', 128: '계', 144: '계', 160: '계', 176: '계', 192: '계'}   # 차종 줄 v → 마지막 글자(系/形)


def make_rec(px):
    px = px.copy()
    reg = px[127:141, 64:192]; reg[np.isin(reg, [12, 14])] = 0           # 持ち時間 → 남은 시간(HUD 와 같은 말)
    put(reg, ink('남은 시간', G11B), 12, ring=14)
    reg = px[148:165, 70:218]; reg[:] = 3                                # 乗務記録 판 안쪽(CLUT 3BE8: 1 투명·3 진회색·6 밝은 회색)
    put(reg, ink('승무 기록', G11B, 2), 6, shadow=1)
    for (y0, y1), (x0, x1), t in GUIDE:                                  # 안내(계조 1~4 = 원본 꼴)
        reg = px[y0:y1, x0:x1]; reg[:] = 0
        c = cov(t, 10, x1 - x0); gh, gw = c.shape
        sub = reg[(y1 - y0 - gh) // 2:(y1 - y0 - gh) // 2 + gh, 0:gw]; m = c > 0.12
        sub[m] = np.clip(np.rint(c[m] * 4), 1, 4).astype(np.uint8)
    reg = px[234:250, 3:37]; reg[:] = 1; put(reg, ink('초', G11B, 2), 10)
    for (u, v), t in TILES:                                              # 이름 입력 칸(틀 v+0~2·13~15 그대로)
        reg = px[v + 3:v + 13, u + 3:u + 29]; reg[:] = 1; put(reg, ink(t, G9), 7)
    for k, t in enumerate(ROUTES):                                       # 노선 이름(회색 판 4)
        reg = px[16 * k:16 * k + 16, 256:418]; reg[:] = 4; put(reg, ink(t, G11B), 6, shadow=1)
    for v, t in SUFFIX.items():                                          # 차종 끝 글자 칸만
        band = px[v:v + 16, 256:424]
        cols = [x for x in range(band.shape[1]) if (band[:, x] != 4).any()]
        segs = []; s = cols[0]; p = cols[0]
        for x in cols[1:]:
            if x - p > 3:
                segs.append((s, p)); s = x
            p = x
        segs.append((s, p))
        a, b = segs[-1]
        a = b - 29                                                       # 마지막 글자 칸 폭 ≈30(形 은 开+彡 두 덩어리라 덩어리로 자르면 00 까지 먹음)
        reg = band[:, a:b + 2]; reg[:] = 4; put(reg, ink(t, G11B, 2), 6, shadow=1)
    reg = px[208:224, 256:302]; reg[reg != 0] = 0; put(reg, ink('다음으로', G9), 8, shadow=4)
    reg = px[208:224, 302:378]; reg[reg != 0] = 0; put(reg, ink('메뉴로 돌아가기', G9), 8, shadow=4)
    return px


def apply(pac):
    a, px = load4(bytes(pac), EVAL, 64, 256)
    enc = BK.encode(make_eval(px)); pac[a:a + len(enc)] = enc
    a, px = load4(bytes(pac), REC, 256, 256)
    enc = BK.encode(make_rec(px)); pac[a:a + len(enc)] = enc
    return 2


def preview():
    import vram as V
    pac = open(PAC, 'rb').read()
    def clut_state(sav, c):
        v = V.vram(os.path.join(ROOT, 'work', 'shots', sav)); cc = v[c >> 6, (c & 63) * 16:(c & 63) * 16 + 16].astype(int)
        p = np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8); p[cc == 0] = (70, 40, 90); return p
    def clut_txs(row):
        a = EVAL + 0x800 + row * 128; cc = np.frombuffer(pac[a:a + 32], '<u2').astype(int)
        p = np.stack([(cc & 31) << 3, ((cc >> 5) & 31) << 3, ((cc >> 10) & 31) << 3], -1).astype(np.uint8)
        if row == 257:
            p[0] = (255, 255, 255)
        return p
    _, e0 = load4(pac, EVAL, 64, 256); e1 = make_eval(e0)
    _, r0 = load4(pac, REC, 256, 256); r1 = make_rec(r0)
    rows = []
    def pair(a, b):
        return np.concatenate([a, np.full((a.shape[0], 4, 3), 255, np.uint8), b], 1)
    g = clut_txs(256); s = clut_txs(257)
    for P in (e0, e1):
        pass
    ev = [pair(np.concatenate([s[e0[0:32, 0:64]], g[e0[0:32, 64:256]]], 1), np.concatenate([s[e1[0:32, 0:64]], g[e1[0:32, 64:256]]], 1)),
          pair(g[e0[32:256]], g[e1[32:256]])]
    pg = clut_state('end_record.sav', 0x3BE8); pc = clut_state('end_record.sav', 0x3CE8); pd = clut_state('end_record.sav', 0x3D28)
    tl = clut_state('end_record.sav', 0x1E60)
    rc = [pair(tl[r0[94:128, 0:256]], tl[r1[94:128, 0:256]]), pair(pc[r0[126:142, 60:192]], pc[r1[126:142, 60:192]]),
          pair(pg[r0[142:170, 60:226]], pg[r1[142:170, 60:226]]), pair(pd[r0[168:232, 0:130]], pd[r1[168:232, 0:130]]),
          pair(pg[r0[232:252, 0:40]], pg[r1[232:252, 0:40]]), pair(pg[r0[0:226, 256:424]], pg[r1[0:226, 256:424]])]
    Wd = 1560; cv = Image.new('RGB', (Wd, 3200), (25, 25, 25)); X = Y = RH = 0
    for im in ev + rc:
        I = Image.fromarray(im).resize((im.shape[1] * 3, im.shape[0] * 3), Image.NEAREST)
        if X + I.width > Wd:
            X = 0; Y += RH + 12; RH = 0
        cv.paste(I, (X, Y)); X += I.width + 12; RH = max(RH, I.height)
    p = os.path.join(ROOT, 'work', 'gfx', 'prev_end_2.png')
    cv.crop((0, 0, Wd, Y + RH + 4)).save(p); print('미리보기', p)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
