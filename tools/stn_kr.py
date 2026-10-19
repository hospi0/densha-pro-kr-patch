# -*- coding: utf-8 -*-
r"""주행 중 노선도 판 세로 역 이름 한글화 (2026-10-19)
  그림 = 노선별 TXS (384,256,64,128) 9장(4bpp 256×128), 칸 표 = PAC 0x7E5AC~(131) · 0x7EBD0~(61) 12 B [u][v][w][h][0 0 0 0][tpage C056][clut 4058].
  칸 → 노선(묶음 SEG, 칸이 노선별로 이어져 있음) · 이름 = tools/stn_scan.py 판독 + FIX(눈 검수) → work/stn_assign.json.
  한글 = work/text/stn_board_kr.tsv(역 사전과 같은 표기). 높이 = 12px×음절(타자와코 8번은 14px), 90px 넘으면 그 이름만 음절 간격을 줄임(사용자 결정).
  노선 그림마다 이름 칸을 다시 배치(열 폭 = 원래 w, 간격 15·17) — 원래 이름 칸 화소만 지우고 그 밖 그림(왼쪽 위 CLUT·아이콘, 0번 오른쪽 위 무늬)은 보존.
  꼴: 상자형(0~7번) = 바탕 7(흰) + 글자 1(검정) · 8번 = 바탕 없음, 글자 3 + 오른쪽 아래 그림자 2.
  python tools/stn_kr.py → work/gfx/prev_stn_N.png (노선 그림 원본|한글)
"""
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import pixfont as PF  # noqa: E402

PAC = os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC')
TXS = [0x9269800, 0x926E800, 0x9273800, 0x9278800, 0x927D800, 0x9282800, 0x9287800, 0x928C800, 0x9291800]
SEG = [(0, 28, 1), (29, 37, 0), (38, 47, 2), (48, 49, 0), (50, 78, 5), (79, 91, 3), (92, 109, 5), (110, 142, 7),
       (143, 152, 4), (153, 162, 6), (163, 181, 8)]
FIX = {32: '大張野', 53: '嵯峨', 74: '大正', 87: '大正', 104: '大正', 110: '博多', 112: '箱崎', 117: '千早', 118: '福間',
       119: '東福間', 120: '東郷', 121: '赤間', 122: '教育大前', 123: '海老津', 124: '遠賀川', 125: '水巻', 126: '折尾',
       127: '黒崎', 128: '八幡', 129: '枝光', 132: '西小倉', 133: '小倉', 134: '折尾', 135: '折尾', 136: '黒崎', 137: '八幡',
       138: '枝光', 139: '戸畑', 140: '九州工大前', 141: '西小倉', 142: '小倉', 145: '大森', 156: '大崎', 159: '浜松町',
       160: '新橋', 163: '大曲', 164: '北大曲', 165: '羽後四ツ屋', 166: '鑓見内', 167: '羽後長野', 168: '鶯野', 169: '角館',
       170: '生田', 171: '神代', 172: '刺巻', 173: '田沢湖', 174: '田沢湖', 175: '田沢湖', 176: '赤渕', 177: '春木場',
       178: '雫石', 179: '小岩井', 180: '大釜', 181: '盛岡'}
MAXH = 90
PITCH = 11                                  # 상자형 음절 간격(12 면 5번 쿄토·사가노 그림에 자리 모자람)
PREV_PAL = np.array([[255, 0, 255], [8, 8, 8], [48, 48, 48], [80, 80, 80], [112, 112, 112], [144, 144, 144], [184, 184, 184],
                     [248, 248, 248]] + [[0, 200, 0]] * 8, np.uint8)


def img(pac, off):
    a = off + 0x800; raw = np.frombuffer(pac[a:a + 64 * 128 * 2], np.uint8).reshape(128, 128)
    return np.stack([raw & 15, raw >> 4], -1).reshape(128, 256).copy()


def encode(px):
    p = px.reshape(128, 128, 2)
    return (p[..., 0] | (p[..., 1] << 4)).astype(np.uint8).tobytes()


def assign():
    A = json.load(open(os.path.join(ROOT, 'work', 'stn_assign.json'), encoding='utf-8'))
    for r in A:
        if r['i'] in FIX:
            r['name'] = FIX[r['i']]
    return A


def kr_names():
    return dict(l.split('\t') for l in open(os.path.join(ROOT, 'work', 'text', 'stn_board_kr.tsv'), encoding='utf-8').read().split('\n')[1:] if l)


def glyph_col(text, w, pitch_full):
    """세로 이름 → 0/1 배열(w × h). 음절 간격 pitch_full(12/14), 90 넘으면 줄임 + 작은 갈무리"""
    m = len(text); h = min(MAXH, pitch_full * m); pitch = h / m
    size = 14 if pitch >= 14 else 11 if pitch >= 11 else 9 if pitch >= 10 else 7
    out = np.zeros((h, w))
    for i, ch in enumerate(text):
        g = PF.cell(ch, size, bold=(size == 11))             # 원본 한자처럼 굵게(갈무리11 Bold) · 고정 칸(잉크로 자르면 좁은 글자가 비뚤어짐)
        gh, gw = g.shape
        if gw > w:
            g = g[:, :w]; gw = w
        y0 = int(round(i * pitch + (pitch - gh) / 2)); x0 = (w - gw) // 2
        y0 = max(0, min(h - gh, y0))
        out[y0:y0 + gh, x0:x0 + gw] = np.maximum(out[y0:y0 + gh, x0:x0 + gw], g)
    return out


def col_lists():
    d = {}
    for l in open(os.path.join(ROOT, 'work', 'text', 'stn_cols.tsv'), encoding='utf-8').read().split('\n')[1:]:
        if l:
            t, u, n = l.split('\t'); d[(int(t), int(u))] = n.split(' ')
    return d


def users():
    """칸 레코드마다 [(노선 그림, 역 이름)] — 같은 칸을 여러 노선이 다른 역으로 씀(80,0 = 峰吉川·有楽町·森ノ宮·丹波口).
    폭 14 칸: 노선 0~6 중 판독 점수 0.36↑ 이고 그 열(u) 이름 목록에 있는 것 + 묶음(SEG) 판정. 폭 12 = 7번, 16 = 8번."""
    S = {s['i']: s for s in json.load(open(os.path.join(ROOT, 'work', 'stn_scan.json'), encoding='utf-8'))}
    A = {r['i']: r for r in assign()}; C = col_lists(); out = []
    for i, r in A.items():
        s = S[i]; us = {r['t']: r['name']}
        for t in range(9):                                          # 칸 폭이 달라도 다른 노선이 쓸 수 있음((64,0,12,60) = 7번 九州工大前·3번 大阪城公園)
            sc, nm = s['per'][t]
            if t == r['t'] or sc < 0.42:                                  # 0.36 은 헛짝이 많아 자리 모자람
                continue
            if t >= 7 or nm in C.get((t, r['u']), []):              # 7·8번은 열 u 목록이 부정확 → 이름 목록만
                us[t] = nm
        out.append(dict(r, users=us))
    return out


def layout(pac):
    U = users(); K = kr_names()
    PX = [img(pac, off) for off in TXS]
    prot = []
    for t, px in enumerate(PX):
        p = np.zeros(px.shape, bool); p[:43, :16] = True
        if t == 0:
            p[:24, 190:] = True                                     # 0번 오른쪽 위 무늬(이름 아님)
        prot.append(p)
    # 칸 묶기: 같은 (u,v,w,h, 사용 노선·이름) = 한 자리
    groups = []                                                     # [(사용 노선→이름, 폭, 레코드들)] — 겹치는 노선에서 이름이 같으면 한 자리로 합침
    def isbox8(r):
        return 8 in r['users'] and (PX[8][r['v']:r['v'] + r['h'], r['u']:r['u'] + r['w']] == 7).mean() > 0.15        # 盛岡 = 0.22(획이 빽빽함)
    for r in sorted(U, key=lambda r: -len(r['users'])):
        for g in groups:
            if g[1] == r['w'] and g[3] == isbox8(r) and all(g[0].get(t, n) == n for t, n in r['users'].items()):
                g[0].update(r['users']); g[2].append(r); break
        else:
            groups.append([dict(r['users']), r['w'], [r], isbox8(r)])
    items = []
    for us, w, rs, box8 in groups:
        gl = {t: glyph_col(K[n], w, 13 if t == 8 else PITCH) for t, n in us.items()}
        h = max(g.shape[0] for g in gl.values())
        items.append(dict(rs=rs, users=us, w=w, h=h, gl=gl, box8=box8))                # 8번도 흰 상자 이름(盛岡·大曲 강조)은 상자형
    items.sort(key=lambda it: (-it['h'], -len(it['users'])))                # 긴 것부터(열 채우기)
    occ = [p.copy() for p in prot]
    for it in items:
        w, h, ts = it['w'], it['h'], list(it['users'])
        done = False
        free = np.ones((128, 256), bool)
        for t in ts:
            free &= ~occ[t]
        ii = np.zeros((129, 257), int); ii[1:, 1:] = (~free).cumsum(0).cumsum(1)       # 적분 영상으로 빈 사각형 찾기
        for u in range(0, 256 - w + 1):                                                 # 1px 단위(왼쪽부터 채움)
            for v in range(0, 128 - h + 1):
                if ii[v + h, u + w] - ii[v, u + w] - ii[v + h, u] + ii[v, u] == 0:
                    for t in ts:
                        occ[t][v:v + h + 1, u:u + w + 1] = True                        # 오른쪽 1열·아래 1줄 띄움
                    it['at'] = (u, v); done = True; break
            if done:
                break
        assert done, '자리 없음 %s' % it['users']
    res = {}
    for t, px in enumerate(PX):
        new = px.copy(); new[~prot[t]] = 0                          # 이름 그림 전부 지움(안 쓰는 옛 이름 포함)
        res[t] = [px, new]
    for it in items:
        u, v = it['at']
        for t, g in it['gl'].items():
            new = res[t][1]; gh = g.shape[0]
            reg = new[v:v + it['h'], u:u + it['w']]
            if t == 8 and not it['box8']:
                gg = np.zeros((it['h'], it['w'])); gg[:gh] = g
                sh = np.zeros_like(gg); sh[1:, 1:] = gg[:-1, :-1]
                reg[sh > 0] = 2; reg[gg > 0] = 3
            else:
                sub = reg[:gh]; sub[:] = 7; sub[g > 0] = 1              # 상자는 글자 길이만큼·위 맞춤(칸 나머지는 투명)
    return res, items


def apply(pac):
    res, items = layout(bytes(pac)); n = 0
    for t, (px, new) in res.items():
        enc = encode(new); a = TXS[t] + 0x800; pac[a:a + len(enc)] = enc
    for it in items:
        u, v = it['at']
        for r in it['rs']:
            pac[r['o']:r['o'] + 4] = bytes([u, v, it['w'], it['h']]); n += 1
    return n


def preview():
    pac = open(PAC, 'rb').read(); res, items = layout(pac)
    rows = []
    for t in range(len(TXS)):
        px, new = res[t][0], res[t][1]
        a = PREV_PAL[px]; b = PREV_PAL[new]
        rows.append(np.concatenate([a, np.full((128, 4, 3), 255, np.uint8), b], 1))
        rows.append(np.full((4, 516, 3), 0, np.uint8))
    sh = np.concatenate(rows, 0)
    for k in range(0, sh.shape[0], 132 * 3):
        p = os.path.join(ROOT, 'work', 'gfx', 'prev_stn_%d.png' % (k // (132 * 3)))
        part = sh[k:k + 132 * 3]
        Image.fromarray(part).resize((part.shape[1] * 2, part.shape[0] * 2), Image.NEAREST).save(p)
    print('미리보기 work/gfx/prev_stn_0~2.png')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    preview()
