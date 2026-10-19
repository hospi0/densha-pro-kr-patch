# -*- coding: utf-8 -*-
r"""빌드: 원본 BIN 복사 → TRAIN.PAC 평문 제자리 한글 → 바뀐 섹터 EDC/ECC (2026-10-18)
  python tools/build.py [출력 bin]   (기본 work/build/<원본 이름>.bin)
  글꼴 = 한글 BIOS(사용자 결정 2026-10-18): 한글 음절 = BIOS 한자 칸 SJIS 0x889F 부터 완성형(KS X 1001) 순서.
    i = 완성형 번호((행-0xB0)*94 + (열-0xA1)) → 선형 = 94 + i → 앞 0x88 + 선형//188, 뒤 = 0x40+t (t<63) / 0x41+t.
    (스샷 4 「맒뷔」 = 現在 로 검산)
  그 밖 글자 = cp932 · 토큰 @1 @2 @3 @r ?0 ?1 = ASCII 그대로.
  ⛔한자 칸(0x889F~)을 쓰는 일본어가 남으면 한글로 깨져 보인다 → BIOS 글꼴 평문은 전부 옮길 것.
  쓰기 = 원문 자리(예산 = 원문 바이트), 남는 바이트 0. 번역 없는 행은 그대로.
"""
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import cdecc  # noqa: E402
import check_tr as C  # noqa: E402

SRC_BIN = r'C:\claude\roms\ps\Densha de Go Professional Shiyou (Japan)\Densha de Go! Professional Shiyou (Japan).bin'
PAC_LBA, SEC, USER = 175, 2352, 2048
EXE_LBA = 25                                                 # SLPM_863.78 (work/jp_files.json)
PAC_SIZE = 194654208


def hangul_code(ch):
    k = ch.encode('euc-kr')                                   # 완성형 밖 글자는 여기서 실패 = 빌드 실패
    row, col = k[0] - 0xB0, k[1] - 0xA1
    assert 0 <= row < 25 and 0 <= col < 94, '완성형 한글 아님 %r' % ch
    lin = 94 + row * 94 + col
    lead, t = 0x88 + lin // 188, lin % 188
    return bytes([lead, 0x40 + t if t < 63 else 0x41 + t])


def encode(s):
    out = bytearray(); i = 0
    while i < len(s):
        if C.TOK.match(s, i):
            out += s[i:i + 2].encode('ascii'); i += 2; continue
        c = s[i]
        out += hangul_code(c) if '가' <= c <= '힣' else c.encode('cp932')
        i += 1
    return bytes(out)


def rows():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'work', 'text', 'tsv', '*.tsv'))):
        for ln in open(p, encoding='utf-8').read().split('\n')[1:]:
            if not ln:
                continue
            a = (ln.split('\t') + [''] * 5)[:5]
            if a[4]:
                out.append(a)
    return out


def all_src():
    return [l.split('\t') for l in open(os.path.join(ROOT, 'work', 'text', 'src.tsv'), encoding='utf-8').read().split('\n')[1:] if l]


def repack(pac, orig, tr):
    """포인터 표로 가리키는 문자열 영역(C.POOLS)을 번역문으로 다시 채우고 포인터를 고친다.
    같은 바이트열은 한 번만 · 1바이트 정렬 · 영역을 넘으면 실패. 포인터 = 파일 범위 안 u32 == 적재 주소 + 옛 위치."""
    import struct
    rep = {}
    for fi, (a, b, base, (flo, fhi)) in C.POOLS.items():
        rs = sorted((int(r[0], 16), r) for r in all_src() if a <= int(r[0], 16) < b)
        ptr = {}
        for i in range(flo, fhi, 4):
            if a <= i < b:
                continue
            v = struct.unpack_from('<I', orig, i)[0] - base + flo
            if a <= v < b:
                ptr.setdefault(v, []).append(i)
        blob = bytearray(); at = {}; newpos = {}
        for o, r in rs:
            assert o in ptr, '포인터 없는 문자열 %X' % o
            s = tr.get(r[0])
            e = encode(C.squeeze(s)) if s else r[3].encode('cp932')
            if e not in at:
                at[e] = a + len(blob); blob += e + b'\0'
            newpos[o] = at[e]
        rep[fi] = (len(blob), b - a)
        assert len(blob) <= b - a, '%s번 영역 넘침 %d > %d B' % (fi, len(blob), b - a)
        pac[a:b] = blob + bytes(b - a - len(blob))
        for o, locs in ptr.items():
            for i in locs:
                pac[i:i + 4] = struct.pack('<I', newpos[o] - flo + base)
    return rep


def main(out_bin):
    pac = bytearray(open(os.path.join(ROOT, 'work', 'disc', 'TRAIN.PAC'), 'rb').read())
    orig = bytes(pac); spans = []; tr = {}
    for pos, fi, bud, src, kr in rows():
        e = C.check([pos, fi, bud, src, kr])
        assert not e, '검사 오류 %s %s' % (pos, e)
        o = int(pos, 16); n = int(bud)
        assert orig[o:o + n] == src.encode('cp932') and orig[o + n] == 0, '원문 자리 다름 %s' % pos
        if C.in_pool(pos):
            tr[pos] = kr; continue
        b = encode(C.squeeze(kr)); assert len(b) <= n
        pac[o:o + n] = b + bytes(n - len(b)); spans.append((o, o + n))
    spans.sort()
    assert all(a1 <= b0 for (_, a1), (b0, _) in zip(spans, spans[1:])), '겹침'
    for fi, (used, cap) in repack(pac, orig, tr).items():
        print('영역 %s번: %d / %d B' % (fi, used, cap))
    # 메뉴 아틀라스(TXS 989A000) 조각 글자 + 22번 파일 조각 레코드(u·v·x·w)
    import menu_kr as MK
    recs, at, _, neww = MK.make(bytes(pac))
    lay = MK.relayout(recs, neww)
    pac[MK.OV:MK.OV + 0x2F800] = MK.patch_ov(bytes(pac[MK.OV:MK.OV + 0x2F800]), recs, lay, neww)
    a = MK.TXS + 0x800; enc = at.encode()
    pac[a:a + len(enc)] = enc
    print('메뉴 조각 %d · 아틀라스 %d B' % (len(neww), len(enc)))
    import ctrl_kr as CK                                         # 컨트롤 화면 패드 라벨(같은 아틀라스 u112~255 v0~255) — 메뉴 다음에
    print('컨트롤 라벨 %d' % CK.apply(pac))
    import header_kr as HK                                       # 머리 제목(같은 아틀라스 tpage E) — 메뉴 다음에
    print('머리 제목 %d' % HK.apply(pac))
    # 모델 플로어 조작 안내(TXS 9B91800) · 버튼 안내(TXS 98D3000)
    import modelfloor_kr as MF
    import btn_kr as BT
    print('모델 플로어 조각 %d · 버튼 안내 옮긴 조각 %d' % (MF.apply(pac), BT.apply(pac)))
    # 철도 자료관(989A000 사각형 1) · 영상 제목·番(사각형 2)
    import floor_kr as FK
    import caption_kr as CP
    print('자료관 옮긴 조각 %d · 영상 제목 조각 %d' % (FK.apply(pac), CP.apply(pac)))
    # 운전 승무: 시각표 역 이름·노선 선택·작은 낱말·배지(TXS 7E54800 + 34번 파일 레코드)
    import unten_kr as UK
    print('시각표 역 레코드 %d · 노선 선택 레코드 %d' % UK.apply_all(pac))
    # MDEC 정지 영상(제목 24·총 주행 거리 11) — 미리 구운 work/bs_out (tools/mdec_kr.py make)
    import mdec_kr as MD
    print('MDEC 정지 영상 %d' % MD.apply(pac))
    # 게임 중 HUD: 아틀라스 낱말(hud_kr, 7 사본) · 열차 소개 상자 90(desc_kr) · 노선명·차종 78(rname_kr)
    import hud_kr as HU
    import desc_kr as DK
    import rname_kr as RN
    print('HUD 낱말 %d · 열차 소개 %d · 노선명 %d' % (HU.apply(pac), DK.apply(pac), RN.apply(pac)))
    import stn_kr as SK                                          # 노선도 판 세로 역 이름(노선 그림 9 + 칸 표 레코드)
    print('역 이름 레코드 %d' % SK.apply(pac))
    # HUD 낱말 조각(hud_words, 갈무리 비트맵 B안) — 옮긴 조각은 실행 파일 레코드 u·v·CLUT 고침
    import hud_words as HW
    exe0 = open(os.path.join(ROOT, 'work', 'disc', 'SLPM_863.78'), 'rb').read()
    exe, nrec = HW.apply(pac, exe0)
    import banner_kr as BN                                       # 주행 중 노란 띠 큰 역 이름(노선 TXS 띠 + 모듈·실행 파일 레코드)
    exe, nb = BN.apply(pac, exe)
    print('노란 띠 역 이름 레코드 %d' % nb)
    print('HUD 낱말 조각 · 실행 파일 레코드 %d' % nrec)
    import logo_kr as LG                                         # HUD 4비트 로고 運転評価·ボーナス·完通
    print('HUD 로고 %d' % LG.apply(pac))
    import loading_kr as LK                                      # 로딩 화면 今度の発車は·停車駅(TXS 5사본)
    print('로딩 화면 머리 %d' % LK.apply(pac))
    import plate_kr as PK                                        # 로딩 화면 정차역 판 세로 역 이름(25번 판 레코드 + TXS 칸 띠)
    print('정차역 판 레코드 %d' % PK.apply(pac))
    import stn_small as SS                                       # 위 줄 노란 작은 역 이름(TXS 8장, 비트 층별)
    print('작은 역 이름 그림 %d' % SS.apply(pac))
    import scen_kr as SC                                         # 풍경 속 장소 이름표(노선 그림 사각형 1)
    print('풍경 이름표 그림 %d' % SC.apply(pac))
    import scen2_kr as S2                                        # 풍경 이름표 2차(큐슈·오사카·야마노테·케이힌2, 노선 모듈 이름표 표)
    print('풍경 이름표 2차 %d' % S2.apply(pac))
    import photobar_kr as PB                                     # 사진 보기 위 안내 띠(TXS AC03800 16비트)
    print('사진 안내 띠 구간 %d' % PB.apply(pac))
    import result_kr as RK                                       # 결과 화면 運転終了·全区間走破
    print('결과 제목 그림 %d' % RK.apply(pac))
    import bonus_kr as BK                                        # 보너스 게임·병합 안내·출발 점검(仕業検査)
    print('보너스·점검 그림 %d' % BK.apply(pac))
    import end_kr as EK                                          # 운행 종료 평가표(9798800)·승무 기록 화면(97A2000)
    print('운행 종료 그림 %d' % EK.apply(pac))
    secs = sorted({i // USER for i in range(0, len(pac), USER) if pac[i:i + USER] != orig[i:i + USER]})
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    if not os.path.exists(out_bin) or os.path.getsize(out_bin) != os.path.getsize(SRC_BIN):
        print('BIN 복사…'); shutil.copyfile(SRC_BIN, out_bin)
    with open(SRC_BIN, 'rb') as fs, open(out_bin, 'r+b') as f:
        # 지난 빌드 흔적 지우기: PAC «전체»에서 이번에 안 쓰는 섹터가 원본과 다르면 되돌림
        # (2026-10-19 12차: 정해 둔 구간만 되돌려 HUD 사본 0xE8800~ 의 9~11차 계기판 덮어쓰기가 남아 있었음)
        sset = set(secs); nrest = 0
        for s in range(-(-len(pac) // USER)):
            if s in sset:
                continue
            fs.seek((PAC_LBA + s) * SEC); a = fs.read(SEC)
            f.seek((PAC_LBA + s) * SEC); b = f.read(SEC)
            if a != b:
                f.seek((PAC_LBA + s) * SEC); f.write(a); nrest += 1
        print('지난 빌드 흔적 되돌림 %d 섹터' % nrest)
        for s in secs:
            lba = PAC_LBA + s
            f.seek(lba * SEC); sec = bytearray(f.read(SEC))
            assert not cdecc.is_form2(sec)
            sec[24:24 + USER] = pac[s * USER:(s + 1) * USER]
            cdecc.fix_sector(sec)
            f.seek(lba * SEC); f.write(sec)
        # 실행 파일 SLPM_863.78 (LBA 25, 150 섹터): 원본으로 되돌린 뒤 바뀐 섹터만
        exe_secs = [i // USER for i in range(0, len(exe), USER) if exe[i:i + USER] != exe0[i:i + USER]]
        for s in range(len(exe0) // USER):
            fs.seek((EXE_LBA + s) * SEC); f.seek((EXE_LBA + s) * SEC); f.write(fs.read(SEC))
        for s in exe_secs:
            f.seek((EXE_LBA + s) * SEC); sec = bytearray(f.read(SEC))
            assert not cdecc.is_form2(sec)
            sec[24:24 + USER] = exe[s * USER:(s + 1) * USER]
            cdecc.fix_sector(sec)
            f.seek((EXE_LBA + s) * SEC); f.write(sec)
    print('문장 %d · 섹터 %d · 실행 파일 섹터 %d · 출력 %s' % (len(spans), len(secs), len(exe_secs), out_bin))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'work', 'build', os.path.basename(SRC_BIN)))
