# -*- coding: utf-8 -*-
r"""전차로 GO! 프로페셔널 사양 배포 묶음 — dist/DenshaDeGoPro_KR_<VER>/ : BIN xdelta + xdelta.exe + readme.txt(CP949) + 한글패치_적용.bat (2026-10-19)
  검증: 원본 BIN → xdelta 적용 → md5 = 빌드 결과 md5. CUE 는 원본 그대로 씀(트랙 1개, 크기 같음).
  python tools/make_dist.py   (먼저 python tools/mdec_kr.py make · python tools/build.py)
"""
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
VER = 'v0.9'
XDELTA = r'C:\claude\utils\xdelta.exe'
ROMNAME = 'Densha de Go! Professional Shiyou (Japan)'
SRC = r'C:\claude\roms\ps\Densha de Go Professional Shiyou (Japan)\Densha de Go! Professional Shiyou (Japan).bin'
BUILD = os.path.join(ROOT, 'work', 'build', ROMNAME + '.bin')
NAME = 'DenshaDeGoPro_KR_' + VER
TITLE = '전차로 GO! 프로페셔널 사양 (PS1 일본판 SLPM-86378) 한글 패치 ' + VER


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest().upper()


HEAD = """{rom}.bin (CD 이미지, SLPM-86378) 에 패치하시면 됩니다.

원본md5 : {o}
패치md5 : {d}

입니다.
"""

BODY = """

[ 적용 방법 ]

  1) 원본 BIN 파일을 이 폴더에 복사
       "{bin}"
  2) 한글패치_적용.bat 실행 → 이름 끝에 [KR] 이 붙은 BIN 이 만들어집니다
  3) 원본 CUE 를 복사해 [KR] BIN 이름으로 고치거나, CUE 안의 파일 이름을 [KR] BIN 으로 바꿔 주세요

  직접 적용:
    xdelta.exe -d -s "원본 BIN" "{patch}" "결과 BIN"
  (Delta Patcher 같은 xdelta3 GUI 도구로 적용해도 됩니다. 원본이 다르면 xdelta 가 적용을 거부합니다.)


[ 한글 표시 ]

  - 게임 속 문장(평문)은 BIOS 한자 글꼴 자리를 쓰므로 "한글 BIOS" 가 필요합니다.
    일본 BIOS 로 실행하면 문장이 한자로 깨져 보입니다.
  - 그림 글자(메뉴·계기판·안내 말풍선·사진 설명 등)는 BIOS 와 상관없이 한글로 나옵니다.


[ 바뀌는 것 ]

  - 메뉴·운전 승무(노선·다이아 선택, 시각표 역 이름)·철도 자료관·로드/세이브·게임 설정
  - 게임 중 계기판 글자·신호/제한 안내·말풍선 안내문·일시정지 메뉴·노선도 역 이름·노란 띠 역 이름·풍경 이름표
  - 로딩 화면(다음 발차·정차역 판)·운전 종료/평가표·승무 기록·보너스 게임·출발 점검
  - 열차 소개 90장·자료관 사진 설명 397장·총 주행 거리 화면
  - 제목 로고의 "電車で" 를 "전차로" 로 (GO! 는 그대로)
  - 스태프 롤·영어 표기는 원본 그대로입니다


[ 알려진 점 ]

  - 아직 끝까지 실기로 확인하지 못했습니다. 남은 일본어나 깨지는 글자가 있으면 알려 주세요.
"""

BAT = r"""@echo off
chcp 949 >nul
set "XD=%~dp0xdelta.exe"
if not exist "%~dp0{bin}" (
  echo   [오류] 원본 BIN 파일을 이 폴더에 넣어 주세요 - readme 참고.
  pause & exit /b 1
)
echo   패치 중...
"%XD%" -d -f -s "%~dp0{bin}" "%~dp0{patch}" "%~dp0{kbin}"
if errorlevel 1 (
  echo   [오류] 패치 실패 - 원본이 다를 수 있습니다 - readme 의 원본md5 확인.
  pause & exit /b 1
)
echo   완료: "{kbin}"
pause
"""


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    d = os.path.join(ROOT, 'dist', NAME)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    binf = ROMNAME + '.bin'
    patch = NAME + '.xdelta'; pp = os.path.join(d, patch)
    subprocess.run([XDELTA, '-e', '-9', '-f', '-B', '268435456', '-s', SRC, BUILD, pp], check=True)
    chk = os.path.join(d, '_check.bin')
    subprocess.run([XDELTA, '-d', '-f', '-B', '268435456', '-s', SRC, pp, chk], check=True)
    o, want, got = md5(SRC), md5(BUILD), md5(chk)
    os.remove(chk)
    assert got == want, ('패치 적용 결과가 빌드와 다름', got, want)
    kbin = ROMNAME + ' [KR].bin'
    shutil.copy2(XDELTA, os.path.join(d, 'xdelta.exe'))
    readme = (TITLE + '\n' + '=' * 60 + '\n\n' + HEAD.format(rom=ROMNAME, o=o, d=want)
              + BODY.format(bin=binf, patch=patch))
    open(os.path.join(d, 'readme.txt'), 'wb').write(readme.replace('\n', '\r\n').encode('cp949'))
    open(os.path.join(d, '한글패치_적용.bat'), 'wb').write(
        BAT.format(bin=binf, patch=patch, kbin=kbin).replace('\n', '\r\n').encode('cp949'))
    zp = shutil.make_archive(d, 'zip', os.path.dirname(d), NAME)
    print('원본md5 %s → 패치md5 %s · %s %d B' % (o, want, patch, os.path.getsize(pp)))
    print('zip', zp, os.path.getsize(zp), md5(zp))
    for f in sorted(os.listdir(d)):
        print('  %-50s %12d' % (f, os.path.getsize(os.path.join(d, f))))


if __name__ == '__main__':
    main()
