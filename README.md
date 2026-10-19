# 전차로 GO! 프로페셔널 사양 (PS1 일본판 SLPM-86378) 한글 패치

- 대상: `Densha de Go! Professional Shiyou (Japan).bin` (CD, 원본md5 `4E487C1FA9F4E2B886FF24C0AB977606`)
- 배포: Releases 의 `DenshaDeGoPro_KR_v0.9.zip` (xdelta + 적용 bat + readme) → 패치md5 `029AFFF6DAA99D07C85BF756B3FDDB34`
- 평문 문장은 BIOS 한자 글꼴 자리를 쓰므로 한글 BIOS 가 필요합니다(그림 글자는 상관없음).
- 작업 기록: `docs/00_초기조사.md` · `docs/00_이어하기.md`
- 빌드: `python tools/mdec_kr.py make` → `python tools/build.py` → 배포 `python tools/make_dist.py`
