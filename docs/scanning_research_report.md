# 스마트폰 스캐닝 강아지 치수 측정 — 종합 리서치 보고서

작성일: 2026-05-18
프로젝트: dog-measurement

## 요약

강아지 주위를 스캔하여 치수를 측정하는 방식의 기술적 실현 가능성을 조사한 결과,
현재 기술로 주력 경로가 되기 어렵다는 결론. "사진 → AI 추정" 방식이 최적.

## 1. 시장 현황

- 강아지 체형 AI 측정 앱은 전 세계에 존재하지 않음
- 모든 반려동물 의류 업체가 줄자 수동 측정 방식 사용
- 이 시장 공백이 사업 기회

## 2. 스캔 방식의 3가지 치명적 문제

### 2-1. 강아지 움직임
- 모든 3D 스캔 기술(포토그래메트리, NeRF, LiDAR)은 피사체 정지가 전제
- 강아지는 5초 이상 가만히 있기 어려움
- Scaniverse, Polycam, Luma AI 모두 움직이는 피사체 미지원

### 2-2. 털에 의한 깊이 센서 교란
- LiDAR/ToF는 털 표면을 측정, 실제 체형이 아님
- 포메라니안 등 장모종은 실제 체형보다 30-50% 크게 측정
- 3D 스캔에서 가장 어려운 표면 유형 중 하나

### 2-3. 안드로이드 깊이 센서 부재
- 2026년 기준 안드로이드 주력 폰에 LiDAR/ToF 없음
- Samsung도 Galaxy S20 이후 ToF 센서 제거
- ARCore Depth API (소프트웨어 기반) 오차 ~8cm → 의류용 부적합

## 3. 타 분야 기술 벤치마크

### 인체 의류 측정 (3DLOOK)
- 방식: 사진 2장 (전면 + 측면), 처리 30초 미만
- 정확도: 96-97%, IEEE 인증
- 기기: 모든 스마트폰 (카메라만 필요)
- 비즈니스 성과: 전환율 4배, AOV 20% 상승, 반품률 6% 감소
- 기술: 독자 뉴럴 네트워크, 수십만 장 학습 데이터

### 인체 전용 스캐너 (Styku, TG3D)
- Styku: 35초 풀 바디 스캔, 2mm 이내, 회전 플랫폼 필요
- TG3D (Scanatic 360): 3초 스캔, +-5mm, 18개 센서 멀티카메라
- 소비자용 불가 (전용 하드웨어)

### 가축 측정 (소/돼지)
- ToF 고정식 (3대 깊이카메라): 오차 2-5%, 5mm 수준
- iPhone LiDAR (Scanabull): 93%+ 정확도, 180도 아크 스캔
- 스마트폰 사진 (돼지): 상관 0.99, 체중 예측
- 핵심: 통제된 환경 + 짧은 털 + 체형 편차 적음 → 강아지와 다른 조건

### 반려견 의지 (3DPets)
- iPhone LiDAR로 개 스캔 → 맞춤 의지 제작
- 첫 착용 성공률 90% 이상, 재작업 40% 감소
- 가장 가까운 성공 사례지만 정밀 치수보다는 형태 매칭 목적

## 4. 기술별 정확도 요약

| 기술 | 정확도 | 기기 호환 | 움직이는 대상 | 비용 |
|---|---|---|---|---|
| 사진 2장 + AI (3DLOOK식) | 1-3cm | 모든 폰 (~100%) | 가능 (순간 촬영) | 앱 |
| iPhone LiDAR 스캔 | 1-2cm (정지 시) | iPhone Pro (~15%) | 불가 | 앱 |
| ARCore Depth (Android) | ~8cm | Android 87% | 불가 (정지 필요) | 앱 |
| 포토그래메트리 (비디오) | 2-5cm | 모든 폰 | 불가 (정지 필수) | 앱 |
| NeRF/Gaussian Splatting | 연구 수준 | GPU 폰 | 불가 (30초+ 정지) | N/A |
| 전용 깊이카메라 3대 | 5mm | 전용 하드웨어 | 제한적 | $5K-20K |

## 5. 결론: 최적 접근법

### 주력: 사진 → AI 추정 (clother-v2 앱화)

3DLOOK이 인체에서 증명한 방식을 강아지에 적용:
- 사용자가 옆모습 1장 + 앞/위 1장 촬영 (5-10초)
- AI가 견종 식별 + 체형 추정 + 치수 출력
- 체중 입력으로 절대 스케일 보정
- 모든 스마트폰 지원

### 보조: LiDAR 프리미엄 경로

iPhone Pro 사용자 대상 추가 정확도:
- LiDAR 깊이 데이터로 스케일 보정 강화
- 짧은 영상(5-10초) 촬영으로 3D 데이터 보충

### 로드맵

| Phase | 내용 | 기기 | 예상 정확도 |
|---|---|---|---|
| 1 (MVP) | 사진 + AI + 체중/견종 | 모든 폰 | 오차 2-3cm |
| 2 | + Metric Depth 보정 | 모든 폰 | 오차 1.5-2.5cm |
| 3 | + LiDAR 프리미엄 | iPhone Pro | 오차 1-2cm |
| 4 | + 사용자 피드백 학습 | 모든 폰 | 지속 개선 |

## 6. Day 0 Spike 최종 결론

- Spike 1 (BITE 3D 복원): PASS — 형상 복원 성공, 견종 차이 반영
- Spike 2 (치수 추출): FAIL — SMAL 절대 스케일 복원 불가, clother-v2보다 나쁨
- 스캔 방식: NOT RECOMMENDED — 움직임/털/기기 호환 문제
- 최적 경로: clother-v2 개선 + Metric Depth + LiDAR 보조

## 참고 자료

### 인체 측정
- 3DLOOK: https://3dlook.ai/technology/
- Styku: https://www.styku.com/body-scanner
- TG3D Studio: https://www.tg3ds.com/3d-body-scanner
- SHAPY (CVPR 2022): 3D body shape regression

### 가축 측정
- Scanabull: https://scanabull.com/our-tech/ (소, iPhone LiDAR, 93%+)
- CattleWeight AI: https://cattleweightestimation.com/ (소, LiDAR, +-3%)
- 가축 체형 측정 서베이: PMC 10934764

### 반려동물
- 3DPets: https://3dpetsprosthetics.com/ (개 의지, LiDAR, 90% 성공률)
- BARC/BITE: 개 3D 형상 추정 모델

### 깊이 추정
- Depth Anything V2: 야외 동물 MAE 0.454m
- Apple Depth Pro: 단일 이미지 메트릭 깊이, 털/모발 경계 고정확도
- iPhone LiDAR: RMSE 4.89mm (근거리), 1-2% 오차

### 스마트폰 AR
- Apple Measure App: LiDAR 기기 1-2cm, 비LiDAR 2-20% 오차
- ARCore Depth API: 소프트웨어 깊이 ~8cm 오차
- 학술 연구 (PMC 10939328): LiDAR 1.0cm, ARKit 1.1cm, CoreML 4.4cm
