# Dog Measurement System

스마트폰 사진으로 강아지 신체 치수를 측정하는 시스템.
SMAL/BITE 파라메트릭 3D 모델을 활용하여 단일 사진에서 3D 형상을 복원하고 치수를 추출한다.

## 프로젝트 상태

**Day 0 Spike 진행 중** (2026-05-09 ~)

## 구조

```
dog-measurement/
├── notebooks/
│   └── day0_spike.ipynb       # Day 0 Spike (Colab/GPU 데스크탑에서 실행)
├── src/
│   └── measurement/
│       ├── __init__.py
│       └── extractor.py       # SMAL 메시 → 치수 추출 파이프라인
├── data/
│   └── test_images/           # 테스트용 강아지 사진
├── docs/
├── .gitignore
└── README.md
```

## Day 0 Spike 실행 방법

### 환경

- Google Colab (GPU 런타임) 또는 CUDA GPU가 있는 데스크탑
- Python 3.7+ / PyTorch 1.6+

### 실행

1. `notebooks/day0_spike.ipynb`을 Colab에 업로드
2. 런타임 → 런타임 유형 변경 → GPU 선택
3. 셀을 순서대로 실행
4. Step 6에서 Pass/Fail 판정

### Pass/Fail 기준

| Spike | Pass 조건 | Fail 시 |
|-------|----------|---------|
| Spike 1 (3D 복원) | 강아지 형상이 인식 가능한 메시 생성 | Approach A로 피벗 |
| Spike 2 (치수 추출) | 가슴둘레/등길이 오차 3cm 이내 (10마리 중 7마리) | 스케일 보정 방법 재탐색 |

## 기술 기반

- **BITE** (CVPR 2023): 단일 사진 → SMAL 파라메트릭 3D 모델 피팅
- **SMAL**: Skinned Multi-Animal Linear model (3,889 vertices, 33 body segments)
- 치수 추출: 메시 cross-section 기반 둘레 계산
