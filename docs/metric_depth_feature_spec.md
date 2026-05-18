# Metric Depth Feature Spec for clother-v2

## 배경

dog-measurement 프로젝트 Day 0 Spike 결과:
- BITE/SMAL 3D 복원 → 치수 측정: clother-v2(MAE 17-25mm)보다 나쁨 (MAE 41-63mm)
- SMAL betas → 추가 feature: 신호 약함 (개별 beta와 치수 상관 |r| < 0.45)
- 결론: 3D 복원 방향 중단, 대신 Metric Depth를 clother-v2에 추가

## 목표

clother-v2 모델에 **monocular metric depth** 정보를 추가 feature로 입력하여 정확도 개선.
특히 대형견 가슴둘레(MAE 36mm)와 짧은 피모 가슴둘레(MAE 33mm)에서 효과 기대.

## 왜 Metric Depth인가

사진만으로 강아지의 절대 크기를 아는 것은 어렵다.
- 가까이서 찍은 작은 강아지 vs 멀리서 찍은 큰 강아지가 사진에서 같아 보임
- 체중/견종 메타데이터가 보완하지만, 같은 견종 내 개체 차이는 못 잡음

Metric Depth 모델(Depth Anything V2 등)은 사진 한 장에서 **각 픽셀의 실제 거리(미터)**를 추정한다.
이걸 쓰면:
1. 강아지까지의 거리를 알 수 있고
2. 사진에서 강아지가 차지하는 픽셀 크기 + 거리 → 실제 물리 크기 추정 가능
3. 이 정보를 모델에 추가 feature로 넣으면 스케일 모호성이 줄어듦

## 추천 모델

**Depth Anything V2 (Metric)**
- 정확도: 야외 동물 벤치마크에서 MAE 0.454m (1-7m 거리), 상관 0.962
- 속도: 이미지당 ~0.1초 (GPU)
- 설치: `pip install transformers` (HuggingFace에서 로드)
- 라이선스: Apache 2.0

대안:
- Depth Pro (Apple): 정확하지만 상업 라이선스 제한
- Metric3D v2: 구조적 일관성 높지만 절대 오차 큼

## Feature 설계

각 이미지에서 추출할 depth feature (8-10차원):

```python
# 1. 전체 depth map 생성
depth_map = depth_model(image)  # (H, W) 각 픽셀의 미터 거리

# 2. 강아지 영역 마스크 (YOLO bbox 또는 segmentation)
dog_mask = detect_dog_region(image)

# 3. 강아지 영역의 depth 통계
dog_depth = depth_map[dog_mask]
features = {
    'depth_mean': dog_depth.mean(),        # 평균 거리
    'depth_median': np.median(dog_depth),  # 중앙값 거리
    'depth_std': dog_depth.std(),          # 거리 편차 (깊이감)
    'depth_min': dog_depth.min(),          # 가장 가까운 부분
    'depth_max': dog_depth.max(),          # 가장 먼 부분
    'depth_range': dog_depth.max() - dog_depth.min(),  # 앞뒤 차이 (체장 힌트)
}

# 4. 스케일 추정 feature
dog_bbox = get_dog_bbox(image)
pixel_height = dog_bbox[3] - dog_bbox[1]
pixel_width = dog_bbox[2] - dog_bbox[0]
features['physical_height_est'] = pixel_height * depth_mean / focal_length  # 추정 체고(m)
features['physical_width_est'] = pixel_width * depth_mean / focal_length    # 추정 체폭(m)
```

## clother-v2 아키텍처 수정

### 현재 (v2/v3)
```
ImageEncoder(6-13 views) → 1536d
MetadataEncoder(weight, breed, sex) → 128d
concat(1536d + 128d) → FusionHead → 3 predictions
```

### 수정 후
```
ImageEncoder(6-13 views) → 1536d
MetadataEncoder(weight, breed, sex) → 128d
DepthEncoder(depth_features per view) → 32d   ← NEW
concat(1536d + 128d + 32d) → FusionHead → 3 predictions
```

### DepthEncoder 구조
```python
class DepthEncoder(nn.Module):
    def __init__(self, input_dim=8, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
    
    def forward(self, depth_features):
        # depth_features: (B, N_views, 8) 또는 (B, 8) if view-aggregated
        return self.net(depth_features)
```

### View 처리 방식
- 방법 A: 각 view마다 depth feature 추출 → view별로 image feature와 concat → attention pooling
- 방법 B: 모든 view의 depth feature를 평균 → 단일 depth vector → metadata와 concat
- **추천: 방법 B** (더 간단하고 depth 정보는 view 간 크게 다르지 않음)

## 데이터 파이프라인

### Phase 1: Depth 캐시 생성 (Colab A100에서)
```
5484 dogs × 평균 8 images = ~44,000 images
이미지당 ~0.1초 → 총 ~73분 (A100)
결과: depth_cache.pkl (dog_id → {view_idx: depth_features})
```

### Phase 2: 기존 데이터셋에 통합
```
train_v2.csv의 각 row에 depth feature 추가
→ data/processed/depth_features_v2.pkl
```

### Phase 3: 모델 학습
```
기존 학습 코드에 DepthEncoder 추가
configs/에 depth 관련 설정 추가:
  depth:
    enabled: true
    model: depth_anything_v2
    features: [mean, median, std, min, max, range, phys_height, phys_width]
    encoder_dim: 32
```

## 실행 계획

| 단계 | 어디서 | 시간 | 내용 |
|---|---|---|---|
| 1 | Colab | ~73분 | Depth Anything V2로 44,000장 depth 추출 + 캐시 |
| 2 | 로컬 | 30분 | DepthEncoder 코드 작성 + 기존 모델에 통합 |
| 3 | Colab | ~2시간 | 학습 (80 epochs, baseline 대비 비교) |
| 4 | 로컬 | 30분 | 결과 분석 + 그룹별 개선 확인 |

## 기대 효과

- 전체 MAE: 17-25mm → **15-22mm** (1-3mm 개선)
- 대형견 가슴둘레: 36mm → **30mm 이하** (스케일 정보 보강)
- 짧은 피모 가슴둘레: 33mm → **28mm 이하**

## 리스크

- Depth 모델의 실내/실외 정확도 차이 (학습 데이터가 다양한 환경)
- 강아지 사진에 depth 모델이 정확한지 별도 검증 필요
- Feature가 노이즈만 추가할 가능성 → ablation으로 확인

## 참고

- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- Wildlife depth benchmark: MAE 0.454m, 상관 0.962
- clother-v2 현재 depth 캐시: data/processed/depth_cache_384.pkl (v3d/v4에서 사용 중이지만 global stats만)
- 이 spec은 pixel-level depth → physical size estimation으로 확장하는 것
