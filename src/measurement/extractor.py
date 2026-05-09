"""
SMAL 메시에서 강아지 신체 치수를 추출하는 파이프라인.

SMAL 모델은 3,889 vertices, 33 body segments로 구성.
각 vertex는 해부학적 위치에 고정 → 치수 추출 가능.

사용법:
    from src.measurement.extractor import DogMeasurementExtractor
    extractor = DogMeasurementExtractor()
    measurements = extractor.extract(vertices, faces)
"""

import numpy as np
from typing import Dict, Tuple, Optional


# SMAL 메시 해부학적 랜드마크 vertex indices
# 실제 값은 SMAL 메시 토폴로지 분석 후 업데이트 필요 (Day 0 Spike에서 수행)
# 아래는 SMAL 39dogs_norm 모델 기준 추정치 — 실행 후 시각적 검증 필수
LANDMARKS = {
    "neck_center": None,        # 목 중심점
    "chest_front": None,        # 가슴 앞쪽 (앞다리 사이)
    "chest_widest": None,       # 가슴 가장 넓은 부분
    "back_start": None,         # 등 시작점 (목-등 경계)
    "back_end": None,           # 등 끝점 (꼬리 시작 전)
    "belly_center": None,       # 배 중심점
    "shoulder_left": None,      # 왼쪽 어깨
    "shoulder_right": None,     # 오른쪽 어깨
    "hip_left": None,           # 왼쪽 엉덩이
    "hip_right": None,          # 오른쪽 엉덩이
    "front_leg_top_left": None, # 왼쪽 앞다리 시작
    "front_leg_top_right": None,
    "rear_leg_top_left": None,  # 왼쪽 뒷다리 시작
    "rear_leg_top_right": None,
}


class DogMeasurementExtractor:
    """SMAL 3D 메시에서 강아지 신체 치수를 추출."""

    def __init__(self, landmarks: Optional[Dict[str, int]] = None):
        self.landmarks = landmarks or LANDMARKS
        self._validate_landmarks()

    def _validate_landmarks(self):
        missing = [k for k, v in self.landmarks.items() if v is None]
        if missing:
            print(f"[WARNING] {len(missing)}개 랜드마크 미설정: {missing[:5]}...")
            print("  → discover_landmarks()로 SMAL 메시에서 자동 탐색하거나")
            print("  → 수동으로 vertex index를 설정하세요.")

    def extract(self, vertices: np.ndarray, faces: np.ndarray,
                scale_factor: float = 1.0) -> Dict[str, float]:
        """
        3D 메시에서 주요 치수 추출.

        Args:
            vertices: (N, 3) 메시 정점 좌표
            faces: (F, 3) 면 인덱스
            scale_factor: 절대 스케일 보정 계수 (SMAL 단위 → cm)

        Returns:
            치수 딕셔너리 (단위: cm)
        """
        measurements = {}

        if all(v is not None for v in self.landmarks.values()):
            measurements["back_length_cm"] = self._compute_back_length(vertices) * scale_factor
            measurements["chest_circumference_cm"] = self._compute_circumference(
                vertices, faces, "chest_widest", axis="sagittal"
            ) * scale_factor
            measurements["neck_circumference_cm"] = self._compute_circumference(
                vertices, faces, "neck_center", axis="sagittal"
            ) * scale_factor
            measurements["belly_circumference_cm"] = self._compute_circumference(
                vertices, faces, "belly_center", axis="sagittal"
            ) * scale_factor
            measurements["shoulder_width_cm"] = self._compute_distance(
                vertices, "shoulder_left", "shoulder_right"
            ) * scale_factor
        else:
            measurements = self._estimate_from_bounding_box(vertices, scale_factor)

        return measurements

    def _compute_distance(self, vertices: np.ndarray,
                          landmark_a: str, landmark_b: str) -> float:
        idx_a = self.landmarks[landmark_a]
        idx_b = self.landmarks[landmark_b]
        return float(np.linalg.norm(vertices[idx_a] - vertices[idx_b]))

    def _compute_back_length(self, vertices: np.ndarray) -> float:
        return self._compute_distance(vertices, "back_start", "back_end")

    def _compute_circumference(self, vertices: np.ndarray, faces: np.ndarray,
                                landmark: str, axis: str = "sagittal") -> float:
        """
        특정 랜드마크 위치에서 메시를 절단하고 둘레를 계산.

        절단면(cross-section)을 만들어 교차 곡선의 길이를 구함.
        """
        idx = self.landmarks[landmark]
        point = vertices[idx]

        if axis == "sagittal":
            normal = np.array([1.0, 0.0, 0.0])
        elif axis == "coronal":
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = np.array([0.0, 1.0, 0.0])

        cross_section = self._mesh_cross_section(vertices, faces, point, normal)

        if cross_section is None or len(cross_section) < 3:
            return 0.0

        return self._polygon_perimeter(cross_section)

    def _mesh_cross_section(self, vertices: np.ndarray, faces: np.ndarray,
                             plane_point: np.ndarray, plane_normal: np.ndarray
                             ) -> Optional[np.ndarray]:
        """
        메시를 평면으로 절단하여 교차점들을 반환.

        평면 방정식: dot(normal, (x - point)) = 0
        각 삼각형 edge가 평면을 횡단하면 교차점 계산.
        """
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        signed_dists = np.dot(vertices - plane_point, plane_normal)
        intersection_points = []

        for face in faces:
            edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
            for i, j in edges:
                d_i, d_j = signed_dists[i], signed_dists[j]
                if d_i * d_j < 0:
                    t = d_i / (d_i - d_j)
                    point = vertices[i] + t * (vertices[j] - vertices[i])
                    intersection_points.append(point)

        if len(intersection_points) < 3:
            return None

        points = np.array(intersection_points)
        return self._order_points_on_plane(points, plane_normal)

    def _order_points_on_plane(self, points: np.ndarray,
                                normal: np.ndarray) -> np.ndarray:
        """교차점들을 평면 위에서 각도순으로 정렬 (둘레 계산을 위해)."""
        centroid = points.mean(axis=0)
        rel = points - centroid

        abs_normal = np.abs(normal)
        if abs_normal[0] < abs_normal[1] and abs_normal[0] < abs_normal[2]:
            ref = np.array([1.0, 0.0, 0.0])
        elif abs_normal[1] < abs_normal[2]:
            ref = np.array([0.0, 1.0, 0.0])
        else:
            ref = np.array([0.0, 0.0, 1.0])

        u = np.cross(normal, ref)
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        angles = np.arctan2(np.dot(rel, v), np.dot(rel, u))
        order = np.argsort(angles)
        return points[order]

    def _polygon_perimeter(self, points: np.ndarray) -> float:
        """정렬된 3D 점들의 폐합 둘레 길이."""
        n = len(points)
        perimeter = 0.0
        for i in range(n):
            perimeter += np.linalg.norm(points[(i + 1) % n] - points[i])
        return perimeter

    def _estimate_from_bounding_box(self, vertices: np.ndarray,
                                     scale_factor: float) -> Dict[str, float]:
        """
        랜드마크 없이 bounding box 기반 대략적 치수 추정.
        Day 0 Spike용 폴백 — 정확도 낮음.
        """
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        dims = maxs - mins

        height = dims[1] * scale_factor
        length = dims[0] * scale_factor
        width = dims[2] * scale_factor

        return {
            "body_length_cm": length,
            "body_height_cm": height,
            "body_width_cm": width,
            "estimated_chest_circumference_cm": (height + width) * 1.1,
            "estimated_back_length_cm": length * 0.6,
            "note": "bounding box 기반 추정치 — 랜드마크 설정 후 정확한 값으로 교체 필요",
        }


def discover_landmarks(vertices: np.ndarray, smal_parts: np.ndarray) -> Dict[str, int]:
    """
    SMAL 파트 레이블에서 랜드마크 vertex index를 자동 탐색.

    SMAL 모델의 33 body segments:
    0-3: torso, 4-7: front legs, 8-11: rear legs,
    12-15: head/neck, 16-19: tail, 20+: ears 등

    Args:
        vertices: (N, 3) 정점 좌표
        smal_parts: (N,) 각 vertex의 파트 레이블 (0-32)

    Returns:
        랜드마크 vertex index 딕셔너리
    """
    landmarks = {}

    torso_mask = np.isin(smal_parts, [0, 1, 2, 3])
    torso_verts = vertices[torso_mask]
    torso_indices = np.where(torso_mask)[0]

    if len(torso_verts) > 0:
        x_min_idx = torso_indices[np.argmin(torso_verts[:, 0])]
        x_max_idx = torso_indices[np.argmax(torso_verts[:, 0])]
        landmarks["back_start"] = int(x_max_idx)
        landmarks["back_end"] = int(x_min_idx)

        x_mid = (torso_verts[:, 0].min() + torso_verts[:, 0].max()) / 2
        mid_dists = np.abs(torso_verts[:, 0] - x_mid)
        top_mask = torso_verts[:, 1] > np.median(torso_verts[:, 1])
        if top_mask.any():
            mid_top = np.where(top_mask)[0]
            best = mid_top[np.argmin(mid_dists[mid_top])]
            landmarks["chest_widest"] = int(torso_indices[best])

    neck_mask = np.isin(smal_parts, [12, 13])
    neck_verts = vertices[neck_mask]
    neck_indices = np.where(neck_mask)[0]
    if len(neck_verts) > 0:
        centroid = neck_verts.mean(axis=0)
        dists = np.linalg.norm(neck_verts - centroid, axis=1)
        landmarks["neck_center"] = int(neck_indices[np.argmin(dists)])

    return landmarks
