"""
SMAL 메시에서 강아지 신체 치수를 추출하는 파이프라인.

SMAL 39dogs_norm 모델 기준:
- 3,889 vertices, 7,774 faces, 35 joints
- 좌표계: X=앞(머리), Y=옆(좌우대칭), Z=위아래(위가 양수)
- 단위: SMAL 자체 단위 (실제 cm 아님, 스케일 계수 필요)

Joint 구조 (토폴로지 분석 2026-05-09):
  0: hip_root, 1: spine_low, 2: spine_mid, 3: spine_upper
  4: chest, 5: shoulder_area, 6: neck_base
  7-10: 오른쪽 앞다리, 11-14: 왼쪽 앞다리
  15: neck_upper, 16: head
  17-20: 오른쪽 뒷다리, 21-24: 왼쪽 뒷다리
  25-31: 꼬리, 32: 코, 33-34: 귀

사용법:
    extractor = DogMeasurementExtractor.from_smal_model("path/to/smal.pkl")
    measurements = extractor.extract(vertices, faces, scale_factor=61.0)
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional


SMAL_JOINT_NAMES = {
    0: "hip_root", 1: "spine_low", 2: "spine_mid", 3: "spine_upper",
    4: "chest", 5: "shoulder_area", 6: "neck_base",
    7: "R_front_shoulder", 8: "R_front_elbow", 9: "R_front_wrist", 10: "R_front_paw",
    11: "L_front_shoulder", 12: "L_front_elbow", 13: "L_front_wrist", 14: "L_front_paw",
    15: "neck_upper", 16: "head",
    17: "R_rear_hip", 18: "R_rear_knee", 19: "R_rear_ankle", 20: "R_rear_paw",
    21: "L_rear_hip", 22: "L_rear_knee", 23: "L_rear_ankle", 24: "L_rear_paw",
    25: "tail_0", 26: "tail_1", 27: "tail_2", 28: "tail_3",
    29: "tail_4", 30: "tail_5", 31: "tail_tip",
    32: "nose", 33: "R_ear", 34: "L_ear",
}

LEG_JOINT_IDS = set(range(7, 15)) | set(range(17, 25))


class DogMeasurementExtractor:
    """SMAL 3D 메시에서 강아지 신체 치수를 추출."""

    def __init__(self, joint_regressor: np.ndarray, skinning_weights: np.ndarray):
        self.J_reg = joint_regressor      # (35, 3889)
        self.weights = skinning_weights   # (3889, 35)
        self.part_labels = np.argmax(skinning_weights, axis=1)
        self.torso_mask = ~np.isin(self.part_labels, list(LEG_JOINT_IDS))

    @classmethod
    def from_smal_model(cls, model_path: str) -> "DogMeasurementExtractor":
        with open(model_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        J_reg = data["J_regressor"]
        if hasattr(J_reg, "toarray"):
            J_reg = J_reg.toarray()
        return cls(
            joint_regressor=np.array(J_reg, dtype=np.float64),
            skinning_weights=np.array(data["weights"], dtype=np.float64),
        )

    def extract(self, vertices: np.ndarray, faces: np.ndarray,
                scale_factor: float = 1.0) -> Dict[str, float]:
        """
        BITE 출력 메시에서 치수 추출.

        Args:
            vertices: (3889, 3) SMAL 메시 정점 좌표
            faces: (7774, 3) 면 인덱스
            scale_factor: SMAL 단위 → cm 변환 계수.
                          견종/체중 기반으로 결정됨.

        Returns:
            치수 딕셔너리 (단위: cm)
        """
        joints = self._compute_joints(vertices)

        neck_x = joints[6][0]
        chest_x = (joints[4][0] + joints[5][0]) / 2
        belly_x = joints[2][0]

        neck_circ = self._circumference_at(vertices, faces, neck_x)
        chest_circ = self._circumference_at(vertices, faces, chest_x)
        belly_circ = self._circumference_at(vertices, faces, belly_x)
        back_length = abs(joints[6][0] - joints[0][0])

        return {
            "neck_circumference_cm": neck_circ * scale_factor,
            "chest_circumference_cm": chest_circ * scale_factor,
            "belly_circumference_cm": belly_circ * scale_factor,
            "back_length_cm": back_length * scale_factor,
            "raw_smal": {
                "neck_circ": neck_circ,
                "chest_circ": chest_circ,
                "belly_circ": belly_circ,
                "back_length": back_length,
            },
            "joints_used": {
                "neck": f"Joint 6 (X={neck_x:.4f})",
                "chest": f"Joint 4-5 avg (X={chest_x:.4f})",
                "belly": f"Joint 2 (X={belly_x:.4f})",
                "back": f"Joint 6→0 (X={joints[6][0]:.4f}→{joints[0][0]:.4f})",
            },
        }

    def _compute_joints(self, vertices: np.ndarray) -> np.ndarray:
        if hasattr(self.J_reg, 'toarray'):
            return self.J_reg.toarray() @ vertices
        return self.J_reg @ vertices

    def _circumference_at(self, vertices: np.ndarray, faces: np.ndarray,
                           x_pos: float) -> float:
        """X 위치에서 몸통만 수직 절단하여 둘레 계산."""
        signed = vertices[:, 0] - x_pos
        points = []
        seen_edges = set()

        for face in faces:
            torso_count = sum(1 for v in face if self.torso_mask[v])
            if torso_count < 2:
                continue
            for i, j in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
                edge_key = (min(i, j), max(i, j))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                if signed[i] * signed[j] < 0:
                    t = signed[i] / (signed[i] - signed[j])
                    pt = vertices[i] + t * (vertices[j] - vertices[i])
                    points.append(pt)

        if len(points) < 3:
            return 0.0

        pts = np.array(points)
        return self._ordered_perimeter(pts)

    def _ordered_perimeter(self, points: np.ndarray) -> float:
        """교차점들을 각도순으로 정렬 후 폐합 둘레 계산."""
        centroid = points.mean(axis=0)
        rel = points - centroid
        angles = np.arctan2(rel[:, 2], rel[:, 1])
        ordered = points[np.argsort(angles)]
        n = len(ordered)
        return sum(
            np.linalg.norm(ordered[(i + 1) % n] - ordered[i])
            for i in range(n)
        )

    # neutral 포즈에서 joint 6 (neck_base) ~ joint 0 (hip_root) X 거리
    NEUTRAL_BACK_LENGTH_SMAL = 0.574

    def estimate_scale_factor(self, weight_kg: float) -> float:
        """
        체중에서 SMAL→cm 스케일 계수를 추정.

        spike용 근사치. 실제 서비스에서는 견종별 DB 필요.
        """
        if weight_kg <= 0:
            return 1.0
        estimated_back_cm = 8.5 * (weight_kg ** 0.33) + 15
        return estimated_back_cm / self.NEUTRAL_BACK_LENGTH_SMAL


def compute_scale_from_ground_truth(smal_back_length: float,
                                      actual_back_length_cm: float) -> float:
    """줄자 측정값이 있을 때 정확한 스케일 계수 계산."""
    if smal_back_length <= 0:
        return 1.0
    return actual_back_length_cm / smal_back_length
