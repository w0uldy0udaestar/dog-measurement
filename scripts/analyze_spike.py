"""
Day 0 Spike 결과 분석 (로컬에서 실행).

Colab에서 저장한 spike_results.pkl을 로드하여:
1. 각 메시의 치수 추출
2. 시각화 (side/top/front view)
3. 줄자 ground truth와 비교 (있는 경우)
4. Pass/Fail 판정

사용법:
    python scripts/analyze_spike.py --results path/to/spike_results.pkl
    python scripts/analyze_spike.py --results path/to/spike_results.pkl --ground-truth path/to/gt.csv
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.measurement.extractor import DogMeasurementExtractor


SMAL_MODEL_PATH = "bite_release/data/smal_data/new_dog_models/my_smpl_39dogsnorm_Jr_4_dog.pkl"


def load_results(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze(results: list, extractor: DogMeasurementExtractor,
            ground_truth: dict = None, default_weight: float = 10.0):
    print(f"{'='*60}")
    print(f"Day 0 Spike 분석 결과")
    print(f"{'='*60}")

    successful = [r for r in results if "vertices" in r]
    failed = [r for r in results if "error" in r]

    print(f"\n추론 결과: 성공 {len(successful)} / 실패 {len(failed)} / 총 {len(results)}")

    if failed:
        print(f"\n실패 목록:")
        for r in failed:
            print(f"  {r['name']}: {r['error']}")

    if not successful:
        print("\n[SPIKE 1 FAIL] 성공한 추론 없음")
        return

    print(f"\n[SPIKE 1 PASS] 3D 메시 {len(successful)}개 생성 성공")

    all_measurements = []
    for r in successful:
        verts = r["vertices"]
        faces = r["faces"]

        weight = r.get("weight_kg", default_weight)
        scale = extractor.estimate_scale_factor(weight)
        m = extractor.extract(verts, faces, scale_factor=scale)
        m["name"] = r["name"]
        all_measurements.append(m)

        print(f"\n  {r['name']}:")
        print(f"    목둘레: {m['neck_circumference_cm']:.1f} cm")
        print(f"    가슴둘레: {m['chest_circumference_cm']:.1f} cm")
        print(f"    등길이: {m['back_length_cm']:.1f} cm")
        print(f"    (추론시간: {r.get('elapsed_sec', 0):.0f}초)")

    if ground_truth:
        print(f"\n{'='*60}")
        print(f"Ground Truth 비교 (Spike 2)")
        print(f"{'='*60}")

        errors = []
        for m in all_measurements:
            name = m["name"]
            if name in ground_truth:
                gt = ground_truth[name]
                err_back = abs(m["back_length_cm"] - gt.get("back_length_cm", 0))
                err_chest = abs(m["chest_circumference_cm"] - gt.get("chest_circumference_cm", 0))
                errors.append({"name": name, "back_err": err_back, "chest_err": err_chest})
                print(f"  {name}: 등길이 오차 {err_back:.1f}cm, 가슴둘레 오차 {err_chest:.1f}cm")

        if errors:
            n_pass = sum(1 for e in errors if e["back_err"] <= 3.0 and e["chest_err"] <= 3.0)
            n_total = len(errors)
            pass_rate = n_pass / n_total
            print(f"\n  3cm 이내: {n_pass}/{n_total} ({pass_rate:.0%})")
            print(f"  기준: 70% 이상이면 PASS")
            print(f"  판정: {'[SPIKE 2 PASS]' if pass_rate >= 0.7 else '[SPIKE 2 FAIL]'}")

    visualize(successful)


def visualize(results: list):
    n = min(len(results), 6)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, r in enumerate(results[:n]):
        verts = r["vertices"]
        ax = axes[0, i]
        ax.scatter(verts[:, 0], verts[:, 2], s=0.3, alpha=0.3, c=verts[:, 0], cmap="viridis")
        ax.set_title(r["name"], fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")

        ax = axes[1, i]
        ax.scatter(verts[:, 0], verts[:, 1], s=0.3, alpha=0.3, c=verts[:, 0], cmap="viridis")
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    axes[0, 0].set_ylabel("Side (Z)")
    axes[1, 0].set_ylabel("Top (Y)")
    plt.tight_layout()
    out = Path("outputs/spike_meshes.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\n시각화 저장: {out}")


def main():
    parser = argparse.ArgumentParser(description="Day 0 Spike 결과 분석")
    parser.add_argument("--results", required=True, help="spike_results.pkl 경로")
    parser.add_argument("--ground-truth", help="ground truth CSV (name,back_length_cm,chest_circumference_cm)")
    parser.add_argument("--weight", type=float, default=10.0, help="기본 체중 (kg). GT CSV에 weight_kg 열이 있으면 무시됨")
    parser.add_argument("--smal-model", default=SMAL_MODEL_PATH, help="SMAL 모델 경로")
    args = parser.parse_args()

    extractor = DogMeasurementExtractor.from_smal_model(args.smal_model)
    results = load_results(args.results)

    gt = None
    if args.ground_truth:
        import csv
        gt = {}
        with open(args.ground_truth) as f:
            for row in csv.DictReader(f):
                gt[row["name"]] = {k: float(v) for k, v in row.items() if k != "name"}

    analyze(results, extractor, ground_truth=gt, default_weight=args.weight)


if __name__ == "__main__":
    main()
