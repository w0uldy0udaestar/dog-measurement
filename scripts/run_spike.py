"""
Day 0 Spike 실행 스크립트 (데스크탑에서 실행).

순서:
1. BITE 모델 로드
2. 테스트 이미지들에 대해 추론 실행
3. 결과를 spike_results.pkl로 저장
4. 치수 추출 + 시각화

사용법:
    python scripts/run_spike.py --bite-dir bite_release --images data/test_images/
    python scripts/run_spike.py --bite-dir bite_release --images dog1.jpg dog2.jpg
"""

import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser(description="Day 0 Spike 실행")
    parser.add_argument("--bite-dir", default="bite_release", help="BITE 코드 디렉토리")
    parser.add_argument("--images", nargs="+", required=True, help="이미지 파일 또는 디렉토리")
    parser.add_argument("--output", default="outputs/spike", help="결과 저장 디렉토리")
    parser.add_argument("--no-ttopt", action="store_true", help="test-time optimization 비활성화 (빠르지만 부정확)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    image_paths = []
    for path in args.images:
        if os.path.isdir(path):
            image_paths.extend(sorted(glob.glob(os.path.join(path, "*.jpg"))))
            image_paths.extend(sorted(glob.glob(os.path.join(path, "*.jpeg"))))
            image_paths.extend(sorted(glob.glob(os.path.join(path, "*.png"))))
        elif os.path.isfile(path):
            image_paths.append(path)

    if not image_paths:
        print("[ERROR] 이미지를 찾을 수 없습니다.")
        return

    print(f"이미지 {len(image_paths)}장 발견")
    os.makedirs(args.output, exist_ok=True)

    # 우리 프로젝트의 src/와 BITE의 src/가 이름 충돌하므로
    # bite_runner.py를 importlib로 직접 로드 (src 패키지 우회)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bite_runner",
        os.path.join(PROJECT_ROOT, "src", "inference", "bite_runner.py"),
    )
    bite_runner_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bite_runner_mod)
    BITERunner = bite_runner_mod.BITERunner

    print(f"\n{'='*60}")
    print(f"BITE 모델 로드 중...")
    print(f"{'='*60}")
    runner = BITERunner(
        bite_dir=args.bite_dir,
        device=args.device,
        apply_ttopt=not args.no_ttopt,
    )

    results = []
    total_start = time.time()

    for idx, img_path in enumerate(image_paths):
        name = Path(img_path).stem
        print(f"\n[{idx+1}/{len(image_paths)}] {name}")
        t0 = time.time()

        try:
            result = runner.run(img_path)
            elapsed = time.time() - t0
            result["elapsed_sec"] = elapsed
            result["name"] = name
            results.append(result)
            print(f"  OK ({elapsed:.0f}초) | vertices: {result['vertices'].shape}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAIL ({elapsed:.0f}초): {e}")
            results.append({"name": name, "error": str(e), "image_path": img_path})

    total_elapsed = time.time() - total_start

    pkl_path = os.path.join(args.output, "spike_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    success_count = sum(1 for r in results if "vertices" in r)
    print(f"\n{'='*60}")
    print(f"Day 0 Spike 완료")
    print(f"{'='*60}")
    print(f"  성공: {success_count}/{len(results)}")
    print(f"  총 시간: {total_elapsed:.0f}초")
    print(f"  결과: {pkl_path}")
    print(f"\n다음 단계:")
    print(f"  python scripts/analyze_spike.py --results {pkl_path}")


if __name__ == "__main__":
    main()
