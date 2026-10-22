"""책등·표지 통합 검출기(YOLO11n-OBB, 2클래스) 학습

데이터: tools/synth/make_dataset.py 로 만든 datasets/books_obb (합성 + 기존 실사진 자동 라벨)
모든 방향(0~360°)·상하/좌우 뒤집기로 학습해 근접 촬영에서 사진이 돌아가거나 뒤집혀도 같은 방식으로 검출한다.
테스트 사진(eval/books)은 학습에 쓰지 않는다.

사용법 (shelf_ocr_test 폴더에서):
  python tools/train_detector.py                 # 본 학습 (M3 Pro MPS 기준 약 3시간)
  python tools/train_detector.py --epochs 1 --fraction 0.1   # 동작 확인용
결과: models/books_obb_best.pt  (노트북·평가 스크립트의 UNIFIED_WEIGHTS)
"""
import argparse, shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--imgsz", type=int, default=768)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--device", default="mps")
a = ap.parse_args()

m = YOLO(str(ROOT / "models" / "yolo11n-obb.pt"))          # DOTA 사전학습 OBB → 책등/표지 2클래스로 미세조정
r = m.train(data=str(ROOT / "datasets" / "books_obb" / "data.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            fraction=a.fraction, device=a.device, workers=0, patience=12,
            degrees=180, flipud=0.5, fliplr=0.5, mosaic=0.7, scale=0.6, translate=0.1, hsv_v=0.5,
            project=str(ROOT / "outputs" / "train"), name="books_obb", exist_ok=True)
best = ROOT / "outputs" / "train" / "books_obb" / "weights" / "best.pt"
shutil.copy(best, ROOT / "models" / "books_obb_best.pt")
print("저장:", ROOT / "models" / "books_obb_best.pt")
