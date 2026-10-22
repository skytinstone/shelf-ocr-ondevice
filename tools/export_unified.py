"""통합 검출기(models/books_obb_best.pt)를 웹 데모용 ONNX로 변환 → web/e_unified.onnx

웹앱(app.html)은 web/e_unified.onnx 가 있으면 기존 책등·표지(58MB) 모델 대신 이 모델 하나로 책을 검출한다.
사용법 (shelf_ocr_test 폴더에서):  python tools/export_unified.py
"""
import shutil, json, argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--imgsz", type=int, default=768, help="입력 크기 (학습 크기와 같게: 768)")
ap.add_argument("--zoom", type=float, default=0.3, help="웹앱에서 사진 둘레 여백을 붙여 한 번 더 검출할 비율 (0 = 안 함)")
a = ap.parse_args()
src = ROOT / "models" / "books_obb_best.pt"
out = YOLO(str(src), task="obb").export(format="onnx", imgsz=a.imgsz, opset=17, simplify=True, dynamic=False)
dst = ROOT / "web" / "e_unified.onnx"
shutil.copy(out, dst)
json.dump({"imgsz": a.imgsz, "zoom": a.zoom}, open(ROOT / "web" / "e_unified.json", "w"))   # 웹앱이 이 값으로 입력 크기·여백 검출을 맞춤

import onnxruntime as ort, numpy as np
s = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
o = s.run(None, {s.get_inputs()[0].name: np.zeros((1, 3, a.imgsz, a.imgsz), np.float32)})[0]
print("저장:", dst, f"{dst.stat().st_size / 1e6:.1f}MB", "· 입력", s.get_inputs()[0].shape, "· 출력", o.shape, "(= 1, 4+2클래스+각도, 상자 수)")
