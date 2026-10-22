"""통합 검출기(models/books_obb_best.pt)를 웹 데모용 ONNX로 변환 → web/e_unified.onnx

웹앱(app.html)은 web/e_unified.onnx 가 있으면 기존 책등·표지(58MB) 모델 대신 이 모델 하나로 책을 검출한다.
사용법 (shelf_ocr_test 폴더에서):  python tools/export_unified.py
"""
import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "models" / "books_obb_best.pt"
out = YOLO(str(src), task="obb").export(format="onnx", imgsz=960, opset=17, simplify=True, dynamic=False)
dst = ROOT / "web" / "e_unified.onnx"
shutil.copy(out, dst)

import onnxruntime as ort, numpy as np
s = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
o = s.run(None, {s.get_inputs()[0].name: np.zeros((1, 3, 960, 960), np.float32)})[0]
print("저장:", dst, f"{dst.stat().st_size / 1e6:.1f}MB", "· 입력", s.get_inputs()[0].shape, "· 출력", o.shape, "(= 1, 4+2클래스+각도, 상자 수)")
