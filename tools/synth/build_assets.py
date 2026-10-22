"""합성 학습 데이터 재료 만들기: 기존 사진에서 책등·표지 크롭(고해상도)과 배경 조각을 잘라 저장

테스트 공정성: eval/books(테스트 8장)와, 같은 책(Embedded Recipes·회로이론·Kreyszig)이 들어 있는
home_campbell / home_kreyszig 사진은 재료에서 제외한다.
사용법 (shelf_ocr_test 폴더에서):  python tools/synth/build_assets.py
결과: datasets/synth_assets/{spine,cover,bg}/*.jpg
"""
import sys, os, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import numpy as np, cv2
from shelfkit import BookDetector, detect_image, crop_book, _scale_book

EXCLUDE = {"home_campbell", "home_kreyszig"}
OUT = ROOT / "datasets" / "synth_assets"
for d in ["spine", "cover", "bg"]:
    (OUT / d).mkdir(parents=True, exist_ok=True)
random.seed(0)
det = BookDetector(ROOT / "models" / "spine_obb_best.pt", ROOT / "models" / "yolo11x.pt", imgsz=960, conf=0.45, iou=0.5)
n = {"spine": 0, "cover": 0, "bg": 0}
for p in sorted((ROOT / "images").rglob("*.jpg")):
    if p.stem in EXCLUDE:
        continue
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    d = detect_image(img, det)
    small = d["img"]
    k = min(1.0, 3000 / max(img.shape[:2])); full = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    s = full.shape[1] / small.shape[1]
    mask = np.zeros(small.shape[:2], np.uint8)
    for b in d["found"]:
        cv2.fillPoly(mask, [np.int32(b["poly"])], 255)
        if b.get("spiral"):
            continue
        crop, _ = crop_book(full, _scale_book(b, s))
        h, w = crop.shape[:2]
        if min(h, w) < 24:
            continue
        kind = b["kind"]
        if kind == "spine" and h < 3 * w:        # 책등은 세로로 충분히 길어야 재료로 씀
            continue
        if kind == "cover" and max(h, w) > 2.2 * min(h, w):
            continue
        cv2.imwrite(str(OUT / kind / f"{p.stem}_{n[kind]:03d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92]); n[kind] += 1
    # 배경 조각: 책과 거의 겹치지 않는 영역
    mask = cv2.dilate(mask, np.ones((25, 25), np.uint8))
    H, W = small.shape[:2]
    for _ in range(40):
        sz = random.randint(int(0.2 * min(H, W)), int(0.6 * min(H, W)))
        x, y = random.randint(0, W - sz), random.randint(0, H - sz)
        if mask[y:y + sz, x:x + sz].mean() < 0.03 * 255:
            X, Y, S = int(x * s), int(y * s), int(sz * s)
            cv2.imwrite(str(OUT / "bg" / f"{p.stem}_{n['bg']:03d}.jpg"), full[Y:Y + S, X:X + S], [cv2.IMWRITE_JPEG_QUALITY, 90]); n["bg"] += 1
    print(p.name, n, flush=True)
print("완료:", n)
