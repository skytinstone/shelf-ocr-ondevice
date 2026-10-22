"""PC 평가: 노트북과 같은 E 파이프라인으로 eval/images 9장을 돌려 Claude 판독 정답과 비교

사용법 (shelf_ocr_test 폴더에서):
  python eval/run_pc_eval.py            # DB 있는 조건 (eval/catalog_db.json, 230권)
  python eval/run_pc_eval.py --nodb     # DB 없는 조건 (기존 데모 카탈로그만, 테스트 책 제외)
결과: eval/res_pc_db.json → eval/score_pc_db.json (또는 _nodb)
"""
import sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; EV = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import os; os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import numpy as np, cv2
import shelfkit
from shelfkit import BookDetector, PaddleRecEngine, TitleRuleReader, TitleMatcher, detect_image, run_image, load_catalog, load_non_titles

ap = argparse.ArgumentParser(); ap.add_argument("--nodb", action="store_true"); a = ap.parse_args()
tag = "nodb" if a.nodb else "db"
if a.nodb:
    catalog = load_catalog(ROOT)                       # 기존 데모 카탈로그 (my_titles.txt 포함)
    gt_titles = set()
    for g in json.load(open(EV / "gt.json", encoding="utf-8")).values(): gt_titles |= set(g["titles"])
    catalog = [c for c in catalog if c["title"] not in gt_titles]   # 테스트 사진 속 책은 뺌 = 순수 OCR 조건
else:
    catalog = [{"isbn": f"db-{i}", "title": t} for i, t in enumerate(json.load(open(EV / "catalog_db.json", encoding="utf-8")))]
load_non_titles(EV)                                     # eval/catalog/non_titles.txt (저자·출판사 189개)
matcher = TitleMatcher(catalog)
print(f"[{tag}] 카탈로그 {len(catalog)}권")

det = BookDetector(ROOT / "models" / "spine_obb_best.pt", ROOT / "models" / "yolo11x.pt", imgsz=960, conf=0.45, iou=0.5, min_aspect=2.5)
ko = PaddleRecEngine("korean_PP-OCRv5_mobile_rec", "ko"); en = PaddleRecEngine("en_PP-OCRv5_mobile_rec", "en")
reader = TitleRuleReader(ko, en, det.textdet)

def imread(p, max_side=1600):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    k = min(1.0, max_side / max(img.shape[:2]))
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img

out = {}
for p in sorted((EV / "images").glob("*.jpg")):
    img = imread(p, max_side=3000); t0 = time.perf_counter()   # 글자는 고해상도에서 읽음 (검출은 내부에서 1600)
    d = detect_image(img, det); r = run_image(img, det, reader, matcher, det=d)
    H, W = r["img"].shape[:2]
    out[p.name] = {"W": W, "H": H, "tDet": d["t_det"] * 1000, "tAll": (time.perf_counter() - t0) * 1000,
                   "books": [{"kind": b["kind"], "final": b["final"], "source": b["source"], "raw": b.get("raw") or "", "bbox": [round(v) for v in b["bbox"]]} for b in r["books"]]}
    print(f"{p.name}: {len(r['books'])}권 · {out[p.name]['tAll'] / 1000:.1f}s", flush=True)
    json.dump(out, open(EV / f"res_pc_{tag}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.system(f'"{sys.executable}" "{EV / "score.py"}" "{EV / "gt.json"}" "{EV / f"res_pc_{tag}.json"}" "{EV / f"score_pc_{tag}.json"}"')
