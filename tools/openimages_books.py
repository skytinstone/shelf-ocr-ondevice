"""Open Images V6 'Book' 실사진을 골라 받아 통합 검출기 학습 데이터(datasets/books_obb)에 더하기

합성 데이터만으로는 실제 사진(책상 위 물건·모니터·그림자·원근)과의 차이가 커서, 사람이 라벨을 단 실사진을 섞는다.
고르는 기준: 책 상자가 '여러 권 묶음(IsGroupOf)'·'그림 속 책(IsDepiction)'이 아니고, 1~12개이며, 상자가 사진의 1% 이상인 사진.
라벨 변환: Open Images 상자는 회전이 없는 직사각형 → 각도 0의 회전 상자. 긴 변 ÷ 짧은 변 ≥ 3 이면 책등, 아니면 표지.
라이선스: 라벨 CC BY 4.0, 사진은 사진마다 CC BY 2.0 (Flickr). 받은 사진은 저장소에 올리지 않는다 (.gitignore).

사용법 (shelf_ocr_test 폴더에서):
  python tools/openimages_books.py --n 1500
  (라벨 파일 oidv6-train-annotations-bbox.csv 는 datasets/openimages/ 에 먼저 받아 둔다)
결과: datasets/books_obb/{images,labels}/train/oi_*.jpg|txt
"""
import argparse, random, concurrent.futures as cf, urllib.request
from pathlib import Path
import numpy as np, cv2, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OI = ROOT / "datasets" / "openimages"
OUT = ROOT / "datasets" / "books_obb"
BOOK = "/m/0bt_c3"
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-side", type=int, default=1024); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()

rows = []
for ch in pd.read_csv(OI / "oidv6-train-annotations-bbox.csv", chunksize=2_000_000,
                      usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"]):
    rows.append(ch[ch.LabelName == BOOK])
b = pd.concat(rows); b["area"] = (b.XMax - b.XMin) * (b.YMax - b.YMin)
g = b.groupby("ImageID").agg(n=("area", "size"), grp=("IsGroupOf", "max"), dep=("IsDepiction", "max"), amin=("area", "min"))
ok = g[(g.grp == 0) & (g.dep == 0) & (g.n <= 12) & (g.amin >= 0.01)].index.tolist()
print(f"책 상자 {len(b)} · 사진 {g.shape[0]} · 조건에 맞는 사진 {len(ok)}")
random.Random(a.seed).shuffle(ok); pick = ok[: a.n]
(OUT / "images" / "train").mkdir(parents=True, exist_ok=True); (OUT / "labels" / "train").mkdir(parents=True, exist_ok=True)


def fetch(iid):
    dst = OUT / "images" / "train" / f"oi_{iid}.jpg"
    if not dst.exists():
        try:
            data = urllib.request.urlopen(f"https://open-images-dataset.s3.amazonaws.com/train/{iid}.jpg", timeout=30).read()
        except Exception:
            return False
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        k = min(1.0, a.max_side / max(img.shape[:2]))
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    img = cv2.imread(str(dst)); H, W = img.shape[:2]
    lines = []
    for r in b[b.ImageID == iid].itertuples():
        w, h = (r.XMax - r.XMin) * W, (r.YMax - r.YMin) * H
        cls = 0 if max(w, h) >= 3 * min(w, h) else 1          # 가늘고 길면 책등, 아니면 표지
        pts = [(r.XMin, r.YMin), (r.XMax, r.YMin), (r.XMax, r.YMax), (r.XMin, r.YMax)]
        lines.append(f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts))
    (OUT / "labels" / "train" / f"oi_{iid}.txt").write_text("\n".join(lines))
    return True


with cf.ThreadPoolExecutor(a.workers) as ex:
    done = sum(ex.map(fetch, pick))
print(f"받음 {done} / {len(pick)} → {OUT / 'images' / 'train'}")
