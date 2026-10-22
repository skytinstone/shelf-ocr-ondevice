"""책등·표지 통합 검출기(YOLO11n-OBB, 2클래스) 학습용 합성 데이터 생성

재료(datasets/synth_assets)의 책등·표지 크롭을 배경 위에 여러 방향·크기로 붙여 정확한 회전 상자 라벨을 만든다.
장면: 가까이 찍은 표지 / 가까이 찍은 책등 / 눕혀 쌓은 책(+세운 책) / 책장 줄 / 표지 진열 / 흩어진 책
방해물(기둥·막대·상자·원형 소품)은 라벨 없이 붙여 오검출을 줄인다. LLM은 쓰지 않는다 (라벨 = 붙인 위치).

사용법 (shelf_ocr_test 폴더에서):  python tools/synth/make_dataset.py --train 1600 --val 200
결과: datasets/books_obb/{images,labels}/{train,val}, datasets/books_obb/data.yaml  (클래스 0 spine, 1 cover)
"""
import sys, os, random, argparse, math
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "datasets" / "synth_assets"
OUT = ROOT / "datasets" / "books_obb"
ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=1600); ap.add_argument("--val", type=int, default=200)
ap.add_argument("--size", type=int, default=1024); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-real", action="store_true", help="기존 실사진(자동 라벨)을 학습에 넣지 않음")
args = ap.parse_args()
rng = random.Random(args.seed); np.random.seed(args.seed)
SZ = args.size

load = lambda d: [cv2.imread(str(p)) for p in sorted((A / d).glob("*.jpg"))]
SPINES, COVERS, BGS = load("spine"), load("cover"), load("bg")
print(f"재료: 책등 {len(SPINES)} · 표지 {len(COVERS)} · 배경 {len(BGS)}")


# ---------------------------------------------------------------- 배경
def proc_bg():
    t = rng.random()
    if t < 0.35:   # 나뭇결 책상
        base = np.array([rng.randint(120, 210), rng.randint(140, 200), rng.randint(160, 230)], np.float32)
        y = np.arange(SZ)[:, None].astype(np.float32)
        x = np.arange(SZ)[None, :].astype(np.float32)
        f = rng.uniform(0.01, 0.04); ph = rng.uniform(0, 6.28)
        grain = 0.5 + 0.5 * np.sin(f * (x if rng.random() < 0.5 else y) + 6 * np.sin(0.004 * (y if rng.random() < 0.5 else x)) + ph)
        img = np.broadcast_to(base[None, None, :] * (0.8 + 0.25 * grain[..., None]), (SZ, SZ, 3)).copy()
    elif t < 0.65:   # 벽·책상 단색 + 그라데이션
        c = np.array([rng.randint(60, 250) for _ in range(3)], np.float32)
        g = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.15), SZ, dtype=np.float32)
        img = np.broadcast_to(c[None, None, :] * (g[:, None, None] if rng.random() < 0.5 else g[None, :, None]), (SZ, SZ, 3)).copy()
    else:   # 선반·사무실 느낌: 수평 띠 여러 개
        img = np.full((SZ, SZ, 3), rng.randint(150, 240), np.float32)
        for _ in range(rng.randint(1, 4)):
            y0 = rng.randint(0, SZ - 40); h = rng.randint(15, 80)
            img[y0:y0 + h] = [rng.randint(80, 250) for _ in range(3)]
    img += np.random.normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def real_bg():
    b = rng.choice(BGS)
    b = cv2.resize(b, (SZ, SZ), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5: b = cv2.flip(b, 1)
    return b


def background():
    return real_bg() if BGS and rng.random() < 0.3 else proc_bg()


# ---------------------------------------------------------------- 붙이기
def jitter(im):
    im = im.astype(np.float32) * rng.uniform(0.75, 1.2) + rng.uniform(-20, 20)
    if rng.random() < 0.2:
        im = cv2.GaussianBlur(im, (0, 0), rng.uniform(0.5, 1.5))
    return np.clip(im, 0, 255).astype(np.uint8)


def rect_poly(cx, cy, w, h, ang):
    return cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float32)


def place(canvas, asset, cx, cy, w, h, ang):
    """asset을 (w×h) 크기로 줄이고 ang도 회전해 (cx,cy)에 붙임. 원본 가로·세로가 그대로 w·h가 되도록"""
    a = cv2.resize(jitter(asset), (max(2, int(w)), max(2, int(h))), interpolation=cv2.INTER_AREA)
    src = np.float32([[0, 0], [a.shape[1], 0], [a.shape[1], a.shape[0]], [0, a.shape[0]]])
    t = math.radians(ang); c, s = math.cos(t), math.sin(t)
    dst = np.float32([[cx + (x - w / 2) * c - (y - h / 2) * s, cy + (x - w / 2) * s + (y - h / 2) * c] for x, y in [(0, 0), (w, 0), (w, h), (0, h)]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(a, M, (SZ, SZ), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(np.full(a.shape[:2], 255, np.uint8), M, (SZ, SZ), flags=cv2.INTER_LINEAR)
    m = (mask.astype(np.float32) / 255)[..., None]
    canvas[:] = (canvas * (1 - m) + warped * m).astype(np.uint8)
    return dst


def visible_frac(poly):
    a = cv2.contourArea(poly)
    clip, _ = cv2.intersectConvexConvex(poly, np.float32([[0, 0], [SZ, 0], [SZ, SZ], [0, SZ]]))
    return clip / max(a, 1e-6)


def overlaps(poly, placed, tol=0.04):
    a = cv2.contourArea(poly)
    return any(cv2.intersectConvexConvex(poly, q)[0] > tol * min(a, cv2.contourArea(q)) for q, _ in placed)


def try_place(canvas, placed, asset, cls, cx, cy, w, h, ang, need_vis=0.6, allow_overlap=False):
    poly = rect_poly(cx, cy, w, h, ang)
    if visible_frac(poly) < need_vis or (not allow_overlap and overlaps(poly, placed)):
        return False
    place(canvas, asset, cx, cy, w, h, ang)
    placed.append((poly, cls)); return True


def spine_dims(asset, length):
    h, w = asset.shape[:2]          # 재료 책등은 세로로 긴 이미지
    return max(8, length * w / h), length


def cover_dims(asset, long_side):
    h, w = asset.shape[:2]
    k = long_side / max(h, w); return w * k, h * k


Q = [0, 90, 180, 270]


# ---------------------------------------------------------------- 장면
def scene_close_cover(cv, placed):
    a = rng.choice(COVERS); w, h = cover_dims(a, SZ * rng.uniform(0.5, 0.95))
    ang = rng.choice(Q) + rng.uniform(-12, 12)
    try_place(cv, placed, a, 1, SZ / 2 + rng.uniform(-0.12, 0.12) * SZ, SZ / 2 + rng.uniform(-0.12, 0.12) * SZ, w, h, ang, need_vis=0.75)
    if rng.random() < 0.4:   # 가장자리에 다른 책 일부
        b = rng.choice(SPINES); sw, sh = spine_dims(b, SZ * rng.uniform(0.5, 0.9))
        try_place(cv, placed, b, 0, rng.choice([sw * 0.4, SZ - sw * 0.4]), SZ / 2, sw, sh, rng.choice([0, 180]) + rng.uniform(-5, 5), need_vis=0.5)


def scene_close_spine(cv, placed):
    a = rng.choice(SPINES); w, h = spine_dims(a, SZ * rng.uniform(0.6, 0.98))
    ang = rng.choice(Q) + rng.uniform(-10, 10)
    try_place(cv, placed, a, 0, SZ / 2 + rng.uniform(-0.1, 0.1) * SZ, SZ / 2 + rng.uniform(-0.15, 0.15) * SZ, w, h, ang, need_vis=0.8)
    if rng.random() < 0.4:   # 아래·위에 걸친 다른 책
        b = rng.choice(COVERS + SPINES); cls = 1 if any(b is c for c in COVERS) else 0
        bw, bh = cover_dims(b, SZ * 0.6) if cls else spine_dims(b, SZ * 0.8)
        try_place(cv, placed, b, cls, SZ / 2, rng.choice([0.05, 0.95]) * SZ, bw, bh, rng.choice(Q) + rng.uniform(-8, 8), need_vis=0.5)


def scene_stack(cv, placed):
    n = rng.randint(2, 5); L = SZ * rng.uniform(0.45, 0.85); y = SZ * rng.uniform(0.75, 0.95)
    x0 = SZ * rng.uniform(0.08, 0.5) + L / 2 if rng.random() < 0.5 else SZ / 2
    x0 = min(x0, SZ - L / 2 - 5)
    for _ in range(n):
        a = rng.choice(SPINES); th, _ = spine_dims(a, L * rng.uniform(0.85, 1.1))
        th = min(th, SZ * 0.15)
        cy = y - th / 2
        ang = rng.choice([90, 270]) + rng.uniform(-3, 3)            # 눕힌 책: 긴 변이 가로
        if not try_place(cv, placed, a, 0, x0 + rng.uniform(-0.04, 0.04) * L, cy, th, L * rng.uniform(0.9, 1.05), ang, need_vis=0.7):
            break
        y -= th + rng.uniform(0, 4)
        if y < SZ * 0.15: break
    if rng.random() < 0.6:   # 옆에 세운 책
        a = rng.choice(SPINES); w, h = spine_dims(a, SZ * rng.uniform(0.5, 0.8))
        bottom = SZ * rng.uniform(0.85, 0.97)
        try_place(cv, placed, a, 0, min(SZ - w / 2 - 2, x0 + L / 2 + w / 2 + rng.uniform(5, 40)), bottom - h / 2, w, h, rng.choice([0, 180]) + rng.uniform(-4, 4), need_vis=0.75)
    if rng.random() < 0.3:   # 위에 눕힌 표지
        b = rng.choice(COVERS); bw, bh = cover_dims(b, L * 0.8)
        try_place(cv, placed, b, 1, x0, max(bh / 2, y - bh * 0.35), bw, bh * 0.6, rng.choice(Q) + rng.uniform(-6, 6), need_vis=0.6)


def scene_shelf(cv, placed):
    rows = rng.choice([1, 1, 2])
    for r in range(rows):
        base = SZ * (rng.uniform(0.85, 0.97) if rows == 1 else (0.48 + 0.48 * r))
        H = SZ * rng.uniform(0.3, 0.75) / rows
        cv2.rectangle(cv, (0, int(base)), (SZ, int(base) + rng.randint(8, 25)), [rng.randint(170, 250)] * 3, -1)   # 선반 판
        x = rng.uniform(-10, 60); flip = rng.random() < 0.15
        while x < SZ - 10:
            a = rng.choice(SPINES); w, h = spine_dims(a, H * rng.uniform(0.75, 1.1))
            tilt = rng.uniform(-3, 3) if rng.random() < 0.8 else rng.uniform(-12, 12)
            ok = try_place(cv, placed, a, 0, x + w / 2, base - h / 2, w, h, (180 if flip else 0) + tilt, need_vis=0.6)
            x += w + (rng.uniform(0, 3) if ok else 10)


def scene_display(cv, placed):
    n = rng.randint(3, 12); cols = rng.randint(2, 4); L = SZ / cols * rng.uniform(0.6, 0.9)
    for i in range(n * 3):
        if len(placed) >= n: break
        a = rng.choice(COVERS); w, h = cover_dims(a, L)
        try_place(cv, placed, a, 1, rng.uniform(w / 2, SZ - w / 2), rng.uniform(h / 2, SZ - h / 2), w, h, rng.choice([0, 0, 0, 180, 90, 270]) + rng.uniform(-8, 8), need_vis=0.65)


def scene_mixed(cv, placed):
    for _ in range(rng.randint(3, 9) * 3):
        if len(placed) >= 9: break
        if rng.random() < 0.5:
            a = rng.choice(COVERS); w, h = cover_dims(a, SZ * rng.uniform(0.2, 0.45)); cls = 1
        else:
            a = rng.choice(SPINES); w, h = spine_dims(a, SZ * rng.uniform(0.3, 0.7)); cls = 0
        try_place(cv, placed, a, cls, rng.uniform(0, SZ), rng.uniform(0, SZ), w, h, rng.uniform(0, 360), need_vis=0.65)


SCENES = [(scene_close_cover, 0.2), (scene_close_spine, 0.15), (scene_stack, 0.22), (scene_shelf, 0.18), (scene_display, 0.13), (scene_mixed, 0.12)]


def distractors(cv, placed):
    """라벨 없는 방해물: 선반 기둥·막대(세로·가로), 상자, 원형 소품. 책과 겹치지 않게 먼저 그림"""
    for _ in range(rng.randint(0, 4)):
        col = [rng.randint(30, 240) for _ in range(3)]; t = rng.random()
        if t < 0.4:      # 기둥·막대
            w = rng.randint(8, 40); L = rng.randint(SZ // 3, SZ)
            if rng.random() < 0.5:
                x = rng.randint(0, SZ - w); cv2.rectangle(cv, (x, rng.randint(0, SZ - L)), (x + w, SZ), col, -1)
                for k in range(rng.randint(0, 12)):   # 선반 기둥 구멍
                    yy = rng.randint(0, SZ); cv2.rectangle(cv, (x + w // 3, yy), (x + 2 * w // 3, yy + 6), [max(0, c - 80) for c in col], -1)
            else:
                y = rng.randint(0, SZ - w); cv2.rectangle(cv, (0, y), (SZ, y + w), col, -1)
        elif t < 0.75:   # 상자
            w, h = rng.randint(60, 300), rng.randint(60, 260); x, y = rng.randint(0, SZ - w), rng.randint(0, SZ - h)
            cv2.rectangle(cv, (x, y), (x + w, y + h), col, -1); cv2.rectangle(cv, (x, y), (x + w, y + h), [max(0, c - 60) for c in col], 3)
        else:            # 원형 소품
            cv2.circle(cv, (rng.randint(0, SZ), rng.randint(0, SZ)), rng.randint(20, 120), col, -1)


def finish(cv, placed):
    """원근 왜곡·조명·노이즈·JPEG 품질"""
    d = SZ * rng.uniform(0, 0.07)
    src = np.float32([[0, 0], [SZ, 0], [SZ, SZ], [0, SZ]])
    dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
    M = cv2.getPerspectiveTransform(src, dst)
    cv = cv2.warpPerspective(cv, M, (SZ, SZ), borderMode=cv2.BORDER_REFLECT)
    polys = [(cv2.perspectiveTransform(p[None], M)[0], c) for p, c in placed]
    g = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.95, 1.2), SZ, dtype=np.float32)
    cv = cv.astype(np.float32) * (g[:, None, None] if rng.random() < 0.5 else g[None, :, None])
    cv += np.random.normal(0, rng.uniform(0, 6), cv.shape)
    cv = np.clip(cv, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = rng.uniform(0.4, 0.8); cv = cv2.resize(cv2.resize(cv, None, fx=k, fy=k), (SZ, SZ))
    return cv, polys


def make_one():
    cv = background(); placed = []
    distractors(cv, placed)
    f = rng.choices([s for s, _ in SCENES], weights=[w for _, w in SCENES])[0]
    f(cv, placed)
    return finish(cv, placed)


def write(split, i, img, polys, H=SZ, W=SZ):
    cv2.imwrite(str(OUT / "images" / split / f"syn_{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
    lines = []
    for p, c in polys:
        p = np.clip(p, 0, [W - 1, H - 1]) / [W, H]
        lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in p.ravel()))
    (OUT / "labels" / split / f"syn_{i:05d}.txt").write_text("\n".join(lines))


for split in ["train", "val"]:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True); (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
for split, n in [("train", args.train), ("val", args.val)]:
    for i in range(n):
        img, polys = make_one()
        write(split, i, img, polys)
        if (i + 1) % 200 == 0: print(split, i + 1, flush=True)

# ---------------------------------------------------------------- 기존 실사진 (현재 검출기 결과를 라벨로, 테스트 책 사진 제외)
if not args.no_real:
    sys.path.insert(0, str(ROOT / "src")); os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from shelfkit import BookDetector, detect_image
    det = BookDetector(ROOT / "models" / "spine_obb_best.pt", ROOT / "models" / "yolo11x.pt", imgsz=960, conf=0.45, iou=0.5)
    for p in sorted((ROOT / "images").rglob("*.jpg")):
        if p.stem in {"home_campbell", "home_kreyszig"}:
            continue
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        d = detect_image(img, det); small = d["img"]; H, W = small.shape[:2]
        cv2.imwrite(str(OUT / "images" / "train" / f"real_{p.stem}.jpg"), small)
        lines = [f"{1 if b['kind'] == 'cover' else 0} " + " ".join(f"{v:.6f}" for v in (np.clip(np.asarray(b['poly'], np.float32), 0, [W - 1, H - 1]) / [W, H]).ravel()) for b in d["found"]]
        (OUT / "labels" / "train" / f"real_{p.stem}.txt").write_text("\n".join(lines))
        print("real", p.name, len(lines), flush=True)

(OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: spine\n  1: cover\n", encoding="utf-8")
print("완료:", OUT)
