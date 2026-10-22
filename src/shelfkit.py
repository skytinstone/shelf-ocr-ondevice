"""책장·진열대 사진 → 권수 + 제목 (PC 테스트용 공용 모듈)

파이프라인 (LLM·멀티모달 모델은 사용하지 않음)
  Step 1  책 검출      : 책등 = YOLO11n-OBB(자체 가중치), 표지 = YOLO11 COCO 'book' 클래스 → 두 결과를 규칙으로 병합
  Step 2  강조 글자 선택 : 책 크롭 안에서 글자 영역 검출(PP-OCRv5 det) → 글자 크기 × √길이 로 두드러진 상자 상위 k개
          글자 인식      : 선택한 상자만 인식(B/C/D 엔진)
          제목 확정      : 상자 조합을 카탈로그와 매칭·검증 (비신경망)
비교 기준으로 예전 방식(책등 크롭 전체를 통째로 인식)도 mode="whole" 로 남겨 둔다.
"""
import copy, itertools, json, re, time, unicodedata
from pathlib import Path
import numpy as np
import cv2

from matcher import TitleMatcher, is_garbage, normalize

ACCEPT = 72        # 이 점수 이상이고 검증을 통과해야 제목 확정
EARLY = 80         # (whole 모드) 이 점수 이상이 나오면 남은 판독 방향은 건너뜀
ORDER = ["relayout", "rot_ccw", "rot_cw"]
COCO_BOOK = 73     # COCO 클래스 번호 'book'


# ---------------------------------------------------------------- 카탈로그
def load_catalog(root):
    """catalog/catalog.json(데모 기본) + catalog/my_titles.txt(직접 추가, 한 줄에 한 제목)
    catalog/non_titles.txt(저자·출판사·시리즈명)도 함께 불러온다"""
    root = Path(root)
    load_non_titles(root)
    titles = json.loads((root / "catalog" / "catalog.json").read_text(encoding="utf-8"))
    extra = root / "catalog" / "my_titles.txt"
    if extra.exists():
        for line in extra.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                titles.append(t)
    seen, out = set(), []
    for t in titles:
        t = unicodedata.normalize("NFC", t)
        if t not in seen:
            seen.add(t); out.append({"isbn": f"cat-{len(out)}", "title": t})
    return out


# ---------------------------------------------------------------- Step 1: 책 검출
def _bbox(p):
    p = np.asarray(p)
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _area(a):
    return (a[2] - a[0]) * (a[3] - a[1])


class SpineDetector:
    """YOLO11n-OBB 책등 검출기 + 검출 품질을 높이기 위한 옵션

    imgsz      : 입력 크기. 멀리서 찍어 책등이 가늘면 크게(1280 등)
    conf, iou  : 신뢰도 임계값, NMS 중복 제거 기준
    min_aspect : 긴 변/짧은 변 비율 하한. 책등은 길쭉하므로 표지·선반 옆판 같은 오검출을 걸러냄
    adaptive   : True면 640으로 먼저 찾고, 책등 중앙 폭이 22px(640 기준) 미만이면 1280으로 다시 찾음 (웹 데모 B 규칙)
    """

    def __init__(self, weights, imgsz=640, conf=0.35, iou=0.5, min_aspect=0.0, adaptive=False, device=None):
        from ultralytics import YOLO
        self.m = YOLO(str(weights), task="obb")
        self.imgsz, self.conf, self.iou, self.min_aspect, self.adaptive = imgsz, conf, iou, min_aspect, adaptive
        self.device = device

    def _predict(self, img, imgsz):
        kw = dict(conf=self.conf, iou=self.iou, imgsz=imgsz, verbose=False)
        if self.device:
            kw["device"] = self.device
        r = self.m.predict(img, **kw)[0]
        out = []
        if r.obb is None:
            return out
        polys = r.obb.xyxyxyxy.cpu().numpy()
        scores = r.obb.conf.cpu().numpy()
        for p, s in zip(polys, scores):
            rect = cv2.minAreaRect(p.astype(np.float32))
            (w, h) = rect[1]
            long_, short = max(w, h), max(1e-6, min(w, h))
            if long_ / short < self.min_aspect:
                continue
            out.append({"kind": "spine", "rect": rect, "poly": p.tolist(), "score": float(s), "short": float(short), "bbox": _bbox(p)})
        out.sort(key=lambda s: s["bbox"][0])
        return out

    def __call__(self, img):
        if not self.adaptive:
            return self._predict(img, self.imgsz), str(self.imgsz)
        a = self._predict(img, 640)
        if a:
            k = 640 / max(img.shape[:2])
            med = float(np.median([b["short"] for b in a])) * k
            if med < 22:
                return self._predict(img, 1280), "640→1280"
        return a, "640"


class CoverDetector:
    """표지(앞면) 후보 검출: COCO 사전학습 YOLO11x 'book' 클래스를 입력 크기 여러 개 × (원본, 180도 회전)으로 돌려 후보를 모은 뒤
    같은 책을 가리키는 상자를 하나로 융합한다. 흰 표지·거꾸로 놓인 표지까지 놓치지 않도록 신뢰도 기준은 낮게 두고,
    걸러내는 일은 뒤 단계(포함 관계 정리 · 글자 방향 검사)가 맡는다.
    support = 각 실행(pass)에서의 최고 신뢰도 합 / 실행 수 (여러 번 잡힐수록 높음)"""

    def __init__(self, weights="yolo11x.pt", sizes=(960, 1280, 1600), rotate180=True, conf=0.02, device=None):
        from ultralytics import YOLO
        self.m = YOLO(str(weights))
        self.sizes, self.rotate180, self.conf, self.device = sizes, rotate180, conf, device
        self._cache = {}

    def _detect(self, img):
        H, W = img.shape[:2]
        dets, passes = [], 0
        for rot in ((False, True) if self.rotate180 else (False,)):
            im = cv2.rotate(img, cv2.ROTATE_180) if rot else img
            for sz in self.sizes:
                kw = dict(classes=[COCO_BOOK], conf=self.conf, imgsz=sz, verbose=False)
                if self.device:
                    kw["device"] = self.device
                r = self.m.predict(im, **kw)[0]
                for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()):
                    if rot:
                        x1, y1, x2, y2 = W - x2, H - y2, W - x1, H - y1
                    dets.append(([x1, y1, x2, y2], c, passes))
                passes += 1
        return fuse_boxes(dets, passes)

    def __call__(self, img):
        key = (img.shape, hash(img[::31, ::31].tobytes()))   # 튜닝 때 같은 사진을 반복 검출하지 않도록 (내용 기준)
        if key not in self._cache:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[key] = self._detect(img)
        return self._cache[key]


def _iou(a, b):
    i = _inter(a, b)
    return i / max(1e-6, _area(a) + _area(b) - i)


def fuse_boxes(dets, n_pass, iou_thr=0.55):
    """여러 실행 결과의 상자를 IoU로 묶어 신뢰도 가중 평균 상자 하나로 만든다 (Weighted Box Fusion 방식)"""
    clusters = []
    for b, c, k in sorted(dets, key=lambda d: -d[1]):
        best = max(clusters, key=lambda cl: _iou(cl["bbox"], b), default=None)
        if best is not None and _iou(best["bbox"], b) >= iou_thr:
            best["members"].append((b, c, k))
            w = np.array([m[1] for m in best["members"]])
            best["bbox"] = list((np.array([m[0] for m in best["members"]]) * w[:, None]).sum(0) / w.sum())
        else:
            clusters.append({"bbox": list(b), "members": [(b, c, k)]})
    out = []
    for cl in clusters:
        per_pass = {}
        for _, c, k in cl["members"]:
            per_pass[k] = max(per_pass.get(k, 0), c)
        out.append({"bbox": [float(v) for v in cl["bbox"]], "support": sum(per_pass.values()) / max(1, n_pass), "votes": len(per_pass)})
    return out


def _wh(c):
    x1, y1, x2, y2 = c["bbox"]
    return x2 - x1, y2 - y1


def _cover_shaped(c):
    """표지 모양: 세로가 가로와 같거나 더 긴 직사각형 (표지의 그림 부분·띠지처럼 납작한 조각은 제외)"""
    w, h = _wh(c)
    return h >= 0.9 * w


def resolve_containment(cands, inside=0.8):
    """포함 관계 정리 (큰 상자부터)
    (1) 큰 상자 안을 서로 겹치지 않는 '표지 모양' 상자 2개 이상이 60% 이상 채움 → 두 권을 합친 상자, 큰 상자 제거
    (2) 가로로 넓은 상자 안의 한쪽에 높이가 꽉 찬 표지가 있고 나머지도 표지 모양 → 나머지를 다른 한 권으로 분리
    (3) 큰 상자 안의 상자 하나가 절반 이상을 차지하고 지지도가 비슷함 → 더 꼭 맞는 작은 상자를 채택
    (4) 그 밖에 큰 상자 안에 들어간 작은 상자 → 표지의 일부(그림·띠지·제목 영역)로 보고 제거"""
    cands = sorted(cands, key=lambda c: -_area(c["bbox"]))
    removed, added = set(), []
    for i, B in enumerate(cands):
        if i in removed:
            continue
        aB = _area(B["bbox"]); wB, hB = _wh(B)
        inner = [j for j in range(len(cands)) if j != i and j not in removed and _area(cands[j]["bbox"]) < 0.85 * aB
                 and _inter(cands[j]["bbox"], B["bbox"]) >= inside * _area(cands[j]["bbox"])]
        big = sorted((j for j in inner if _area(cands[j]["bbox"]) >= 0.25 * aB and _cover_shaped(cands[j])),
                     key=lambda j: -cands[j]["support"])
        split = []
        for j in big:
            if all(_inter(cands[j]["bbox"], cands[k]["bbox"]) < 0.2 * _area(cands[j]["bbox"]) for k in split):
                split.append(j)
        if len(split) >= 2 and sum(_area(cands[j]["bbox"]) for j in split) >= 0.6 * aB:
            removed.add(i); continue
        if len(split) == 1:
            J = cands[split[0]]; wJ, hJ = _wh(J)
            x1, y1, x2, y2 = B["bbox"]; jx1, _, jx2, _ = J["bbox"]
            rest = [x1, y1, jx1, y2] if (jx1 + jx2) / 2 > (x1 + x2) / 2 else [jx2, y1, x2, y2]
            rw = rest[2] - rest[0]
            if wB >= 0.9 * hB and hJ >= 0.85 * hB and rw >= 0.6 * wJ and hB >= 0.9 * rw:
                removed.add(i); added.append({**B, "bbox": rest}); continue
            if _area(J["bbox"]) >= 0.5 * aB and J["support"] >= 0.8 * B["support"]:
                removed.add(i); continue
        removed.update(inner)
    return [c for i, c in enumerate(cands) if i not in removed] + added


def merge_belts(cands, gap=0.03):
    """띠지 합치기: 폭이 거의 같고 위아래로 맞닿은 두 상자 중 하나가 납작한 띠 모양이면 한 권의 표지로 합친다"""
    cands = list(cands)
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(cands):
            for j, b in enumerate(cands):
                if i == j:
                    continue
                (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a["bbox"], b["bbox"]
                if ay1 > by1:
                    continue                                   # a가 위, b가 아래인 쌍만
                xo = min(ax2, bx2) - max(ax1, bx1)
                aw, ah = _wh(a); bw, bh = _wh(b)
                touching = by1 - ay2 <= gap * max(ah, bh) * 3 and by1 >= ay1 + 0.5 * ah
                strip = bh < 0.6 * bw or ah < 0.6 * aw
                if xo >= 0.85 * max(aw, bw) and touching and strip:
                    cands[i] = {**a, "bbox": [min(ax1, bx1), ay1, max(ax2, bx2), max(ay2, by2)], "support": max(a["support"], b["support"])}
                    del cands[j]; merged = True; break
            if merged:
                break
    return cands


def split_wide(fused, accepted, thr, max_aspect):
    """넓은 상자의 나머지 분리: 이웃 책과 합쳐 잡혀 너무 넓은(비율 초과) 상자의 한쪽 끝을 이미 채택된 표지가 차지하고 있으면,
    나머지 부분을 별개의 책으로 추가한다 (사진 가장자리에 잘려 이웃과 합쳐 잡힌 책 대비)"""
    if not accepted:
        return []
    med_w = float(np.median([_wh(c)[0] for c in accepted]))
    out = []
    for Wb in fused:
        w, h = _wh(Wb)
        if Wb["support"] < thr or w / max(1.0, h) <= max_aspect:
            continue
        x1, y1, x2, y2 = Wb["bbox"]
        for J in accepted:
            jx1, jy1, jx2, jy2 = J["bbox"]
            if min(y2, jy2) - max(y1, jy1) < 0.6 * h:
                continue
            if jx1 <= x1 + 0.05 * w and x1 + 0.1 * w <= jx2 < x2:
                rest = [jx2, y1, x2, y2]                         # 왼쪽 끝을 J가 차지 → 오른쪽 나머지
            elif jx2 >= x2 - 0.05 * w and x1 < jx1 <= x2 - 0.1 * w:
                rest = [x1, y1, jx1, y2]                         # 오른쪽 끝을 J가 차지 → 왼쪽 나머지
            else:
                continue
            R = {"bbox": rest, "support": Wb["support"], "votes": Wb["votes"], "tier": "A"}
            if _wh(R)[0] >= 0.5 * med_w and all(_inter(rest, k["bbox"]) < 0.2 * _area(rest) for k in accepted + out):
                out.append(R)
    return out


def text_orientation(polys):
    """글자 상자들의 방향: (가로 글줄 비율, 글자 상자 수). 길이로 가중, 정사각형에 가까운 상자는 판단에서 제외"""
    hz = vt = 0.0
    for p in polys:
        x1, y1, x2, y2 = _bbox(p)
        w, h = x2 - x1, y2 - y1
        if w >= 1.3 * h:
            hz += w
        elif h >= 1.3 * w:
            vt += h
    return (hz / (hz + vt) if hz + vt else 0.0), len(polys)


def merge_books(spines, covers, vbooks, coco_boxes, confirm_spines=True, cover_scene=False):
    """책등·표지 병합 규칙
    covers : 글자 방향 검사에서 가로 글줄이 주를 이룬 표지 → 그대로 채택, 그 안에 잡힌 책등(표지 속 글자 줄 등)은 제거
    vbooks : 세로 글줄이 주를 이룬 'book' 상자 → 책등 검출기가 이미 잡은 곳이면 버리고, 아무도 못 잡은 책이면 책등으로 추가
    cover_scene: 표지 위주 사진이면 책등 검출기가 표지 속 글줄을 책등으로 잡기 쉬우므로
      표지와 25% 이상, 한 권짜리 세로 글자 상자와 50% 이상 겹치는 책등을 제거한다 (책등 글자 방향 검사는 BookDetector에서)
    confirm_spines: COCO가 근처에 책을 하나도 보지 못한 책등(바닥 무늬·선반 모서리 등)은 제거"""
    # 상자 안에 책등이 2개 이상이면 책등 여러 권을 하나로 잡은 묶음 → 책등을 우선 (표지 위주 사진의 책등은 글자 방향 검사를 통과한 것)
    n_in = lambda c: sum(_inter(s["bbox"], c["bbox"]) > 0.5 * _area(s["bbox"]) for s in spines)
    covers = [c for c in covers if n_in(c) <= 1]
    single = [v for v in vbooks if n_in(v) <= 1]
    out = []
    for s in spines:
        a = _area(s["bbox"])
        if any(_inter(s["bbox"], c["bbox"]) > 0.5 * a for c in covers):
            continue
        if cover_scene and (any(_inter(s["bbox"], c["bbox"]) > 0.25 * a for c in covers)
                            or any(_inter(s["bbox"], v["bbox"]) > 0.5 * a for v in single)):
            continue
        if confirm_spines and coco_boxes and not any(_inter(s["bbox"], b) > 0.2 * a for b in coco_boxes):
            continue
        out.append(s)
    for v in vbooks:
        taken = out + covers
        if any(_inter(v["bbox"], t["bbox"]) > 0.3 * min(_area(v["bbox"]), _area(t["bbox"])) for t in taken):
            continue
        out.append(v)
    return order_books(out + covers)


def order_books(books):
    """읽는 순서로 번호 매기기: 위 선반 → 아래 선반, 같은 줄은 왼쪽 → 오른쪽"""
    if not books:
        return books
    hs = [b["bbox"][3] - b["bbox"][1] for b in books]
    band = max(1.0, 0.5 * float(np.median(hs)))
    rows = []
    for b in sorted(books, key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2):
        cy = (b["bbox"][1] + b["bbox"][3]) / 2
        if rows and abs(cy - rows[-1][0]) < band:
            rows[-1][1].append(b); rows[-1][0] = np.mean([(x["bbox"][1] + x["bbox"][3]) / 2 for x in rows[-1][1]])
        else:
            rows.append([cy, [b]])
    return [b for _, r in rows for b in sorted(r, key=lambda b: b["bbox"][0])]


def _box_book(b, kind, score):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    rect = ((x1 + w / 2, y1 + h / 2), (w, h), 0.0)
    return {"kind": kind, "rect": rect, "poly": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "score": float(score),
            "short": float(min(w, h)), "bbox": list(b)}


class BookDetector:
    """Step 1: 책등 검출 + 표지 후보 검출 → 지지도·모양 필터 → 포함 관계 정리 → 글자 방향 검사 → 병합
    책등 파라미터(imgsz/conf/iou/min_aspect/adaptive)는 SpineDetector와 같다.
    min_support       : 표지 후보로 남길 최소 지지도
    spine_scene_support: 책등 위주 사진(신뢰도 0.7 이상 책등 8개 이상)에서 쓰는 더 높은 기준 (상자·소품 오검출 억제)
    max_aspect, min_area: 표지 모양(긴 변/짧은 변) 상한, 사진 대비 최소 면적"""

    def __init__(self, spine_weights, cover_weights="yolo11x.pt", textdet=None, covers=True, confirm_spines=True, device=None,
                 min_support=0.12, spine_scene_support=0.35, max_aspect=2.4, min_area=0.003, **spine_kw):
        self.spine = SpineDetector(spine_weights, device=device, **spine_kw)
        self.cover = CoverDetector(cover_weights, device=device) if covers else None
        self.textdet = textdet or (TextBoxDetector() if covers else None)
        self.confirm_spines, self.min_support, self.spine_scene_support = confirm_spines, min_support, spine_scene_support
        self.max_aspect, self.min_area = max_aspect, min_area

    def configure(self, **spine_kw):
        """모델은 공유하고 책등 파라미터만 바꾼 검출기 (튜닝용)"""
        d = copy.copy(self); d.spine = copy.copy(self.spine)
        for k, v in spine_kw.items():
            setattr(d.spine, k, v)
        return d

    def verify_spines(self, img, spines, cover_scene):
        """책등 검증 (모든 사진)
        - 스프링 노트: 스프링 무늬 점수 ≥ 0.2 이고 글자 상자 ≤ 1 → 책으로 세되 spiral=True (제목은 Unknown)
        - 상자·소품: 긴 변/짧은 변 < 2.5 인 뭉툭한 '책등' 제거 (게임기 상자 등)
        - 철봉·벽면·선반 모서리: 스프링이 아니면 글자 상자가 있어야 함 (신뢰도 < 0.65 이면 2개 이상)
        - 중복: 더 확실한 책등과 절반 이상 겹치면 제거
        - 외톨이: 책장 사진에서 다른 책등과 높이 범위가 절반 이상 겹치지 않으면 제거 (선반 기둥·벽면)
        - 바닥 정렬: 책장 사진에서 아래쪽 끝이 다른 책등 2권 이상과 맞지 않으면 제거 (공중에 뜬 기둥)
        - 표지 위주 사진: 세워 놓고 봤을 때 세로 글줄이 2개 이상인 책등만 (표지 위 글줄·바닥 무늬 제거)"""
        kept = []
        for s in spines:
            crop = crop_upright(img, s["rect"])
            s["horizontal"], s["n_text"] = text_orientation(self.textdet(crop))
            s["spiral"] = spiral_score(crop) >= 0.2 and s["n_text"] <= 1
            long_, short = max(s["rect"][1]), max(1.0, min(s["rect"][1]))
            if cover_scene:
                ok = s["n_text"] >= 2 and s["horizontal"] < 0.5
            else:
                ok = long_ / short >= 2.5 and (s["spiral"] or (s["n_text"] >= 1 if s["score"] >= 0.65 else s["n_text"] >= 2))
            if ok:
                kept.append(s)
        kept.sort(key=lambda s: -s["score"])
        out = []
        for s in kept:
            a = cv2.contourArea(np.float32(s["poly"]))
            if all(_poly_inter(s["poly"], o["poly"]) < 0.5 * a for o in out):
                out.append(s)
        if not cover_scene and len(out) >= 5:   # 책이 꽂힌 줄에서 위아래로 벗어난 외톨이 → 선반 기둥·벽면
            def vov(a, b):
                return max(0.0, min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1]))
            out = [s for s in out if any(vov(s, o) >= 0.5 * (s["bbox"][3] - s["bbox"][1]) for o in out if o is not s)]
            # 책은 선반 위에 서 있음 → 아래쪽 끝이 다른 책등 2권 이상과 맞아야 함 (공중에 뜬 기둥·벽면 제거)
            tol = 0.15 * float(np.median([max(o["rect"][1]) for o in out]))
            out = [s for s in out if sum(abs(s["bbox"][3] - o["bbox"][3]) <= tol for o in out if o is not s) >= 2]
        return out

    def cover_candidates(self, img, spines):
        """표지 후보 선택
        A 확실한 후보 : 지지도 ≥ min_support (책등 위주 사진은 spine_scene_support)
        B 약한 후보   : 지지도 ≥ 0.03, 2번 이상 잡힘, A 표지들과 크기·모양이 비슷함
        → 포함 관계 정리 → 띠지 합치기
        C 빈자리 채우기: A·B 표지들이 놓인 줄의 빈자리에 있고 크기·모양이 비슷한 후보 (가로 글자가 있어야 채택, __call__에서 검사)"""
        H, W = img.shape[:2]
        fused = self.cover(img)
        spine_scene = sum(s["score"] >= 0.7 for s in spines) >= 8
        thr = self.spine_scene_support if spine_scene else self.min_support

        def shape_ok(c):
            w, h = _wh(c)
            return max(w, h) / max(1.0, min(w, h)) <= self.max_aspect and w * h >= self.min_area * W * H

        A = [{**c, "tier": "A"} for c in fused if c["support"] >= thr and shape_ok(c)]
        pool = A
        if len(A) >= 3 and not spine_scene:
            areas = [_area(c["bbox"]) for c in A]; ratios = [_wh(c)[1] / max(1.0, _wh(c)[0]) for c in A]
            med_a, med_r = float(np.median(areas)), float(np.median(ratios))

            def similar(c):
                w, h = _wh(c)
                return 0.4 <= w * h / med_a <= 1.8 and 0.7 <= (h / max(1.0, w)) / med_r <= 1.4   # 앞 책에 일부 가려진 표지까지

            B = [{**c, "tier": "B"} for c in fused if thr > c["support"] >= 0.03 and c["votes"] >= 2 and shape_ok(c) and similar(c)]
            pool = A + B
        cands = merge_belts(resolve_containment(pool))
        if len(A) >= 3 and not spine_scene:
            cands += split_wide(fused, cands, thr, self.max_aspect)
        if len(A) >= 3 and not spine_scene:
            for c in sorted(fused, key=lambda c: -c["support"]):
                if c["support"] >= 0.03 or not shape_ok(c) or not similar(c):
                    continue
                x1, y1, x2, y2 = c["bbox"]
                in_row = any(min(y2, k["bbox"][3]) - max(y1, k["bbox"][1]) >= 0.6 * (y2 - y1) for k in cands)
                free = all(_inter(c["bbox"], k["bbox"]) < 0.2 * min(_area(c["bbox"]), _area(k["bbox"])) for k in cands)
                if in_row and free:
                    cands.append({**c, "tier": "C"})
        return cands, [c["bbox"] for c in fused if c["support"] >= 0.03]

    def __call__(self, img):
        spines, mode = self.spine(img)
        raw_spines = list(spines)                          # 디버그용: 걸러내기 전 책등 검출 원본
        if self.cover is None:
            return order_books(spines), mode
        cands, coco = self.cover_candidates(img, spines)
        for c in cands:
            x1, y1, x2, y2 = [int(round(v)) for v in c["bbox"]]
            c["horizontal"], c["n_text"] = text_orientation(self.textdet(img[max(0, y1):y2, max(0, x1):x2]))
        # 표지 위주 사진: 가로 글줄 표지가 4권 이상이고 확실한 책등 수의 절반 이상
        n_hz = sum(c["horizontal"] >= 0.5 for c in cands)
        cover_scene = n_hz >= 4 and n_hz >= 0.5 * sum(s["score"] >= 0.7 for s in spines)
        spines = self.verify_spines(img, spines, cover_scene)

        def keep(c):
            if c["tier"] == "C":
                return c["horizontal"] >= 0.5                      # 빈자리 후보는 가로 글자가 있어야 함
            if c["n_text"] > 0:
                return True
            # 글자 없는 상자는 바닥·배경으로 보고 제거. 단 표지 위주 사진의 확실한 표지 모양 후보는 사진 위주 표지일 수 있어 남김
            return (cover_scene and c["tier"] == "A" and _cover_shaped(c)
                    and not any(_inter(c["bbox"], s["bbox"]) > 0.2 * min(_area(c["bbox"]), _area(s["bbox"])) for s in spines))

        cands = [c for c in cands if keep(c)]
        covers, vbooks = [], []
        for c in cands:
            w, h = _wh(c)
            # 가로 글줄(또는 글자 없는 표지) → 표지 / (표지 위주 사진에서) 세로 글줄인데 상자가 가로로 넓음 → 옆으로 눕힌 표지 / 그 밖 → 책등(묶음)
            is_cover = c["horizontal"] >= 0.5 or c["n_text"] == 0 or (cover_scene and w >= 1.1 * h)
            if not cover_scene and w > h and (is_cover or h < 1.1 * w):
                continue                                   # 책장 사진의 가로로 넓은 상자 → 명함 상자·게임기 상자 같은 소품
            if not cover_scene and not is_cover and h < 2.5 * w:
                continue                                   # 책장 사진의 세로 글자 상자도 책등처럼 길쭉해야 함
            (covers if is_cover else vbooks).append(_box_book(c["bbox"], "cover" if is_cover else "spine", c["support"]))
        books = merge_books(spines, covers, vbooks, coco, self.confirm_spines, cover_scene)
        self.last = {"spines": spines, "fused": self.cover(img), "candidates": cands, "covers": covers, "vbooks": vbooks,
                     "cover_scene": cover_scene, "raw_spines": raw_spines}   # 단계별 중간 결과
        return books, mode


# ---------------------------------------------------------------- 책 크롭
def crop_upright(img, rect, pad=2, return_tf=False):
    """회전 박스를 '세로로 긴 + 위쪽이 위' 크롭으로 정면 보정. return_tf면 크롭 좌표 → 원본 좌표 변환 정보도 반환"""
    (cx, cy), (w, h), ang = rect
    while ang > 90: ang -= 180
    while ang <= -90: ang += 180
    if abs(ang) > 45:
        w, h = h, w
        ang -= 90 if ang > 0 else -90
    if w > h:   # 긴 변이 가로(눕힌 책·90° 돌아간 표지): 90° 더 돌려 세움 (크기만 바꾸면 가운데 좁은 띠만 잘림)
        w, h = h, w
        ang += 90 if ang <= 0 else -90
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    x0, y0 = int(cx - w / 2) + pad, int(cy - h / 2) + pad
    x1, y1 = int(cx + w / 2) - pad, int(cy + h / 2) - pad
    ox, oy = max(0, x0), max(0, y0)
    crop = rot[oy:y1, ox:x1]
    return (crop, (cv2.invertAffineTransform(M), ox, oy)) if return_tf else crop


def crop_book(img, book):
    """책등은 세로로 세워서, 표지는 사진 그대로(축 정렬) 자른다. 통합 검출기의 회전 상자(obb)는 표지도 회전을 펴서 자른다"""
    if book["kind"] == "spine" or book.get("obb"):
        return crop_upright(img, book["rect"], return_tf=True)
    x1, y1, x2, y2 = [int(round(v)) for v in book["bbox"]]
    x1, y1 = max(0, x1), max(0, y1)
    return img[y1:y2, x1:x2], (np.float32([[1, 0, 0], [0, 1, 0]]), x1, y1)


def to_image_coords(pts, tf):
    Minv, ox, oy = tf
    p = np.asarray(pts, np.float32).reshape(-1, 2) + np.float32([ox, oy])
    return cv2.transform(p[None], Minv)[0]


# ---------------------------------------------------------------- 판독 입력 (방향별)
def _binarize(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([b[0], b[-1], b[:, 0], b[:, -1]])
    return 255 - b if border.mean() > 127 else b


def relayout_vertical(crop, text_frac=0.88):
    """세로로 쌓인 한글을 글자 단위로 잘라 가로 한 줄로 재배열"""
    h, w = crop.shape[:2]
    if h < 8 or w < 4:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    b = _binarize(gray)
    b[:, : max(1, w // 12)] = 0; b[:, -max(1, w // 12):] = 0
    b = b[: int(h * text_frac)]
    on = (b > 0).sum(1) > max(1, 0.03 * w)
    segs, start = [], None
    for y, v in enumerate(on):
        if v and start is None: start = y
        elif not v and start is not None: segs.append((start, y)); start = None
    if start is not None: segs.append((start, len(on)))
    merged = []
    for a, c in segs:
        if merged and a - merged[-1][1] < max(2, w * 0.06) and (c - merged[-1][0]) < w * 0.95:
            merged[-1] = (merged[-1][0], c)
        else:
            merged.append((a, c))
    merged = [(a, c) for a, c in merged if c - a > w * 0.12]
    sq = [c - a for a, c in merged if 0.45 * w < c - a < 1.15 * w]
    pitch = np.median(sq) if sq else 0.7 * w
    split = []
    for a, c in merged:
        n = int(round((c - a) / pitch))
        if n >= 2 and (c - a) > 1.5 * pitch:
            st = (c - a) / n
            split += [(int(a + k * st), int(a + (k + 1) * st)) for k in range(n)]
        else:
            split.append((a, c))
    if not split:
        return None
    tgt = int(np.median([c - a for a, c in split])) + 8
    tiles = []
    for a, c in split:
        tile = gray[max(0, a - 3):c + 3]
        tile = cv2.resize(tile, (max(1, int(tile.shape[1] * tgt / tile.shape[0])), tgt))
        tiles += [tile, np.full((tgt, max(4, tgt // 6)), int(np.median(gray)), np.uint8)]
    return cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_GRAY2BGR)


def views(crop):
    """(기존 방식) 책등 크롭 전체의 판독 입력 3방향"""
    out = {"rot_cw": cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), "rot_ccw": cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    rl = relayout_vertical(crop)
    if rl is not None:
        out["relayout"] = rl
    return [(k, out[k]) for k in ORDER if k in out]


def box_views(box):
    """글자 상자 하나의 판독 입력.
    세로로 긴 상자: 세로쓰기 한글(재배열) / 눕힌 글자(좌·우 회전) 3방향
    가로로 긴 상자: 그대로 / 거꾸로 놓인 표지 대비 180도"""
    h, w = box.shape[:2]
    if h > 1.2 * w:
        out = [("rot_ccw", cv2.rotate(box, cv2.ROTATE_90_COUNTERCLOCKWISE)), ("rot_cw", cv2.rotate(box, cv2.ROTATE_90_CLOCKWISE))]
        rl = relayout_vertical(box, text_frac=1.0)
        return ([("relayout", rl)] if rl is not None else []) + out
    return [("as_is", box), ("rot180", cv2.rotate(box, cv2.ROTATE_180))]


# ---------------------------------------------------------------- 글자 인식 엔진 (인식 전용 모델만 사용)
class EasyOCREngine:
    """B: EasyOCR 한국어 (korean_g2, CRNN+CTC). 웹 데모는 INT8 ONNX, 여기는 PyTorch FP32"""
    name = "B · EasyOCR korean_g2"
    models = ["EasyOCR korean_g2 (CRNN 인식)"]

    def __init__(self, gpu=False):
        import easyocr
        self.r = easyocr.Reader(["ko", "en"], gpu=gpu, verbose=False, detector=False)   # 인식(recognize)만 쓰므로 CRAFT 검출 모델은 불러오지 않음

    def rec(self, imgs):
        out = []
        for img in imgs:
            res = self.r.recognize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), detail=1)
            out.append((" ".join(t for _, t, _ in res), float(np.mean([c for _, _, c in res]))) if res else ("", 0.0))
        return out

    def __call__(self, img):
        return self.rec([img])[0][0]


class PaddleRecEngine:
    """C/D: PaddleOCR 인식 모델만 사용 (TextRecognition 모듈. PaddleOCR-VL·PP-ChatOCR 등 LLM 기능은 쓰지 않음)"""

    def __init__(self, model_name, label):
        from paddleocr import TextRecognition
        self.m = TextRecognition(model_name=model_name)
        self.name, self.models = label, [model_name]

    def rec(self, imgs):
        ok = [i for i, im in enumerate(imgs) if im.shape[0] >= 8 and im.shape[1] >= 8]
        out = [("", 0.0)] * len(imgs)
        if ok:
            for i, r in zip(ok, self.m.predict(input=[imgs[i] for i in ok], batch_size=min(16, len(ok)))):
                out[i] = (r["rec_text"], float(r["rec_score"]))
        return out

    def __call__(self, img):
        return self.rec([img])[0][0]


class TextBoxDetector:
    """책 크롭 안의 글자 영역 검출 (PaddleOCR TextDetection, PP-OCRv5 DBNet 계열). 글자 위치만 찾고 내용은 읽지 않는다."""

    def __init__(self, model_name="PP-OCRv5_mobile_det"):
        from paddleocr import TextDetection
        self.m = TextDetection(model_name=model_name)
        self.models = [model_name]

    def __call__(self, crop):
        if crop.shape[0] < 16 or crop.shape[1] < 16:
            return []
        polys = [np.asarray(p, np.float32) for p in next(iter(self.m.predict(crop)))["dt_polys"]]
        if polys:
            return polys
        # 가늘고 긴 책등은 검출 입력 크기로 줄면서 글자가 뭉개짐 → 정사각형 여백을 붙여 한 번 더 (좌표는 원래 크롭 기준으로 되돌림)
        h, w = crop.shape[:2]; side = max(h, w); ox, oy = (side - w) // 2, (side - h) // 2
        sq = np.full((side, side, 3), int(np.median(crop)), np.uint8); sq[oy:oy + h, ox:ox + w] = crop
        polys = [np.asarray(p, np.float32) - np.float32([ox, oy]) for p in next(iter(self.m.predict(sq)))["dt_polys"]]
        return [np.clip(p, 0, [w - 1, h - 1]).astype(np.float32) for p in polys]


def spiral_score(crop):
    """스프링 제본 무늬 점수: 책등을 따라 같은 간격(5~45px)으로 반복되는 가로 줄무늬의 세기 (주파수 분석).
    스프링 노트는 0.2 이상, 일반 책등은 보통 0.02~0.1"""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    if h < 64 or w < 8:
        return 0.0
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    prof = gy[:, w // 6: w - w // 6].mean(1)
    prof = prof - cv2.blur(prof.reshape(-1, 1), (1, 31)).ravel()
    spec = np.abs(np.fft.rfft(prof * np.hanning(len(prof)))) ** 2
    f = np.fft.rfftfreq(len(prof))
    band = (f > 1 / 45) & (f < 1 / 5)
    return float(spec[band].max() / (spec[1:].sum() + 1e-6))


def _poly_inter(a, b):
    """두 볼록 다각형(회전 상자)의 교집합 면적"""
    area, _ = cv2.intersectConvexConvex(np.float32(a), np.float32(b))
    return float(area)


def salience(poly):
    """글자 상자의 두드러짐 = 글자 크기(짧은 변) × √길이(긴 변). 제목은 보통 가장 크고 길게 쓰인다"""
    (w, h) = cv2.minAreaRect(np.asarray(poly, np.float32))[1]
    return min(w, h) * max(w, h) ** 0.5


# ---------------------------------------------------------------- Step 2: 책 한 권 판독
def _decide(text, matcher, verify, context=""):
    m = matcher.match(text, context=context)
    s = m[0]["score"] if m else 0
    if verify and m and m[0].get("verified") is False:
        s = min(s, ACCEPT - 0.1)
    return s, m


def read_spine(crop, ocr, matcher, early=True, verify=True):
    """(기존 방식) 책등 크롭 전체를 방향별로 통째 인식"""
    best, calls = None, 0
    for name, im in views(crop):
        if early and best and best["score"] >= EARLY:
            break
        calls += 1
        text = ocr(im)
        s, m = _decide(text, matcher, verify)
        if best is None or s > best["score"]:
            best = {"view": name, "raw": text, "score": s, "top": m}
    top = best["top"][0] if best["top"] else None
    title = top["title"] if top and best["score"] >= ACCEPT else None
    return {"title": title, "raw": best["raw"], "score": best["score"], "view": best["view"], "calls": calls,
            "no_text": (not title) and is_garbage(best["raw"]), "top1": top["title"] if top else "", "text_boxes": []}


class TitleReader:
    """Step 2 판독기 = 글자 인식 엔진 + 판독 방식
    mode="salient": 글자 영역 검출 → 두드러진 상자 상위 topk → 상자별 인식(방향은 인식 신뢰도로 선택)
                    → 상자 조합(순서 포함)을 카탈로그와 매칭해 최고점 채택  ← 이번 구조
    mode="whole"  : 책 크롭 전체를 통째로 인식 (예전 방식, 비교 기준)"""

    def __init__(self, recognizer, textdet=None, mode="salient", topk=3, label=None):
        self.ocr, self.td, self.mode, self.topk = recognizer, textdet, mode, topk
        self.name = label or recognizer.name
        self.models = list(recognizer.models) + (textdet.models if (textdet and mode == "salient") else [])

    def __call__(self, crop, kind, matcher, verify=True):
        if self.mode == "whole":
            if kind == "spine":
                return read_spine(crop, self.ocr, matcher, verify=verify)
            return self._read_whole_cover(crop, matcher, verify)
        return self._read_salient(crop, kind, matcher, verify)

    def _read_whole_cover(self, crop, matcher, verify):
        res = self.ocr.rec([crop, cv2.rotate(crop, cv2.ROTATE_180)])
        best = max(((_decide(t, matcher, verify), t, v) for (t, _), v in zip(res, ["as_is", "rot180"])), key=lambda x: x[0][0])
        (s, m), raw, view = best
        top = m[0] if m else None
        title = top["title"] if top and s >= ACCEPT else None
        return {"title": title, "raw": raw, "score": s, "view": view, "calls": 2,
                "no_text": (not title) and is_garbage(raw), "top1": top["title"] if top else "", "text_boxes": []}

    def _read_salient(self, crop, kind, matcher, verify):
        polys = [p for p in self.td(crop) if min(cv2.minAreaRect(p)[1]) >= 6]
        if not polys:   # 글자 상자를 못 찾으면 예전 방식으로 대체
            r = read_spine(crop, self.ocr, matcher, verify=verify) if kind == "spine" else self._read_whole_cover(crop, matcher, verify)
            return {**r, "view": "whole(대체)", "calls": r["calls"] + 1}
        boxes = sorted(({"poly": p, "sal": salience(p)} for p in polys), key=lambda b: -b["sal"])
        top = boxes[: self.topk]
        H, W = crop.shape[:2]
        jobs = []
        for bi, b in enumerate(top):
            x1, y1, x2, y2 = [int(v) for v in _bbox(b["poly"])]
            sub = crop[max(0, y1 - 3):min(H, y2 + 3), max(0, x1 - 3):min(W, x2 + 3)]
            jobs += [(bi, vn, im) for vn, im in box_views(sub)]
        res = self.ocr.rec([im for _, _, im in jobs])
        for (bi, vn, _), (t, s) in zip(jobs, res):
            if t.strip() and s > top[bi].get("rec_score", -1):
                top[bi].update(text=t, rec_score=s, view=vn)
        read = [bi for bi, b in enumerate(top) if b.get("text")]
        best = (0, [], "", [])
        for n in range(1, len(read) + 1):           # 상자 조합: 제목이 여러 줄·여러 상자로 나뉜 경우 대비
            for combo in itertools.permutations(read, n):
                text = " ".join(top[i]["text"] for i in combo)
                s, m = _decide(text, matcher, verify)
                if s > best[0]:
                    best = (s, m, text, list(combo))
        s, m, raw, used = best
        if not used and read:
            raw, used = top[read[0]]["text"], [read[0]]
        t1 = m[0] if m else None
        title = t1["title"] if t1 and s >= ACCEPT else None
        text_boxes = [{"poly": b["poly"].tolist(), "sal": round(b["sal"], 1), "text": b.get("text", ""),
                       "selected": i < self.topk, "used": i in used} for i, b in enumerate(boxes)]
        return {"title": title, "raw": raw, "score": s, "view": "강조 " + "+".join(str(i + 1) for i in used) if used else "-",
                "calls": 1 + len(jobs), "no_text": (not title) and is_garbage(raw), "top1": t1["title"] if t1 else "",
                "text_boxes": text_boxes}


_LATIN = r"[A-Za-z][A-Za-z0-9'&.\-]*"


def script_of(text):
    """글자 구성: 'en'(영어만) / 'ko'(한글만) / 'mix'(섞임) / ''(글자 없음)"""
    ko = sum("가" <= c <= "힣" for c in text)
    en = sum(c.isascii() and c.isalpha() for c in text)
    return "mix" if ko and en else "ko" if ko else "en" if en else ""


def merge_mixed(ko_text, en_text):
    """한영 혼합: 한국어 모델 결과의 영어 단어 자리를 영어 모델이 읽은 영어 단어로 순서대로 바꿔 끼운다"""
    en_words = re.findall(_LATIN, en_text)
    out, i = [], 0
    for tok in re.split(f"({_LATIN})", ko_text):
        if tok and re.fullmatch(_LATIN, tok):
            out.append(en_words[i] if i < len(en_words) else tok); i += 1
        else:
            out.append(tok)
    return "".join(out)


def reading_order(boxes):
    """제목 순서: 위에 있는 글자가 앞, 같은 줄이면 왼쪽 → 오른쪽 (세로 책등·가로 표지 공통)"""
    if not boxes:
        return boxes
    hs = [_bbox(b["poly"])[3] - _bbox(b["poly"])[1] for b in boxes]
    band = 0.5 * float(np.median(hs))
    rows = []
    for b in sorted(boxes, key=lambda b: (_bbox(b["poly"])[1] + _bbox(b["poly"])[3]) / 2):
        cy = (_bbox(b["poly"])[1] + _bbox(b["poly"])[3]) / 2
        if rows and abs(cy - rows[-1][0]) < band:
            rows[-1][1].append(b)
        else:
            rows.append([cy, [b]])
    return [b for _, r in rows for b in sorted(r, key=lambda b: _bbox(b["poly"])[0])]


def _clean(text):
    """앞뒤의 잡음 기호(→ - ~ 등) 제거"""
    return re.sub(r"^[^0-9A-Za-z가-힣]+|[^0-9A-Za-z가-힣]+$", "", text.strip())


# 제목이 아닌 글자 (저자 · 출판사 · 시리즈명): catalog/non_titles.txt
# 카탈로그 확정에 실패해 'OCR로 읽은 제목'을 보여 줄 때만 사용 (웹 데모 e_pipeline.js 와 같은 규칙)
#  (1) 3글자 이상 항목은 읽은 글자에서 지움 (띄어쓰기·기호 무시)   예: "세계 문학 전집 파우스트" → "파우스트"
#  (2) 남은 글자가 항목과 자모 유사도 70 이상이면(2글자 이하 항목은 완전 일치) 제목이 아님 → Unknown   예: "좁지올" ≈ "조지 오웰"
_NON = []


def load_non_titles(root):
    from matcher import to_jamo
    from rapidfuzz.distance import Indel
    _NON.clear()
    f = Path(root) / "catalog" / "non_titles.txt"
    if not f.exists():
        return 0
    for line in f.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        n = normalize(t)
        if len(n) >= 2:
            pat = re.compile("[^0-9A-Za-z가-힣]*".join(map(re.escape, n)), re.I) if len(n) >= 3 else None
            _NON.append((n, to_jamo(n), pat))
    return len(_NON)


def strip_non_title(text):
    from matcher import to_jamo
    from rapidfuzz.distance import Indel
    t = text
    for _, _, pat in _NON:
        if pat is not None:
            t = pat.sub(" ", t)
    t = _clean(re.sub(r"\s+", " ", t))
    q = normalize(t)
    if len(q) < 2:
        return ""
    qj = to_jamo(q)
    for n, j, _ in _NON:
        if (q == n) if len(n) <= 2 else (Indel.normalized_similarity(qj, j) * 100 >= 70):
            return ""
    return t


ORIENTS = ("up", "down", "cw", "ccw")   # 글자 방향: 정방향 / 180도 뒤집힘 / 시계 90도 돌아감 / 반시계 90도 돌아감


def oriented_view(box, o):
    """글자 상자를 방향 o 가정으로 바로 세운 판독 입력. 세로로 쌓인 한글(정방향·뒤집힘)은 글자 단위로 가로 재배열"""
    if o == "cw":
        return cv2.rotate(box, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if o == "ccw":
        return cv2.rotate(box, cv2.ROTATE_90_CLOCKWISE)
    im = box if o == "up" else cv2.rotate(box, cv2.ROTATE_180)
    h, w = im.shape[:2]
    if h > 1.2 * w:
        rl = relayout_vertical(im, text_frac=1.0)
        return rl if rl is not None else im
    return im


def orient_points(pts, o, W, H):
    """크롭 좌표를 방향 o로 바로 세운 좌표로 변환 (읽는 순서 계산용)"""
    p = np.asarray(pts, np.float32).reshape(-1, 2); x, y = p[:, 0], p[:, 1]
    if o == "down":
        return np.stack([W - x, H - y], 1)
    if o == "cw":            # 시계 방향으로 돌아간 글자 → 반시계로 돌려 세움
        return np.stack([y, W - x], 1)
    if o == "ccw":
        return np.stack([H - y, x], 1)
    return p


class TitleRuleReader:
    """E: 제목 규칙 판독기
    (0) 글자 방향: 책 한 권의 글자 상자들을 네 방향(정방향·180도·시계 90도·반시계 90도)으로 모두 읽어 보고,
        인식 신뢰도 × 글자 크기 합이 가장 큰 방향 하나를 책 전체에 적용 (돌아가거나 뒤집힌 책등·표지 대응)
    (1) 글자 하나당 크기(= √(글자 상자 면적 ÷ 읽은 글자 수))를 구해, 가장 큰 글씨 ≥ 가장 작은 글씨 × 2 이면
        '큰 글씨'(최대 크기의 절반 이상)만 제목으로 사용 (차이가 2배 미만이면 가장 큰 글씨 그룹 = 최대 크기의 80% 이상)
        글자 수로 나누므로 두 글자씩 쌓인 '세계문학전집' 같은 시리즈명이 제목보다 크게 잡히지 않는다
    (2) 한국어 모델로 먼저 읽어 글자 구성 판정 → 영어만: 영어 전용 모델 / 한글만: 한국어 모델
    (3) 한영 혼합: 영어 모델로 영어 단어를 읽고 한국어 모델 결과와 합침
    (4) 제목 상자 순서: (0)에서 정한 방향으로 바로 세운 좌표에서 위 → 아래, 같은 줄은 왼쪽 → 오른쪽
    글자 상자를 못 찾으면 책 전체를 두 모델로 읽어 신뢰도가 높은 쪽을 사용
    카탈로그 확정이 안 되면 이미 읽은 큰 글씨 상위 3개 상자의 조합으로 한 번 더 맞춰 봄 (OCR 추가 호출 없음)
    최종 제목: 카탈로그 확정 제목 > (확정 실패 시) OCR로 읽은 제목 > Unknown"""

    def __init__(self, ko_rec, en_rec, textdet, ratio=2.0, max_read=6, max_boxes=4, min_conf=0.5, low_conf=0.75, label=None):
        self.ko, self.en, self.td = ko_rec, en_rec, textdet
        self.ratio, self.max_read, self.max_boxes, self.min_conf, self.low_conf = ratio, max_read, max_boxes, min_conf, low_conf
        self.name = label or "E · 제목 규칙"
        self.models = list(ko_rec.models) + list(en_rec.models) + list(textdet.models)

    def _read(self, views_):
        """방향별 입력들 → (글자, 신뢰도, 방향, 언어, 호출 수)  ※ 글자 상자가 없을 때 책 전체 판독용"""
        ko = self.ko.rec([im for _, im in views_])
        bi = max(range(len(views_)), key=lambda i: ko[i][1])
        ko_text, ko_score = ko[bi]
        sc, calls = script_of(ko_text), len(views_)
        if sc in ("en", "mix"):
            en = self.en.rec([im for _, im in views_]); calls += len(views_)
            ei = max(range(len(views_)), key=lambda i: en[i][1])
            if sc == "en":
                return _clean(en[ei][0]), en[ei][1], views_[ei][0], "en", calls
            return _clean(merge_mixed(ko_text, en[bi][0])), ko_score, views_[bi][0], "mix", calls
        return _clean(ko_text), ko_score, views_[bi][0], sc or "-", calls

    def _finish(self, title_text, conf, s, m, view, calls, text_boxes):
        t1 = m[0] if m else None
        confirmed = t1["title"] if t1 and s >= ACCEPT else None
        shown = strip_non_title(title_text)              # 저자·출판사·시리즈명은 제목으로 쓰지 않음
        ocr_ok = len(normalize(shown)) >= 2 and not is_garbage(shown) and conf >= self.min_conf
        final, source = (confirmed, "카탈로그") if confirmed else ((shown, "OCR") if ocr_ok else ("Unknown", "-"))
        return {"title": confirmed, "raw": title_text, "score": s, "view": view, "calls": calls,
                "no_text": not title_text.strip(), "top1": t1["title"] if t1 else "", "text_boxes": text_boxes,
                "final": final, "source": source}

    def __call__(self, crop, kind, matcher, verify=True):
        polys = [p for p in self.td(crop) if min(cv2.minAreaRect(p)[1]) >= 6]
        if not polys:   # 글자 상자를 못 찾음 → 책 전체를 네 방향으로 두 모델로 읽기
            vs = [(o, oriented_view(crop, o)) for o in ORIENTS]
            text, conf, view, lang, calls = self._read(vs)
            s, m = _decide(text, matcher, verify) if text else (0, [])
            return self._finish(text, conf, s, m, f"책 전체 ({lang}, {view})", calls + 1, [])
        boxes = [{"poly": p, "line": float(min(cv2.minAreaRect(p)[1]))} for p in polys]
        cand = sorted(boxes, key=lambda b: -b["line"])[: self.max_read]          # 글줄이 굵은 상자부터 읽기
        H, W = crop.shape[:2]
        subs = []
        for b in cand:
            x1, y1, x2, y2 = [int(v) for v in _bbox(b["poly"])]
            pad = max(3, int(0.12 * b["line"]))                                        # 고해상도 크롭에서도 같은 비율의 여백
            subs.append(crop[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)])
        # (0) 책 단위 글자 방향: 상자 × 4방향을 한국어 모델로 한 번에 읽고, 신뢰도² × 읽은 글자 수 × 글줄 굵기 합이 최대인 방향
        #     글자 수를 곱해 긴 제목 한 줄이 제목과 직각으로 쓰인 작은 글자(출판사·판차) 여러 개보다 무겁게 반영되게 함
        res = self.ko.rec([oriented_view(sb, o) for sb in subs for o in ORIENTS])
        ko = {(i, o): res[i * 4 + k] for i in range(len(cand)) for k, o in enumerate(ORIENTS)}
        #     투표는 굵은 글줄(가장 굵은 글줄의 60% 이상)만: 책등의 저자 목록·출판사처럼 제목과 직각인 얇은 글줄이 방향을 뒤집지 않게
        voters = [i for i in range(len(cand)) if cand[i]["line"] >= 0.6 * cand[0]["line"]]
        orient = max(ORIENTS, key=lambda o: sum(ko[(i, o)][1] ** 2 * len(normalize(ko[(i, o)][0])) * cand[i]["line"] for i in voters))
        calls = 1 + len(res)
        # (2)(3) 영어가 섞인 상자 + 한국어 신뢰도가 낮은 상자(영어를 한글로 잘못 읽었을 수 있음)는 영어 모델로도 읽음
        en_jobs = [i for i in range(len(cand)) if script_of(ko[(i, orient)][0]) in ("en", "mix") or ko[(i, orient)][1] < self.low_conf]
        en = dict(zip(en_jobs, self.en.rec([oriented_view(subs[i], orient) for i in en_jobs]))) if en_jobs else {}
        calls += len(en_jobs)
        for i, b in enumerate(cand):
            kt, ks = ko[(i, orient)]
            sc = script_of(kt)
            et, es = en.get(i, ("", 0.0))
            if sc == "en":
                b["text"], b["conf"], b["lang"] = _clean(et), es, "en"
            elif sc == "mix":
                b["text"], b["conf"], b["lang"] = _clean(merge_mixed(kt, et)), ks, "mix"
            elif i in en and script_of(et) == "en" and es >= ks + 0.1 and len(normalize(et)) >= 2:
                b["text"], b["conf"], b["lang"] = _clean(et), es, "en"           # 한국어 모델이 영어를 한글로 잘못 읽은 경우
            else:
                b["text"], b["conf"], b["lang"] = _clean(kt), ks, sc or "-"
            # 카탈로그 매칭용 대안: 한국어 모델 원문 / 영어 모델 원문 / (한글은 한국어 모델 + 영어는 영어 모델) 두 순서
            ko_h = " ".join(re.findall(r"[가-힣0-9]+", kt)); en_l = " ".join(w for w in re.findall(_LATIN, et) if len(w) >= 2)
            b["alts"] = [t for t in {_clean(kt), _clean(et), f"{en_l} {ko_h}".strip(), f"{ko_h} {en_l}".strip()} if t and t != b["text"]]
            w, h = cv2.minAreaRect(b["poly"])[1]
            b["size"] = float(np.sqrt(w * h / max(1, len(normalize(b["text"]))))) if b["text"] else 0.0
            b["upright"] = orient_points(b["poly"], orient, W, H)                   # 바로 세운 좌표 (읽는 순서용)
        read = [b for b in cand if b["text"]]
        if read:
            big, small = max(b["size"] for b in read), min(b["size"] for b in read)
            cut = big / 2 if big >= self.ratio * small else 0.8 * big                 # (1) 큰 글씨만
            title = sorted([b for b in read if b["size"] >= cut], key=lambda b: -b["size"])[: self.max_boxes]
            title = reading_order([{**b, "poly": b["upright"], "_ref": b} for b in title])   # (4) 바로 세운 좌표에서 위→아래, 왼→오
            title = [t["_ref"] for t in title]
        else:
            title = []
        title_text = " ".join(b["text"] for b in title)
        context = " ".join(b["text"] for b in read)    # 같은 책에서 읽은 글자 전체 (권차 '하'가 작은 글씨로 따로 있는 경우)
        best = (0, [])                                  # 카탈로그 확정: 규칙 제목 → 제목 상자 하나씩 → 상자별 대안 → 한글만 / 영어만 뽑은 문자열
        texts = [title_text] + [b["text"] for b in title] + [a for b in title for a in b.get("alts", [])]
        if len(title) >= 2:                              # 대안을 섞은 제목 전체 (읽는 순서 유지)
            texts += [" ".join(combo) for combo in itertools.product(*[[b["text"]] + b.get("alts", [])[:2] for b in title])][:16]
        texts += [re.sub(r"[^가-힣0-9 ]", " ", t) for t in texts if script_of(t) == "mix"]
        texts += [re.sub(r"[^A-Za-z0-9 ]", " ", t) for t in texts if script_of(t) == "mix"]
        for text in dict.fromkeys(texts):
            if text.strip():
                s_, m_ = _decide(text, matcher, verify, context)
                if s_ > best[0]:
                    best = (s_, m_)
        if best[0] < ACCEPT:                            # 확정 실패 시: 이미 읽은 큰 글씨 상위 3개의 조합으로 한 번 더
            top3 = sorted(read, key=lambda b: -b["size"])[:3]
            for n in range(2, len(top3) + 1):
                for combo in itertools.permutations(top3, n):
                    s_, m_ = _decide(" ".join(b["text"] for b in combo), matcher, verify, context)
                    if s_ > best[0]:
                        best = (s_, m_)
        conf = float(np.mean([b["conf"] for b in title])) if title else 0.0
        langs = "/".join(sorted({b["lang"] for b in title})) or "-"
        tb = [{"poly": b["poly"].tolist(), "size": round(b.get("size", 0), 1), "text": b.get("text", ""),
               "selected": "text" in b, "used": any(b is t for t in title)} for b in boxes]
        names = {"up": "정방향", "down": "180°", "cw": "시계 90°", "ccw": "반시계 90°"}
        return self._finish(title_text, conf, best[0], best[1], f"{names[orient]} · 제목 {len(title)}상자 ({langs})", calls, tb)


# ---------------------------------------------------------------- 사진 한 장 처리
def detect_image(img, detector, max_side=1600):
    """Step 1만 실행 (엔진이 여러 개여도 사진당 한 번만 검출하도록 결과를 넘겨 재사용)"""
    k = min(1.0, max_side / max(img.shape[:2]))
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    t0 = time.perf_counter()
    found, mode = detector(img)
    return {"img": img, "found": found, "mode": mode, "t_det": time.perf_counter() - t0, "stages": getattr(detector, "last", None)}


def _scale_book(bk, s):
    if s == 1.0:
        return bk
    (cx, cy), (w, h), a = bk["rect"]
    return {**bk, "rect": ((cx * s, cy * s), (w * s, h * s), a), "poly": (np.asarray(bk["poly"]) * s).tolist(),
            "bbox": [v * s for v in bk["bbox"]]}


def run_image(img, detector, reader, matcher, max_side=1600, verify=True, det=None, rec_side=3000):
    """Step 1 + Step 2. det에 detect_image 결과를 넘기면 검출은 건너뛴다
    검출은 max_side(1600)로 줄인 사진에서, 글자 읽기는 원본에 가까운 해상도(최대 rec_side)에서 잘라서 한다 (작은 글씨 보존)"""
    det = det or detect_image(img, detector, max_side)
    small, found, mode, t_det = det["img"], det["found"], det["mode"], det["t_det"]
    full = img
    kf = min(1.0, rec_side / max(full.shape[:2]))
    if kf < 1:
        full = cv2.resize(full, None, fx=kf, fy=kf, interpolation=cv2.INTER_AREA)
    s = full.shape[1] / small.shape[1]
    if s < 1.05:
        full, s = small, 1.0
    img = small
    t0 = time.perf_counter() - t_det
    books = []
    for bk in found:
        crop, tf = crop_book(full, _scale_book(bk, s))
        if crop.size == 0:
            continue
        if bk.get("spiral"):   # 스프링 노트: 책등에 제목이 없으므로 OCR 없이 Unknown
            r = {"title": None, "raw": "", "score": 0, "view": "스프링 제본", "calls": 0, "no_text": True, "top1": "",
                 "text_boxes": [], "final": "Unknown", "source": "스프링 노트"}
        else:
            r = reader(crop, bk["kind"], matcher, verify=verify)
        r["text_boxes"] = [{**b, "poly": (to_image_coords(b["poly"], tf) / s).tolist()} for b in r["text_boxes"]]   # 표시용 사진 좌표
        if "final" not in r:   # 제목 규칙 판독기가 아닌 엔진: 카탈로그 확정 > Unknown
            r["final"], r["source"] = (r["title"], "카탈로그") if r["title"] else ("Unknown", "-")
        books.append({**bk, **r})
    return {"img": img, "books": books, "mode": mode, "t_det": t_det, "t_all": time.perf_counter() - t0,
            "calls": sum(b["calls"] for b in books), "stages": det["stages"]}


def score_titles(confirmed, gt_titles):
    left = list(gt_titles); right, wrong = 0, []
    for t in confirmed:
        if t in left:
            right += 1; left.remove(t)
        else:
            wrong.append(t)
    return right, wrong


# ---------------------------------------------------------------- 레포트용 그림
_FONT_PATHS = ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               "C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
_fonts = {}


def _font(size):
    from PIL import ImageFont
    if size not in _fonts:
        for p in _FONT_PATHS:
            if Path(p).exists():
                _fonts[size] = ImageFont.truetype(p, size); break
        else:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def _pil_text(bgr, items):
    """한글이 들어간 글자를 그린다. items: (x, y, text, size, rgb, 배경rgb 또는 None)"""
    from PIL import Image, ImageDraw
    im = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
    for x, y, text, size, rgb, bg in items:
        f = _font(size)
        if bg is not None:
            l, t, r, b = d.textbbox((x, y), text, font=f); d.rectangle((l - 3, t - 2, r + 3, b + 2), fill=bg)
        d.text((x, y), text, font=f, fill=rgb)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _fit(text, size, width):
    f = _font(size)
    if f.getlength(text) <= width:
        return text
    while text and f.getlength(text + "…") > width:
        text = text[:-1]
    return text + "…"


_PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240),
            (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (128, 0, 0)]


def _label_pos(p, H):
    p = np.asarray(p)
    return int(p[:, 0].mean()), int(min(H - 20, max(18, p[:, 1].min() + 14)))


def draw_step1(img, books, title=None, alpha=0.45):
    """레포트 Step 1: 검출된 책마다 반투명 색 마스크 + 외곽선 + 번호. 책등은 실선 색, 표지는 흰 테두리를 덧그림"""
    H, W = img.shape[:2]
    over, vis = img.copy(), img.copy()
    lw = max(2, W // 500)
    for i, b in enumerate(books):
        col = _PALETTE[i % len(_PALETTE)][::-1]
        cv2.fillPoly(over, [np.int32(b["poly"])], col)
    vis = cv2.addWeighted(over, alpha, vis, 1 - alpha, 0)
    for i, b in enumerate(books):
        p = np.int32(b["poly"]); col = _PALETTE[i % len(_PALETTE)][::-1]
        if b["kind"] == "cover":
            cv2.polylines(vis, [p], True, (255, 255, 255), lw + 3)
        cv2.polylines(vis, [p], True, col, lw)
    fs = max(14, W // 70)
    items = []
    for i, b in enumerate(books):
        x, y = _label_pos(b["poly"], H)
        items.append((x - fs // 2, y, str(i + 1), fs, (255, 255, 255), (0, 0, 0)))
    n_sp = sum(b["kind"] == "spine" for b in books); n_cv = len(books) - n_sp
    head = title or "Step 1 · 책 검출"
    items.append((12, 10, f"{head}  |  총 {len(books)}권 (책등 {n_sp} · 표지 {n_cv})", max(18, W // 55), (255, 255, 255), (20, 20, 20)))
    return _pil_text(vis, items)


def draw_step2(img, books, engine_label, alpha=0.35, show_text_boxes=True):
    """레포트 Step 2: 왼쪽 = 검출 + 제목 확정 여부(초록 확정/주황 미확정) + 선택된 강조 글자 상자, 오른쪽 = 번호별 제목 목록"""
    H, W = img.shape[:2]
    lw = max(2, W // 500)
    GREEN, ORANGE = (60, 170, 20), (0, 140, 255)
    over = img.copy()
    for b in books:
        cv2.fillPoly(over, [np.int32(b["poly"])], GREEN if b.get("title") else ORANGE)
    vis = cv2.addWeighted(over, alpha, img, 1 - alpha, 0)
    for b in books:
        cv2.polylines(vis, [np.int32(b["poly"])], True, GREEN if b.get("title") else ORANGE, lw)
        if show_text_boxes:
            for tb in b.get("text_boxes", []):
                if tb["used"]:
                    cv2.polylines(vis, [np.int32(tb["poly"])], True, (255, 0, 255), lw + 1)     # 제목으로 쓴 강조 글자
                elif tb["selected"]:
                    cv2.polylines(vis, [np.int32(tb["poly"])], True, (255, 255, 0), max(1, lw - 1))  # 후보였던 상자
    fs = max(14, W // 70)
    items = [(x - fs // 2, y, str(i + 1), fs, (255, 255, 255), (0, 0, 0)) for i, b in enumerate(books) for x, y in [_label_pos(b["poly"], H)]]
    vis = _pil_text(vis, items)

    PW = max(560, int(W * 0.5))
    panel = np.full((H, PW, 3), 255, np.uint8)
    n = max(1, len(books))
    head = max(20, min(34, H // 30))
    row = max(14, min(30, int((H - head * 4) / (n + 1) / 1.35)))
    ok = sum(1 for b in books if b.get("title"))
    items = [(16, 12, f"Step 2 · {engine_label}", head, (20, 20, 20), None),
             (16, 12 + int(head * 1.4), f"제목 확정 {ok} / 검출 {len(books)}권", int(head * 0.8), (90, 90, 90), None)]
    y = 12 + int(head * 2.8)
    for i, b in enumerate(books):
        kind = "표지" if b["kind"] == "cover" else "책등"
        if b.get("title"):
            txt, col = f"{i + 1:>2}. [{kind}] {b['title']}  ({b['score']:.0f}점)", (30, 130, 20)
        else:
            raw = (b.get("raw") or "").replace("\n", " ").strip() or "글자 없음"
            txt, col = f"{i + 1:>2}. [{kind}] 미확정 · 읽은 글자: {raw}", (215, 110, 0)
        items.append((16, y, _fit(txt, row, PW - 32), row, col, None))
        y += int(row * 1.35)
        if y > H - row:
            items.append((16, y, f"… 외 {len(books) - i - 1}권", row, (90, 90, 90), None)); break
    if show_text_boxes and any(b.get("text_boxes") for b in books):
        items.append((16, H - int(row * 1.6), "자홍 상자 = 제목으로 쓴 강조 글자 · 하늘 상자 = 강조 후보", row, (120, 0, 140), None))
    panel = _pil_text(panel, items)
    return np.concatenate([vis, panel], 1)


def draw_step1_process(img, stages, books):
    """레포트 Step 1 검출 과정: ① COCO 'book' 후보 전체(6회 실행 융합) ② 규칙으로 고른 표지 후보 ③ 최종 검출
    ② 색: 초록 = 확실한 후보(A), 파랑 = 크기·모양이 맞는 약한 후보(B), 주황 = 빈자리 채우기(C)"""
    H, W = img.shape[:2]
    lw = max(2, W // 500); fs = max(18, W // 45)
    p1, p2 = img.copy(), img.copy()
    for c in stages["fused"]:
        if c["support"] >= 0.03:
            x1, y1, x2, y2 = map(int, c["bbox"])
            cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 200, 255), max(1, lw - 1))
    tier_col = {"A": (60, 170, 20), "B": (220, 120, 0), "C": (0, 140, 255)}
    for c in stages["candidates"]:
        x1, y1, x2, y2 = map(int, c["bbox"])
        cv2.rectangle(p2, (x1, y1), (x2, y2), tier_col.get(c["tier"], (0, 0, 255)), lw + 1)
    n1 = sum(c["support"] >= 0.03 for c in stages["fused"])
    p1 = _pil_text(p1, [(12, 10, f"① COCO 'book' 후보 {n1}개 (크기 3종 × 원본·180° 융합)", fs, (255, 255, 255), (20, 20, 20))])
    p2 = _pil_text(p2, [(12, 10, f"② 규칙으로 고른 표지 후보 {len(stages['candidates'])}개", fs, (255, 255, 255), (20, 20, 20))])
    p3 = draw_step1(img, books, "③ 최종 검출")
    gap = np.full((H, 12, 3), 255, np.uint8)
    return np.concatenate([p1, gap, p2, gap, p3], 1)


def draw_final(img, books, engine_label):
    """최종 결과 3단: ① 책 검출(배경 마스크) ② 글자 박스(자홍 굵은 상자 = 제목으로 쓴 큰 글씨, 하늘 = 그 밖의 글자) ③ 인식된 제목 표"""
    H, W = img.shape[:2]
    lw = max(2, W // 500); fs = max(18, W // 45)
    p1 = draw_step1(img, books, "① 책 검출")
    p2 = (img * 0.55 + 255 * 0.45).astype(np.uint8)                      # 글자 상자가 잘 보이도록 배경을 옅게
    for b in books:
        cv2.polylines(p2, [np.int32(b["poly"])], True, (150, 150, 150), 1)
        for tb in b.get("text_boxes", []):
            if not tb["used"]:
                cv2.polylines(p2, [np.int32(tb["poly"])], True, (230, 180, 0), max(1, lw - 1))
        for tb in b.get("text_boxes", []):
            if tb["used"]:
                cv2.polylines(p2, [np.int32(tb["poly"])], True, (255, 0, 255), lw + 1)
    items = [(x - fs // 3, y, str(i + 1), max(14, W // 70), (255, 255, 255), (0, 0, 0)) for i, b in enumerate(books) for x, y in [_label_pos(b["poly"], H)]]
    items.append((12, 10, "② 글자 박스 (자홍 = 제목으로 쓴 큰 글씨)", fs, (255, 255, 255), (20, 20, 20)))
    p2 = _pil_text(p2, items)

    PW = max(620, int(W * 0.6))
    p3 = np.full((H, PW, 3), 255, np.uint8)
    n = max(1, len(books))
    head = max(20, min(34, H // 30))
    row = max(13, min(28, int((H - head * 4) / (n + 1) / 1.35)))
    ok = sum(b.get("final", "Unknown") != "Unknown" for b in books)
    cat = sum(b.get("source") == "카탈로그" for b in books)
    items = [(16, 12, f"③ 인식된 제목 · {engine_label}", head, (20, 20, 20), None),
             (16, 12 + int(head * 1.4), f"제목 인식 {ok} / {len(books)}권 (카탈로그 확정 {cat})", int(head * 0.8), (90, 90, 90), None)]
    y = 12 + int(head * 2.8)
    col = {"카탈로그": (30, 130, 20), "OCR": (20, 90, 200), "-": (150, 150, 150), "스프링 노트": (150, 150, 150)}
    for i, b in enumerate(books):
        kind = "표지" if b["kind"] == "cover" else "책등"
        src = b.get("source", "-")
        txt = f"{i + 1:>2}. [{kind}] {b.get('final', 'Unknown')}" + (f"  ({src})" if src != "-" else "")
        items.append((16, y, _fit(txt, row, PW - 32), row, col.get(src, (150, 150, 150)), None))
        y += int(row * 1.35)
        if y > H - row:
            items.append((16, y, f"… 외 {len(books) - i - 1}권", row, (90, 90, 90), None)); break
    items.append((16, H - int(row * 1.6), "초록 = 카탈로그 확정 · 파랑 = OCR로 읽은 제목 · 회색 = Unknown", row, (90, 90, 90), None))
    p3 = _pil_text(p3, items)
    gap = np.full((H, 12, 3), 255, np.uint8)
    return np.concatenate([p1, gap, p2, gap, p3], 1)


def draw_debug_step1(img, stages, books):
    """Step 1 디버그 4단: ① 책등 모델 원본 검출 ② COCO 'book' 후보 전체(지지도·가로세로비) ③ 규칙 통과 후보(초록 표지/파랑 책등) ④ 최종
    ② 실선 = 표지 모양 비율(≤2.4) 통과, 점선 느낌의 얇은 선 = 비율 초과로 버려진 상자 (예: 눕힌 책)"""
    H, W = img.shape[:2]
    lw = max(2, W // 500); fs = max(16, W // 50)
    p1 = img.copy()
    for s in stages.get("raw_spines", []):
        cv2.polylines(p1, [np.int32(s["poly"])], True, (255, 0, 255), lw)
    p2 = img.copy(); lab = []
    for c in sorted(stages["fused"], key=lambda c: -c["support"])[:25]:
        if c["support"] < 0.02:
            continue
        x1, y1, x2, y2 = map(int, c["bbox"]); w, h = x2 - x1, y2 - y1
        ok = max(w, h) / max(1, min(w, h)) <= 2.4
        cv2.rectangle(p2, (x1, y1), (x2, y2), (0, 165, 255) if ok else (0, 0, 220), lw if ok else 1)
        lab.append((x1 + 4, y1 + 4, f"{c['support']:.2f}·{max(w, h) / max(1, min(w, h)):.1f}", max(12, fs // 2), (255, 255, 255), (0, 120, 255) if ok else (200, 0, 0)))
    p3 = img.copy()
    for c in stages["covers"]:
        x1, y1, x2, y2 = map(int, c["bbox"]); cv2.rectangle(p3, (x1, y1), (x2, y2), (60, 170, 20), lw + 1)
    for s in stages["spines"] + stages.get("vbooks", []):
        cv2.polylines(p3, [np.int32(s["poly"])], True, (220, 120, 0), lw)
    n1 = len(stages.get("raw_spines", [])); n2 = sum(c["support"] >= 0.02 for c in stages["fused"])
    p1 = _pil_text(p1, [(10, 8, f"① 책등 모델 원본 {n1}개", fs, (255, 255, 255), (20, 20, 20))])
    p2 = _pil_text(p2, lab + [(10, 8, f"② COCO book 후보 {n2}개 (지지도·가로세로비, 빨강=비율 초과로 버림)", fs, (255, 255, 255), (20, 20, 20))])
    p3 = _pil_text(p3, [(10, 8, f"③ 규칙 통과: 표지 {len(stages['covers'])} · 책등 {len(stages['spines']) + len(stages.get('vbooks', []))}", fs, (255, 255, 255), (20, 20, 20))])
    p4 = draw_step1(img, books, "④ 최종")
    gap = np.full((H, 10, 3), 255, np.uint8)
    top = np.concatenate([p1, gap, p2], 1); bot = np.concatenate([p3, gap, p4], 1)
    return np.concatenate([top, np.full((10, top.shape[1], 3), 255, np.uint8), bot], 0)


def border_strip(poly, W, H, tol=0.015, thin=0.08):
    """사진 테두리에 긴 변이 붙은 가는 상자 (키보드·모니터 끝, 사진 밖으로 잘린 물건 조각) → 책으로 세지 않음"""
    rect = cv2.minAreaRect(np.float32(poly)); w, h = rect[1]
    if min(w, h) > thin * min(W, H):
        return False
    pts = cv2.boxPoints(rect)
    for i in range(4):
        p, q = pts[i], pts[(i + 1) % 4]
        if np.hypot(*(p - q)) < 0.9 * max(w, h):
            continue
        if (abs(p[0]) < tol * W and abs(q[0]) < tol * W) or (abs(p[0] - W) < tol * W and abs(q[0] - W) < tol * W) or \
           (abs(p[1]) < tol * H and abs(q[1]) < tol * H) or (abs(p[1] - H) < tol * H and abs(q[1] - H) < tol * H):
            return True
    return False


def dedupe_books(books, iou=0.5, inside=0.6):
    """한 권을 여러 상자로 잡은 경우 정리 (점수 높은 상자부터 남김)
    (1) 종류와 관계없이 겹침(IoU)이 iou 이상이면 같은 책
    (2) 더 큰 상자(면적 1.5배 이상) 안에 inside 이상 들어간 작은 상자는 그 책의 일부(표지 속 그림·제목 띠·저자 줄)"""
    area = [cv2.contourArea(np.float32(b["poly"])) for b in books]
    order = sorted(range(len(books)), key=lambda i: -books[i]["score"])
    keep = []
    for i in order:
        pi = np.float32(books[i]["poly"]); dup = False
        for j in keep:
            I = _poly_inter(pi, np.float32(books[j]["poly"]))
            if I >= iou * (area[i] + area[j] - I):
                dup = True; break
        if not dup:
            keep.append(i)
    out = []
    for i in keep:
        pi = np.float32(books[i]["poly"])
        if any(area[j] >= 1.5 * area[i] and _poly_inter(pi, np.float32(books[j]["poly"])) >= inside * area[i] for j in keep if j != i):
            continue
        out.append(books[i])
    return out


class UnifiedBookDetector:
    """Step 1 (통합): 책등·표지 2클래스 YOLO11n-OBB 한 번으로 모든 방향의 책을 검출 (tools/train_detector.py 로 학습)
    합성 데이터로 0~360° 회전·뒤집기를 학습해, 근접 촬영에서 사진이 돌아가거나 뒤집혀도 원거리와 같은 방식으로 검출한다.
    기존 BookDetector(책등 모델 + YOLO11x 6회 + 규칙)를 대체하는 On-Device용 경량 구성.
    스프링 노트 판정(FFT)과 읽는 순서 번호 매기기만 규칙으로 남긴다."""

    def __init__(self, weights, imgsz=768, conf=0.35, iou=0.5, textdet=None, device=None, zoom=0.0, inside=0.6):
        from ultralytics import YOLO
        self.m = YOLO(str(weights), task="obb")
        self.imgsz, self.conf, self.iou, self.device, self.zoom, self.inside = imgsz, conf, iou, device, zoom, inside
        self.textdet = textdet or TextBoxDetector()
        self.last = None

    def _predict(self, img, pad=0.0):
        """pad > 0: 사진 둘레에 회색 여백을 붙여(가까이 찍은 책이 학습 때처럼 화면 안에 들어오게) 검출 후 원래 좌표로"""
        kw = dict(conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
        if self.device:
            kw["device"] = self.device
        H, W = img.shape[:2]; ox, oy = int(W * pad), int(H * pad)
        src = cv2.copyMakeBorder(img, oy, oy, ox, ox, cv2.BORDER_CONSTANT, value=(114, 114, 114)) if pad else img
        r = self.m.predict(src, **kw)[0]
        if r.obb is None:
            return []
        return [(p - np.float32([ox, oy]), float(s), int(c)) for p, s, c in
                zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy(), r.obb.cls.cpu().numpy())]

    def __call__(self, img):
        dets = self._predict(img) + (self._predict(img, self.zoom) if self.zoom else [])
        H, W = img.shape[:2]; frame = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        books = []
        for p, s, c in dets:
            a, clip = cv2.intersectConvexConvex(p.astype(np.float32), frame)   # 여백 쪽으로 나간 부분은 잘라냄
            if clip is None or a < 64:
                continue
            rect = cv2.minAreaRect(clip.reshape(-1, 2)); p = cv2.boxPoints(rect)
            if border_strip(p, W, H):
                continue
            kind = "cover" if c == 1 else "spine"
            books.append({"kind": kind, "obb": True, "rect": rect, "poly": p.tolist(), "score": s,
                          "short": float(min(rect[1])), "bbox": _bbox(p)})
        books = dedupe_books(books, self.iou, self.inside)
        for b in books:
            if b["kind"] == "spine":                       # 스프링 노트: 책등에 제목이 없으므로 제목은 Unknown
                crop = crop_upright(img, b["rect"])
                b["spiral"] = spiral_score(crop) >= 0.2 and len(self.textdet(crop)) <= 1
        books = order_books(books)
        self.last = {"raw_spines": [b for b in books if b["kind"] == "spine"], "fused": [], "candidates": [],
                     "covers": [b for b in books if b["kind"] == "cover"], "spines": [b for b in books if b["kind"] == "spine"],
                     "vbooks": [], "cover_scene": False}
        return books, f"unified {self.imgsz}"
