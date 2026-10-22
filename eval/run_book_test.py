"""단권·3권 책 사진 테스트: 실제 정답 vs Claude 판독(비교용) vs 우리 모델(PC) vs 모바일(Edge)

사용법 (shelf_ocr_test 폴더에서, 가상환경 활성화 후):
  python eval/run_book_test.py                          # PC에서 우리 모델(E) 실행 → 표 출력·저장
  python eval/run_book_test.py --mobile eval/res_mobile.json   # 모바일 웹 데모 결과까지 합쳐 표 작성
     res_mobile.json 형식: {"IMG_1587.jpg": {"count": 1, "titles": ["Embedded Recipes"], "ms": 1234}, ...}
  python eval/run_book_test.py --skip-pc                # PC 결과(eval/res_books_pc.json)를 다시 쓰고 표만 다시 만듦
결과: eval/res_books_pc.json, eval/books_result.csv, eval/books_result.md

채점: 권수 = 검출 권수가 count_ok 안에 있으면 정답
      제목 = 정답 책마다, 화면에 보여 준 제목(카탈로그 확정 + OCR 제목) 중 별칭과 자모 유사도 90 이상인 것이 있으면 정답 (1:1 짝짓기)
※ Claude 판독은 비교용 정답지일 뿐이며 파이프라인 코드에는 LLM을 쓰지 않는다.
"""
import sys, json, time, argparse, os, re, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; EV = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

ap = argparse.ArgumentParser()
ap.add_argument("--mobile", help="모바일 웹 데모 결과 JSON")
ap.add_argument("--skip-pc", action="store_true")
ap.add_argument("--rules", action="store_true", help="통합 검출기가 있어도 기존 규칙 방식으로 검출")
a = ap.parse_args()

GT = json.load(open(EV / "books_gt.json", encoding="utf-8"))
BOOKS = GT["_books"]
PHOTOS = [k for k in GT if not k.startswith("_")]


def norm(s):
    return re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFC", s).lower())


def sim(x, y):
    from matcher import to_jamo
    from rapidfuzz.distance import Indel
    x, y = norm(x), norm(y)
    return 100 * Indel.normalized_similarity(to_jamo(x), to_jamo(y)) if x and y else 0.0


def score(titles, book_keys):
    """정답 책별로 가장 잘 맞는 제목을 1:1로 짝지어 맞힌 권수 (괄호 속 판차·설명은 무시)"""
    titles = [re.sub(r"\s*\(.*?\)", "", t) for t in titles]
    pairs =sorted(((max(sim(t, al) for al in BOOKS[k]["aliases"]), i, j) for i, t in enumerate(titles) for j, k in enumerate(book_keys)), reverse=True)
    ut, ub, hit = set(), set(), 0
    for s, i, j in pairs:
        if s >= 90 and i not in ut and j not in ub:
            ut.add(i); ub.add(j); hit += 1
    return hit


# ---------------------------------------------------------------- PC 실행 (우리 모델 E)
res_pc_path = EV / "res_books_pc.json"
if not a.skip_pc:
    import numpy as np, cv2
    from shelfkit import BookDetector, UnifiedBookDetector, PaddleRecEngine, TitleRuleReader, TitleMatcher, detect_image, run_image, load_catalog
    catalog = load_catalog(ROOT); matcher = TitleMatcher(catalog)
    U = ROOT / "models" / "books_obb_best.pt"
    if U.exists() and not a.rules:
        det = UnifiedBookDetector(U, imgsz=960, conf=0.35); print("검출기: 통합 YOLO11n-OBB")
    else:
        det = BookDetector(ROOT / "models" / "spine_obb_best.pt", ROOT / "models" / "yolo11x.pt", imgsz=960, conf=0.45, iou=0.5); print("검출기: 기존 규칙 방식")
    reader = TitleRuleReader(PaddleRecEngine("korean_PP-OCRv5_mobile_rec", "ko"), PaddleRecEngine("en_PP-OCRv5_mobile_rec", "en"), det.textdet)
    print(f"카탈로그 {len(catalog)}권")
    out = {}
    for name in PHOTOS:
        img = cv2.imdecode(np.fromfile(str(EV / "books" / name), np.uint8), cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        d = detect_image(img, det); r = run_image(img, det, reader, matcher, det=d)
        ms = (time.perf_counter() - t0) * 1000
        out[name] = {"count": len(r["books"]), "titles": [b["final"] for b in r["books"] if b["final"] != "Unknown"],
                     "books": [{"kind": b["kind"], "final": b["final"], "source": b["source"], "raw": b.get("raw") or "", "view": b.get("view", "")} for b in r["books"]],
                     "ms": ms, "ms_det": d["t_det"] * 1000}
        print(f"{name}: {len(r['books'])}권 · {[b['final'] for b in r['books']]} · {ms / 1000:.1f}s", flush=True)
    json.dump(out, open(res_pc_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
pc = json.load(open(res_pc_path, encoding="utf-8")) if res_pc_path.exists() else {}
mob = json.load(open(a.mobile, encoding="utf-8")) if a.mobile else {}

# ---------------------------------------------------------------- 결과 표
rows = []
tot = {k: [0, 0] for k in ["claude_cnt", "claude_ttl", "pc_cnt", "pc_ttl", "mob_cnt", "mob_ttl"]}
for name in PHOTOS:
    g = GT[name]; keys = g["books"]; n = len(keys)
    truth = " / ".join(BOOKS[k]["truth"] for k in keys)
    row = {"사진": name, "구성": g["label"], "실제 권수": g["count"], "실제 제목": truth}
    for tag, src in [("Claude", g["claude"]), ("PC", pc.get(name)), ("모바일", mob.get(name))]:
        key = {"Claude": "claude", "PC": "pc", "모바일": "mob"}[tag]
        if not src:
            row.update({f"{tag} 권수": "-", f"{tag} 제목": "-", f"{tag} 제목 정답": "-", f"{tag} 시간(초)": "-"}); continue
        hit = score(src["titles"], keys)
        cnt_ok = src["count"] in g["count_ok"]
        tot[f"{key}_cnt"][0] += cnt_ok; tot[f"{key}_cnt"][1] += 1
        tot[f"{key}_ttl"][0] += hit; tot[f"{key}_ttl"][1] += n
        row.update({f"{tag} 권수": f"{src['count']}{'' if cnt_ok else ' ✗'}", f"{tag} 제목": " / ".join(src["titles"]) or "Unknown",
                    f"{tag} 제목 정답": f"{hit}/{n}", f"{tag} 시간(초)": f"{src['ms'] / 1000:.2f}" if "ms" in src else "참고용"})
    rows.append(row)


def pct(x):
    return f"{100 * x[0] / x[1]:.0f}% ({x[0]}/{x[1]})" if x[1] else "-"


def avg_ms(d):
    v = [d[k]["ms"] for k in PHOTOS if k in d and "ms" in d[k]]
    return f"{sum(v) / len(v) / 1000:.2f}" if v else "-"


summary = [
    {"구분": "실제 (정답)", "권수 정답률": "100%", "제목 정답률": "100%", "사진당 평균 시간(초)": "-"},
    {"구분": "Claude 판독 (LLM, 비교용)", "권수 정답률": pct(tot["claude_cnt"]), "제목 정답률": pct(tot["claude_ttl"]), "사진당 평균 시간(초)": "참고용 (클라우드 LLM)"},
    {"구분": "우리 모델 E · PC (M3 Pro CPU)", "권수 정답률": pct(tot["pc_cnt"]), "제목 정답률": pct(tot["pc_ttl"]), "사진당 평균 시간(초)": avg_ms(pc)},
    {"구분": "우리 모델 · 모바일 (Edge, 웹)", "권수 정답률": pct(tot["mob_cnt"]), "제목 정답률": pct(tot["mob_ttl"]), "사진당 평균 시간(초)": avg_ms(mob)},
]

import pandas as pd
df = pd.DataFrame(rows); sm = pd.DataFrame(summary)
df.to_csv(EV / "books_result.csv", index=False, encoding="utf-8-sig")
md = "## 요약\n\n" + sm.to_markdown(index=False) + "\n\n## 사진별\n\n" + df.to_markdown(index=False) + "\n"
(EV / "books_result.md").write_text(md, encoding="utf-8")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(sm.to_string(index=False)); print(); print(df[["사진", "실제 권수", "PC 권수", "PC 제목", "PC 제목 정답", "PC 시간(초)", "모바일 제목 정답", "모바일 시간(초)"]].to_string(index=False))
print("\n저장:", EV / "books_result.md", "·", EV / "books_result.csv")
