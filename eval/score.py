"""엣지 모델(E) 결과를 Claude 판독 정답과 비교해 채점
권수: 채점 영역(roi) 안에 중심이 있는 검출 수 vs 정답 권수
제목: 화면에 보여 준 제목(카탈로그 확정 + 글자로 읽은 제목)을 정답 제목과 1:1로 짝지어 판정
  정답  : 자모 유사도 ≥ 90 (띄어쓰기·대소문자·기호 무시)
  부분  : 유사도 ≥ 60 이거나 한쪽이 다른 쪽에 포함 (4글자 이상)
  오답  : 짝지을 정답이 없거나 유사도 < 60
  Unknown: 제목을 보여 주지 않음
"""
import json, re, sys, unicodedata
from rapidfuzz.distance import Indel

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"; JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"; JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
norm = lambda s: re.sub(r"[^0-9a-z가-힣]", "", unicodedata.normalize("NFC", s).lower())
def jamo(s):
    o = ""
    for ch in s:
        c = ord(ch)
        if 0xAC00 <= c <= 0xD7A3:
            c -= 0xAC00; o += CHO[c // 588] + JUNG[(c % 588) // 28] + JONG[c % 28]
        else:
            o += ch
    return o
def sim(a, b):
    a, b = norm(a), norm(b)
    if not a or not b: return 0.0
    return 100 * Indel.normalized_similarity(jamo(a), jamo(b))
VOL = re.compile(r"(?:^|\s)(\d{1,2}|상|중|하)\s*(?:권)?\s*$")
def vol(t):
    m = VOL.search(unicodedata.normalize("NFC", t).strip()); return m.group(1) if m else None
def contains(a, b):
    a, b = norm(a), norm(b)
    return min(len(a), len(b)) >= 4 and (a in b or b in a)

gt = json.load(open(sys.argv[1])); res = json.load(open(sys.argv[2]))
rows, tot = [], dict(gt_count=0, det=0, gt_titles=0, exact=0, exact_cat=0, partial=0, wrong=0, unknown=0, shown=0, unc=0, cat_wrong=0)
detail = {}
for f, g in gt.items():
    if f not in res: continue
    R = res[f]; W, H = R["W"], R["H"]; roi = g.get("roi", {})
    xr, yr = roi.get("x", [0, 1]), roi.get("y", [0, 1])
    books = [b for b in R["books"] if xr[0] <= (b["bbox"][0] + b["bbox"][2]) / 2 / W <= xr[1] and yr[0] <= (b["bbox"][1] + b["bbox"][3]) / 2 / H <= yr[1]]
    shown = [b for b in books if b["final"] != "Unknown"]
    targets = [(t, True) for t in g["titles"]] + [(re.sub(r"\s*\(.*\)$", "", t), False) for t in g.get("uncertain", []) if not t.startswith("(")]
    pairs = sorted(((max(sim(b["final"], t), 100 if contains(b["final"], t) and sim(b["final"], t) >= 50 else 0), i, j) for i, b in enumerate(shown) for j, (t, _) in enumerate(targets)), reverse=True)
    used_b, used_t, match = set(), set(), {}
    for s, i, j in pairs:
        if i in used_b or j in used_t or s < 60 and not contains(shown[i]["final"], targets[j][0]): continue
        used_b.add(i); used_t.add(j); match[i] = (j, s)
    ex = exc = pa = wr = 0; lines = []
    for i, b in enumerate(shown):
        if i in match:
            j, s = match[i]; t, certain = targets[j]
            raw = sim(b["final"], t)
            vb, vt = vol(b["final"]), vol(t)
            if b["source"] == "카탈로그" and (vb != vt or raw < 90):
                verdict = "오답(카탈로그 오확정)"; wr += 1; tot["cat_wrong"] += 1
                lines.append((b["final"], b["source"], t, verdict, round(raw))); continue
            if vb and vt and vb != vt:
                verdict = "부분"; pa += certain
            elif raw >= 90:
                verdict = "정답"
                if certain: ex += 1; exc += b["source"] == "카탈로그"
            else:
                verdict = "부분"; pa += certain
            if not certain: tot["unc"] += 1
            lines.append((b["final"], b["source"], t, verdict, round(raw)))
        else:
            wr += 1; lines.append((b["final"], b["source"], "-", "오답(카탈로그 오확정)" if b["source"] == "카탈로그" else "오답", 0))
            if b["source"] == "카탈로그": tot["cat_wrong"] += 1
    unk = len(books) - len(shown)
    detail[f] = lines
    rows.append(dict(photo=f, label=g["label"], gt=g["count"], det=len(books), titles=len(g["titles"]), exact=ex, exact_cat=exc, partial=pa, wrong=wr, unknown=unk, shown=len(shown), t=R["tAll"] / 1000))
    for k, v in dict(gt_count=g["count"], det=len(books), gt_titles=len(g["titles"]), exact=ex, exact_cat=exc, partial=pa, wrong=wr, unknown=unk, shown=len(shown)).items(): tot[k] += v
json.dump(dict(rows=rows, total=tot, detail=detail), open(sys.argv[3], "w"), ensure_ascii=False, indent=1)
print(f"{'사진':<20}{'정답권수':>6}{'검출':>5}{'정답제목':>6}{'정답':>5}{'(카탈로그)':>8}{'부분':>5}{'오답':>5}{'Unknown':>8}{'초':>6}")
for r in rows:
    print(f"{r['photo']:<20}{r['gt']:>6}{r['det']:>5}{r['titles']:>6}{r['exact']:>5}{r['exact_cat']:>8}{r['partial']:>5}{r['wrong']:>5}{r['unknown']:>8}{r['t']:>6.0f}")
print(tot)
