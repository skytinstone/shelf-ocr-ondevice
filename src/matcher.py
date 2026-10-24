"""서지 매칭: OCR 오인식을 전제로 카탈로그에서 제목을 확정

설계 포인트
- 한글은 음절 단위 편집거리가 거칠다('왕'→'앙'도 1글자 오류) → 자모 단위로 분해해 비교
- 카탈로그가 수십만 권이어도 빠르도록 문자 bigram 역색인으로 후보를 먼저 추린 뒤 정밀 비교
- 신경망 임베딩을 쓰지 않음 (LLM·멀티모달 모델을 쓰지 않음 → 매칭은 비신경망으로 확정)
"""
import re
from collections import defaultdict
from rapidfuzz import fuzz

CHO = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")


def normalize(s: str) -> str:
    s = s.lower()
    return re.sub(r"[^0-9a-z가-힣]", "", s)


def to_jamo(s: str) -> str:
    out = []
    for ch in s:
        c = ord(ch)
        if 0xAC00 <= c <= 0xD7A3:
            c -= 0xAC00
            out += [CHO[c // 588], JUNG[(c % 588) // 28], JONG[c % 28]]
        else:
            out.append(ch)
    return "".join(out)


def is_garbage(text: str) -> bool:
    """OCR 결과가 의미 없는 문자열인지 판단. 스프링 제본의 코일 무늬가 'IIIIl기기기기'처럼 읽히는 경우를 걸러냄"""
    q = normalize(text)
    if len(q) < 2:
        return True
    uniq = len(set(q)) / len(q)
    longest = max(len(m.group(0)) for m in re.finditer(r"(.)\1*", q))
    hangul = sum("가" <= c <= "힣" for c in q)
    alpha = sum(c.isascii() and c.isalpha() for c in q)
    run_heavy = longest >= 4 or uniq < 0.35
    return run_heavy and (hangul + alpha) / len(q) > 0 and uniq < 0.5


def bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


class TitleMatcher:
    def __init__(self, catalog):
        self.catalog = catalog
        self.norm = [normalize(b["title"]) for b in catalog]
        self.jamo = [to_jamo(n) for n in self.norm]
        self.index = defaultdict(set)
        for i, n in enumerate(self.norm):
            for g in bigrams(n):
                self.index[g].add(i)
            for ch in set(n):  # 1글자 제목/심한 오인식 대비 unigram도 색인
                self.index["#" + ch].add(i)

    def candidates(self, q, k=50):
        votes = defaultdict(int)
        for g in bigrams(q):
            for i in self.index.get(g, ()):
                votes[i] += 2
        for ch in set(q):
            for i in self.index.get("#" + ch, ()):
                votes[i] += 1
        return [i for i, _ in sorted(votes.items(), key=lambda x: -x[1])[:k]]

    def score(self, qj, i):
        tj = self.jamo[i]
        full = fuzz.ratio(qj, tj)
        # OCR 결과에 저자·출판사가 섞여도 제목이 부분 문자열로 들어있으면 인정
        part = fuzz.partial_ratio(tj, qj) if len(qj) >= len(tj) else full
        s = max(full, 0.92 * part)
        # 짧은 제목('듄', '모순')은 우연 일치가 쉬우므로 감점
        if len(self.norm[i]) <= 2 and len(qj) > 3 * len(tj):
            s *= 0.85
        return s

    VOL = re.compile(r"(상|중|하|[1-9])$")

    def volume(self, i):
        """제목 끝의 권차 표기(상/중/하, 숫자)와 그것을 뗀 기본 제목"""
        n = self.norm[i]
        m = self.VOL.search(n)
        if m and len(n) > 2:
            return n[: m.start()], m.group(1)
        return n, None

    def verify(self, q, cands, ctx_tokens=()):
        """확정 전 검증. 실사진에서 상/하 뒤바뀜, 동명 도서 혼동이 발생해 추가함.
        (1) 같은 기본 제목의 다른 권이 카탈로그에 있으면, OCR 문자열에 이 권의 권차 표기가 있어야 확정
            제목 글자에 없으면 같은 책의 다른 글자 상자(ctx_tokens)에 권차가 '낱말 하나'로 있을 때만 인정 (예: 동그라미 속 '하')
        (2) 1위와 2위 후보가 서로 다른 책인데 점수 차가 MARGIN 미만이면 확정하지 않음
        반환: (확정 가능 여부, 사유)"""
        if not cands:
            return False, "후보 없음"
        s1, i1 = cands[0]
        base, vol = self.volume(i1)
        siblings = [j for j in range(len(self.norm)) if j != i1 and self.volume(j)[0] == base and self.volume(j)[1] != vol]
        if siblings:
            if vol is None or (vol not in q and vol not in ctx_tokens):
                return False, "권차 불명확"
            if any(self.volume(j)[1] and self.volume(j)[1] in ctx_tokens for j in siblings):
                return False, "권차 혼재"
            if any(self.volume(j)[1] and self.volume(j)[1] in q for j in siblings):
                return False, "권차 혼재"
        if len(cands) > 1:
            s2, i2 = cands[1]
            sib = self.volume(i2)[0] == base and i2 in siblings          # 같은 책의 다른 권은 위의 권차 확인으로 구분됨
            if not sib and self.norm[i2] != self.norm[i1] and self.norm[i2] not in self.norm[i1] and s1 - s2 < self.MARGIN:
                return False, "1·2위 점수 차 작음"
        # (3) 포함 관계 제목: '전자기학'이 'Ulaby의 전자기학' 안에 들어있으면 긴 쪽도 유력할 때 보류
        for s2, i2 in cands[1:]:
            if i2 != i1 and self.norm[i1] in self.norm[i2] and s2 >= 70:
                return False, "포함 관계 제목"
        # (4) 짧은 제목(4자 이하)은 우연 일치가 쉬워 더 높은 점수를 요구
        if len(self.norm[i1]) <= 4 and s1 < 80:
            return False, "짧은 제목 점수 부족"
        return True, ""

    MARGIN = 8

    def match(self, text, topk=3, context=""):
        """context: 같은 책에서 읽은 나머지 글자 (권차 확인에만 사용)"""
        q = normalize(text)
        if not q:
            return []
        qj = to_jamo(q)
        cands = self.candidates(q)
        scored = sorted(((self.score(qj, i), i) for i in cands), reverse=True)[:topk]
        ctx_tokens = {normalize(t) for t in re.split(r"[\s\W_]+", context) if t} - {""}
        if len(scored) > 1:   # 같은 책의 여러 권이 비슷한 점수면, 권차가 글자(제목 또는 같은 책의 다른 상자)에 있는 권을 1위로
            base1 = self.volume(scored[0][1])[0]
            tied = [x for x in scored if self.volume(x[1])[0] == base1 and scored[0][0] - x[0] < self.MARGIN]
            hit = [x for x in tied if self.volume(x[1])[1] and (self.volume(x[1])[1] in q or self.volume(x[1])[1] in ctx_tokens)]
            if len(hit) == 1 and hit[0] is not scored[0]:
                scored.remove(hit[0]); scored.insert(0, hit[0])
        ok, why = self.verify(q, scored, ctx_tokens)
        return [{"isbn": self.catalog[i]["isbn"], "title": self.catalog[i]["title"],
                 "score": round(s, 1), "verified": ok if k == 0 else None, "reason": why if k == 0 else None}
                for k, (s, i) in enumerate(scored)]
