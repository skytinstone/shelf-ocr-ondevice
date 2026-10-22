"""합성 학습 데이터용 절차적 재료: 글자를 그려 넣은 가상 표지·책등, 방해물(모니터·키보드·태블릿)

실사진 재료(datasets/synth_assets)가 책등 78·표지 99장뿐이라 다양성이 부족하다. 여기서 색·그림·글꼴·글자 배치를
무작위로 만든 책을 더해, 검출기가 '책의 윤곽'을 배우고 표지 안의 그림·글줄을 따로 책으로 잡지 않게 한다.
제목 글자는 카탈로그에서 테스트 책(Embedded Recipes·회로이론·공업수학)을 뺀 목록과 일반 단어로 만든다.
"""
import json, random
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
_FD = Path("/System/Library/Fonts"); _FS = _FD / "Supplemental"
KO_FONTS = [(_FD / "AppleSDGothicNeo.ttc", i) for i in (0, 2, 4, 6, 8)] + [(_FS / "AppleMyungjo.ttf", 0), (_FS / "AppleGothic.ttf", 0)]
EN_FONTS = [(_FS / f, 0) for f in ("Arial Bold.ttf", "Arial Black.ttf", "Arial Narrow Bold.ttf", "Georgia Bold.ttf", "Georgia.ttf", "Impact.ttf",
                                   "Times New Roman Bold.ttf", "Trebuchet MS Bold.ttf", "Verdana Bold.ttf")] + \
           [(_FD / "Helvetica.ttc", 1), (_FD / "Avenir Next.ttc", 0), (_FD / "Avenir Next Condensed.ttc", 0), (_FS / "Futura.ttc", 0), (_FS / "Didot.ttc", 0)]
KO_FONTS = [f for f in KO_FONTS if f[0].exists()]; EN_FONTS = [f for f in EN_FONTS if f[0].exists()] or KO_FONTS

_TEST = {"embeddedrecipes", "회로이론", "공업수학", "engineeringcircuitanalysis", "kreyszig"}
try:
    _cat = json.load(open(ROOT / "web" / "catalog.json", encoding="utf-8"))
except Exception:
    _cat = []
TITLES = [t for t in _cat if not any(k in t.lower().replace(" ", "") for k in _TEST)]
EN_WORDS = ("THE ART OF DATA SYSTEMS DESIGN MODERN PHYSICS DEEP LEARNING HISTORY WORLD LIGHT NIGHT RIVER CODE "
            "PYTHON NETWORK THEORY PRINCIPLES GUIDE HANDBOOK SCIENCE LIFE CITY OCEAN MIND POWER ESSENTIAL ADVANCED "
            "INTRODUCTION ALGORITHMS STATISTICS MACHINE VISION SIGNAL CONTROL ECONOMICS NOVEL STORIES").split()
KO_WORDS = ("데이터 과학 역사 세계 문학 철학 경제 수학 물리 화학 생물 공학 설계 원리 이해 입문 실전 기초 고급 개론 "
            "알고리즘 통계 신호 제어 사회 심리 인간 시간 바다 도시 마음 여행 요리 영어 토익 문법 자격증 기출 문제집").split()
AUTHORS = ["김민수", "이서연", "박지훈", "최유진", "정다은", "J. Smith", "A. Kumar", "M. Chen", "R. Brown", "S. Lee 외 3인", "홍길동 옮김"]
PUBS = ["WILEY", "한빛미디어", "민음사", "길벗", "PEARSON", "O'REILLY", "창비", "문학동네", "McGraw Hill", "Springer", "시공사", "위키북스"]


def _font(rng, size, ko):
    path, idx = rng.choice(KO_FONTS if ko else EN_FONTS)
    return ImageFont.truetype(str(path), max(6, int(size)), index=idx)


def title_text(rng):
    t = rng.random()
    if t < 0.45 and TITLES:
        return rng.choice(TITLES)
    if t < 0.7:
        return " ".join(rng.choice(EN_WORDS) for _ in range(rng.randint(1, 3)))
    return " ".join(rng.choice(KO_WORDS) for _ in range(rng.randint(1, 2)))


def _is_ko(s):
    return any("가" <= c <= "힣" for c in s)


def _color(rng, dark=None):
    if dark is None:
        dark = rng.random() < 0.5
    return tuple(rng.randint(0, 90) for _ in range(3)) if dark else tuple(rng.randint(150, 255) for _ in range(3))


def _contrast(c):
    return (20, 20, 20) if sum(c) > 380 else (245, 245, 245)


def _texture(rng, w, h, pool):
    """표지 그림 영역: 사진 재료 조각 / 격자(건물 창문·도표) / 줄무늬 / 원 / 그라데이션"""
    t = rng.random()
    if t < 0.35 and pool:
        a = rng.choice(pool); ah, aw = a.shape[:2]
        s = rng.uniform(0.3, 1.0); cw, ch = max(4, int(aw * s)), max(4, int(ah * s))
        x, y = rng.randint(0, aw - cw), rng.randint(0, ah - ch)
        return cv2.resize(a[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((h, w, 3), np.uint8); img[:] = _color(rng)[::-1]
    if t < 0.6:      # 격자: 건물 창문 · 표 · 타일
        n, m = rng.randint(4, 30), rng.randint(4, 30); c1 = np.array(_color(rng)); c2 = np.array(_color(rng))
        for i in range(n):
            for j in range(m):
                x0, y0 = int(j * w / m), int(i * h / n); x1, y1 = int((j + 0.8) * w / m), int((i + 0.75) * h / n)
                col = (c1 * rng.uniform(0.7, 1.1) if rng.random() < 0.8 else c2).clip(0, 255)
                cv2.rectangle(img, (x0, y0), (x1, y1), col.tolist(), -1)
        if rng.random() < 0.5:   # 원근 (높은 건물처럼 좁아지는 모양)
            k = rng.uniform(0.1, 0.4)
            M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32([[w * k, 0], [w * (1 - k), 0], [w, h], [0, h]]))
            img = cv2.warpPerspective(img, M, (w, h), borderValue=_color(rng))
    elif t < 0.75:   # 줄무늬·선
        for _ in range(rng.randint(3, 25)):
            p = (rng.randint(-w, 2 * w), rng.randint(-h, 2 * h)); q = (rng.randint(-w, 2 * w), rng.randint(-h, 2 * h))
            cv2.line(img, p, q, _color(rng), rng.randint(1, max(2, w // 15)))
    elif t < 0.88:   # 원·도형
        for _ in range(rng.randint(2, 12)):
            cv2.circle(img, (rng.randint(0, w), rng.randint(0, h)), rng.randint(5, max(6, w // 3)), _color(rng), -1 if rng.random() < 0.6 else rng.randint(1, 6))
    else:            # 그라데이션
        g = np.linspace(0, 1, h if rng.random() < 0.5 else w, dtype=np.float32)
        c1, c2 = np.array(_color(rng), np.float32), np.array(_color(rng), np.float32)
        band = (c1[None] * (1 - g[:, None]) + c2[None] * g[:, None]).astype(np.uint8)
        img = np.broadcast_to(band[:, None, :] if len(g) == h else band[None, :, :], (h, w, 3)).copy()
    return img


def _draw_lines(d, rng, box, lines, ko, color, align):
    """box=(x0,y0,x1,y1) 안에 글줄들을 들어갈 만큼 큰 글씨로"""
    x0, y0, x1, y1 = box; bw, bh = x1 - x0, y1 - y0
    if bw < 8 or bh < 8 or not lines:
        return
    ko = ko or any(_is_ko(t) for t in lines)          # 한글이 섞이면 한글 글꼴 (영문 글꼴은 한글이 □로 나옴)
    size = bh / len(lines) * 0.8
    f = _font(rng, size, ko)
    while size > 7:
        f = _font(rng, size, ko)
        if max(d.textlength(t, font=f) for t in lines) <= bw * 0.95:
            break
        size *= 0.88
    y = y0
    for t in lines:
        tw = d.textlength(t, font=f)
        x = x0 + (bw - tw) / 2 if align == "c" else (x0 if align == "l" else x1 - tw)
        d.text((x, y), t, font=f, fill=color); y += size * 1.15


def make_cover(rng, pool, long_side=900):
    """가상 표지 (세로 비율 1.25~1.5). 그림 영역 + 큰 제목 + 부제·저자·출판사"""
    ratio = rng.uniform(1.25, 1.5); h = long_side; w = int(h / ratio)
    bg = _color(rng); img = np.zeros((h, w, 3), np.uint8); img[:] = bg[::-1]
    layout = rng.random()
    if layout < 0.4:      # 그림이 전면
        img = _texture(rng, w, h, pool)
    elif layout < 0.8:    # 그림 영역 일부
        y0 = rng.randint(int(0.15 * h), int(0.55 * h)); y1 = min(h, y0 + rng.randint(int(0.3 * h), int(0.6 * h)))
        x0 = rng.choice([0, rng.randint(0, w // 6)]); img[y0:y1, x0:w - x0] = _texture(rng, w - 2 * x0, y1 - y0, pool)
    if rng.random() < 0.4:   # 세로 띠 (책등 쪽 색 띠)
        bwid = rng.randint(w // 20, w // 6); cv2.rectangle(img, (0, 0), (bwid, h), _color(rng)[::-1], -1)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    t = title_text(rng); ko = _is_ko(t); words = t.split()
    lines = [" ".join(words[i:i + 2]) for i in range(0, len(words), 2)] if len(t) > 10 else [t]
    ty = rng.uniform(0.05, 0.55) * h; th = rng.uniform(0.1, 0.25) * h
    if rng.random() < 0.25:   # 제목 바탕 띠
        d.rectangle([0, ty - 0.02 * h, w, ty + th + 0.02 * h], fill=_color(rng))
    col = _contrast(pil.getpixel((w // 2, int(min(h - 1, ty + th / 2)))))
    _draw_lines(d, rng, (int(0.08 * w), int(ty), int(0.92 * w), int(ty + th)), lines, ko, col, rng.choice("lcc"))
    sub = " ".join(rng.choice(KO_WORDS if ko else EN_WORDS) for _ in range(rng.randint(2, 6)))
    _draw_lines(d, rng, (int(0.1 * w), int(ty + th + 0.02 * h), int(0.9 * w), int(ty + th + 0.06 * h)), [sub], ko, col, "c")
    _draw_lines(d, rng, (int(0.1 * w), int(0.82 * h), int(0.9 * w), int(0.87 * h)), [rng.choice(AUTHORS)], ko, col, rng.choice("lcr"))
    _draw_lines(d, rng, (int(0.3 * w), int(0.92 * h), int(0.7 * w), int(0.97 * h)), [rng.choice(PUBS)], False, col, "c")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def make_spine(rng, length=900):
    """가상 책등 (세로로 긴 이미지, 폭:길이 1:5~1:16). 색 띠 + 제목(옆으로 눕힌 글자 또는 세로 쌓기) + 저자·출판사"""
    L = length; W = max(24, int(L / rng.uniform(5, 16)))
    img = np.zeros((L, W, 3), np.uint8); img[:] = _color(rng)[::-1]
    for _ in range(rng.randint(0, 3)):   # 색 띠
        y0 = rng.randint(0, L); cv2.rectangle(img, (0, y0), (W, y0 + rng.randint(L // 30, L // 6)), _color(rng)[::-1], -1)
    t = title_text(rng); ko = _is_ko(t); col = _contrast(tuple(int(v) for v in img[L // 2, W // 2][::-1]))
    if ko and rng.random() < 0.45 and len(t.replace(" ", "")) <= 9:   # 한글 세로 쌓기
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
        chars = [c for c in t if c != " "]; size = min(W * 0.75, L * 0.6 / max(1, len(chars))); f = _font(rng, size, True)
        y = rng.uniform(0.05, 0.3) * L
        for c in chars:
            d.text(((W - d.textlength(c, font=f)) / 2, y), c, font=f, fill=col); y += size * 1.1
        _draw_lines(d, rng, (int(0.1 * W), int(0.85 * L), int(0.9 * W), int(0.95 * L)), [rng.choice(PUBS)[:4]], False, col, "c")
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    # 옆으로 눕힌 글자: 가로 이미지(L×W)에 쓰고 90° 돌림
    hz = np.ascontiguousarray(np.rot90(img, 1)); pil = Image.fromarray(cv2.cvtColor(hz, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    x0 = rng.uniform(0.05, 0.25) * L; x1 = x0 + rng.uniform(0.4, 0.65) * L
    _draw_lines(d, rng, (int(x0), int(0.15 * W), int(x1), int(0.85 * W)), [t], ko, col, "l")
    _draw_lines(d, rng, (int(x1 + 0.03 * L), int(0.3 * W), int(min(L * 0.85, x1 + 0.25 * L)), int(0.7 * W)), [rng.choice(AUTHORS)], ko, col, "c")
    _draw_lines(d, rng, (int(0.87 * L), int(0.25 * W), int(0.98 * L), int(0.75 * W)), [rng.choice(PUBS)], False, col, "c")
    hz = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(np.rot90(hz, rng.choice([1, 3])))


def page_block(rng, length, thick):
    """책 옆면(종이 묶음): 흰~크림색에 가는 가로줄"""
    base = rng.randint(200, 245); img = np.zeros((max(4, int(thick)), max(4, int(length)), 3), np.uint8)
    img[:] = (base - rng.randint(0, 25), base - rng.randint(0, 10), base)
    for y in range(0, img.shape[0], max(1, rng.randint(1, 3))):
        if rng.random() < 0.5:
            img[y] = (img[y].astype(np.int16) - rng.randint(5, 25)).clip(0, 255)
    return img


# ---------------------------------------------------------------- 방해물 (라벨 없음)
def _text_lines(rng, img, x0, y0, x1, y1, color, n=None):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    n = n or rng.randint(5, 30); lh = max(6, (y1 - y0) / n)
    f = _font(rng, lh * 0.7, rng.random() < 0.4)
    for i in range(n):
        s = " ".join(rng.choice(EN_WORDS + KO_WORDS).lower() for _ in range(rng.randint(2, 10)))
        d.text((x0 + rng.randint(0, 20), y0 + i * lh), s, font=f, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def screen(rng, w, h):
    """모니터·노트북 화면: 테두리 + 글줄·창"""
    img = np.zeros((h, w, 3), np.uint8); img[:] = (25, 25, 25); b = max(3, w // 60)
    inner = (rng.randint(225, 255),) * 3 if rng.random() < 0.6 else (rng.randint(20, 60),) * 3
    img[b:h - b, b:w - b] = inner
    img = _text_lines(rng, img, b + 5, b + 5, w - b, h - b, (60, 60, 60) if inner[0] > 128 else (220, 220, 220))
    for _ in range(rng.randint(0, 3)):
        x, y = rng.randint(b, w // 2), rng.randint(b, h // 2)
        cv2.rectangle(img, (x, y), (x + rng.randint(w // 6, w // 2), y + rng.randint(h // 6, h // 2)), _color(rng), 2)
    return img


def keyboard(rng, w, h):
    img = np.zeros((h, w, 3), np.uint8); img[:] = (rng.randint(15, 60),) * 3
    rows, cols = 5, rng.randint(12, 18); key = (rng.randint(30, 80),) * 3 if img[0, 0, 0] < 40 else (rng.randint(180, 240),) * 3
    for r in range(rows):
        for c in range(cols):
            x0, y0 = int(c * w / cols) + 2, int(r * h / rows) + 2
            cv2.rectangle(img, (x0, y0), (int((c + 0.9) * w / cols), int((r + 0.85) * h / rows)), key, -1)
            if rng.random() < 0.3:
                cv2.circle(img, (x0 + 4, y0 + 4), 2, (255, 160, 60), -1)
    return img


def tablet(rng, w, h):
    img = np.zeros((h, w, 3), np.uint8); img[:] = (rng.randint(5, 40),) * 3
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img + np.uint8(rng.randint(0, 20)), (0, 0), 3)
    return img
