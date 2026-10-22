#!/bin/bash
# 학습 진행 상황 요약: bash tools/train_status.sh   (10초마다 자동 갱신: watch 대신 → while true; do clear; bash tools/train_status.sh; sleep 10; done)
cd "$(dirname "$0")/.."
F=outputs/train/books_obb/results.csv
if pgrep -f train_detector.py >/dev/null; then echo "상태: 학습 중"; else echo "상태: 학습 프로세스 없음 (끝났거나 멈춤)"; fi
[ -f "$F" ] || { echo "기록 파일 아직 없음"; exit 0; }
python3 - "$F" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
rows = [{k.strip(): v for k, v in r.items()} for r in rows]
r = rows[-1]; n = int(float(r["epoch"])); total = 40
t = float(r["time"]); per = t / n
print(f"진행: {n}/{total}회 · 경과 {t/60:.0f}분 · 남은 시간 약 {per*(total-n)/60:.0f}분")
print(f"검증(합성) mAP50 {float(r['metrics/mAP50(B)']):.3f} · mAP50-95 {float(r['metrics/mAP50-95(B)']):.3f} · 정밀도 {float(r['metrics/precision(B)']):.3f} · 재현율 {float(r['metrics/recall(B)']):.3f}")
best = max(rows, key=lambda x: float(x["metrics/mAP50-95(B)"]))
print(f"지금까지 최고: {int(float(best['epoch']))}회째 mAP50-95 {float(best['metrics/mAP50-95(B)']):.3f}")
PY
if [ -f models/books_obb_best.pt ]; then echo "완료: models/books_obb_best.pt 저장됨"; fi
