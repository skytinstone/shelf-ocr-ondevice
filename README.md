# shelf_ocr_test

책장·진열대 사진에서 책 권수와 제목을 추출하는 파이프라인을 PC에서 비교 테스트하는 폴더입니다.
**LLM·멀티모달 모델은 사용하지 않습니다.**

## 파이프라인
1. **Step 1 책 검출**: 책등은 YOLO11n-OBB(`models/spine_obb_best.pt`), 표지는 YOLO11x COCO `book` 클래스(`models/yolo11x.pt`, 입력 크기 3종 × 원본·180° 회전 융합)로 후보를 찾고, 포함 관계·띠지·빈자리·글자 방향 규칙으로 정리해 책등과 합칩니다.
2. **Step 2 제목 인식**: 책 안에서 글자 영역을 검출(PP-OCRv5 det)하고, 가장 두드러진 글자(크기 × √길이 상위 3개)만 인식(B/C/D 엔진)해 카탈로그와 매칭·검증합니다.
   비교 기준 D0은 예전 방식(책 전체를 통째로 인식)입니다.

## 시작하기
1. 가상환경을 만들고 `pip install -r requirements.txt` 를 실행합니다.
2. 이 폴더에서 Jupyter(또는 VS Code)로 `shelf_ocr_test.ipynb` 를 열고 커널을 가상환경으로 고릅니다.
3. 2번 셀부터 순서대로 실행합니다. (1번 설치 셀은 이미 설치했다면 건너뜁니다)

## 사진 추가
노트북 **2-1번 셀(테스트 사진 업로드)** 에서 폴더를 고르고 사진을 올리면 됩니다. 정답 권수·제목도 함께 입력하면 아래 정답 파일과 카탈로그에 자동으로 저장됩니다.
또는 `images/` 아래 상황별 폴더에 사진을 직접 넣어도 자동으로 포함됩니다. 새 폴더를 만들어도 됩니다.
정확도를 보려면 `gt/ground_truth.json` 에 `"폴더/파일명": {"count": 권수, "titles": [정답 제목...]}` 를 추가하고,
사진 속 제목을 `catalog/my_titles.txt` 에도 적어 주세요. (카탈로그에 없는 제목은 글자를 읽어도 확정되지 않습니다)

## 결과
- `outputs/report/step1_detection/` : 레포트 Step 1, 책 검출 마스킹 이미지 (사진당 1장), `process/` 에 표지 사진의 검출 과정 그림
- `outputs/report/step2_titles/<엔진>/` : 레포트 Step 2, 엔진별 책 제목 이미지
- `outputs/report/step2_titles/compare/` : 엔진별 결과를 위아래로 이어 붙인 비교 이미지
- `outputs/*.csv`, `summary_by_engine.png` : 검출 튜닝 표, 사진별·책별 결과, 엔진별 요약
