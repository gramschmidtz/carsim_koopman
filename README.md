carsim_koopman/
│
├── data/                        # 데이터 폴더
│   └── (여기에 carsim_v6_*.mat 파일들을 모두 넣어주세요)
│
├── saved_models/                # 학습된 모델 저장 폴더
│
├── src/                         # 소스 코드 모듈
│   ├── __init__.py              # (빈 파일) 패키지 인식용
│   ├── dataset.py               # 데이터 로드 및 전처리
│   ├── network.py               # NNDMD (Encoder, Decoder, Dynamics) 모델
│   └── loss.py                  # 논문에 나온 손실 함수 정의
│
├── train.py                     # 학습 실행 스크립트
├── evaluate.py                  # 검증 및 결과 시각화 스크립트
└── requirements.txt             # 필요한 라이브러리 목록