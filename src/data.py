"""데이터 읽기.

train·val·test 모두 이 함수로 읽어야 순서와 형태가 같아진다.
"""
from pathlib import Path

import numpy as np

# 하위 폴더 이름 → 라벨 (정상 0, 고장 1)
LABEL_DIRS = (("normal", 0), ("abnormal", 1))


def load_dataset(data_dir):
    """data_dir/normal, data_dir/abnormal 의 CSV를 읽는다.

    반환
    - X: (파일 수, 샘플 수) 신호 배열, 단위 g
    - y: (파일 수,) 라벨, 정상 0 / 고장 1
    - names: 파일 이름(확장자 제외). X·y와 같은 순서
    """
    data_dir = Path(data_dir)
    signals, labels, names = [], [], []
    for sub, label in LABEL_DIRS:
        # sorted: 실행할 때마다 같은 순서를 보장
        for path in sorted((data_dir / sub).glob("*.csv")):
            signals.append(np.loadtxt(path))
            labels.append(label)
            names.append(path.stem)
    # 길이가 다른 파일이 있으면 np.stack이 에러를 낸다 (조용히 넘어가지 않음)
    X = np.stack(signals)
    y = np.array(labels)
    return X, y, names
