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


def load_folder(folder, pattern="*.csv"):
    """라벨이 없는 폴더(val·test)의 CSV를 이름순으로 읽는다. 하위 폴더까지 찾는다

    반환
    - X: (파일 수, 샘플 수) 신호 배열, 단위 g
    - names: 파일 이름(확장자 제외). X와 같은 순서
    """
    # 압축을 풀 때 생기는 숨김 파일(__MACOSX/._val_001.csv 등)은 신호가 아니므로 뺀다
    paths = sorted((p for p in Path(folder).rglob(pattern)
                    if not p.name.startswith(".") and "__MACOSX" not in p.parts), key=lambda p: p.stem)
    if not paths:
        raise FileNotFoundError(f"{folder} 아래에 {pattern} 파일이 없다")
    names = [p.stem for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("이름이 같은 파일이 여러 폴더에 있다. 폴더를 하나만 지정한다")
    signals = []
    for p in paths:
        x = _read_signal(p)
        if x.ndim != 1:
            raise ValueError(f"{p.name}: 열이 {x.shape[-1]}개다. 신호는 1열이어야 한다 (제출 양식 CSV가 폴더에 섞여 있지 않은지 확인)")
        signals.append(x)
    lengths = sorted({len(x) for x in signals})
    if len(lengths) != 1:
        raise ValueError(f"파일 길이가 서로 다르다: {lengths}")
    return np.stack(signals), names


def _read_signal(path):
    """신호 CSV 1개 읽기. 학습 파일과 같은 np.loadtxt를 쓰되 BOM과 머리글 한 줄은 허용한다"""
    try:
        return np.loadtxt(path, encoding="utf-8-sig")
    except ValueError:
        pass
    try:
        return np.loadtxt(path, encoding="utf-8-sig", skiprows=1)   # 첫 줄이 머리글인 경우
    except ValueError as e:
        raise ValueError(f"{path.name}: 숫자 1열로 읽을 수 없다 (쉼표로 나뉜 여러 열이거나 숫자가 아닌 값이 있음). 원인: {e}") from e


def slice_signals(X, win=12500, step=6250):
    """신호마다 길이 win의 조각을 step 간격으로 자른다 (기본 0.5초, 50% 겹침, decisions D10)

    반환
    - S: (조각 수, win) 조각 배열. 파일 0의 조각들, 파일 1의 조각들 … 순서
    - file_idx: (조각 수,) 각 조각이 X의 몇 번째 파일에서 왔는지
    """
    X = np.asarray(X)
    starts = np.arange(0, X.shape[1] - win + 1, step)   # 50,000점이면 0, 6250, …, 37500 → 7개
    S = np.stack([x[s:s + win] for x in X for s in starts])
    file_idx = np.repeat(np.arange(len(X)), len(starts))
    return S, file_idx
