"""25년 데이터 읽기 — zip에서 바로 (2_data/ 폴더는 건드리지 않는다)

25년 데이터: 5 kHz × 10초 = 50,000행 1열, 단위 g (과제 PDF p.4). 우리(26) 데이터는 25 kHz × 2초
반환: X (190, 50000), y (정상 0 / 고장 1 / val −1), names (확장자 제외, train 정상 → 고장 → val 순)
캐시: 4_outputs/cache/y25_all.npz
"""
import os, io, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "1_참고자료", "팀원 공유 정보", "25 품공 소스")
CACHE = os.path.join(ROOT, "4_outputs", "cache")
FS25, N25 = 5000, 50000


def _read_zip(path, prefix):
    out = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if n.startswith(prefix) and n.endswith(".csv"):
                txt = z.read(n).decode("utf-8-sig")
                out.append((os.path.splitext(os.path.basename(n))[0], np.loadtxt(io.StringIO(txt))))
    return out


def load25():
    p = os.path.join(CACHE, "y25_all.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["X"], z["y"], list(z["names"])
    tz, vz = os.path.join(SRC, "25_train.zip"), os.path.join(SRC, "25_validation.zip")
    parts = [(_read_zip(tz, "train/normal/"), 0), (_read_zip(tz, "train/abnormal/"), 1), (_read_zip(vz, "validation/"), -1)]
    names, sigs, y = [], [], []
    for rows, lab in parts:
        for n, s in rows:
            names.append(n); sigs.append(s); y.append(lab)
    X, y = np.stack(sigs), np.array(y)               # 길이가 다르면 여기서 에러
    os.makedirs(CACHE, exist_ok=True)
    np.savez(p, X=X, y=y, names=np.array(names))
    return X, y, names


if __name__ == "__main__":
    X, y, names = load25()
    with open(os.path.join(HERE, "4_outputs", "load25_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"X {X.shape}, 정상 {(y == 0).sum()}, 고장 {(y == 1).sum()}, val {(y == -1).sum()}\n")
        f.write(f"첫/끝 이름: {names[0]} {names[119]} {names[120]} {names[149]} {names[150]} {names[-1]}\n")
        f.write(f"평균 |평균| {np.abs(X.mean(1)).mean():.4g}, RMS 중앙 {np.median(X.std(1)):.4g}\n")
