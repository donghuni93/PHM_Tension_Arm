"""R 4e: 평균 -1 g 파일(N097 N101 N120 A016)의 base16 피처 중 정상에서 크게 벗어난 열. (DC 의존 피처 찾기) 출력 out/R4e.txt"""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); root = here
while not os.path.isdir(os.path.join(root, "src")): root = os.path.dirname(root)
sys.path.insert(0, root); os.chdir(root)
import numpy as np
from src.exp import load_train
from src.features2 import cached_feat2, PeakRef
X, y, names = load_train(); nor = y == 0
Tt, Lt, fr = cached_feat2("train", X)
FJ = json.load(open(os.path.join(root, "단계", "4_피처", "4-2_H피처목록", "자료", "최종피처.json"), encoding="utf-8"))
pk = PeakRef(float(fr[1] - fr[0])).fit(Lt[nor]).transform(Lt); Tt = Tt.copy(); Tt["H16_peak_h"], Tt["H17_peak_w"] = pk[:, 0], pk[:, 1]
mean_ = X.mean(1); low = [i for i in range(len(y)) if abs(mean_[i]) > 0.5]
o = open(os.path.join(here, "out", "R4e.txt"), "w", encoding="utf-8")
o.write("base16 열 = " + str(FJ["final"]) + "\n")
for i in low:
    row = []
    for c in FJ["final"]:
        v = Tt[c].values.astype(float); ref = v[nor]; sd = ref.std() + 1e-9
        row.append((c, (v[i] - np.median(ref)) / sd))
    row.sort(key=lambda kv: -abs(kv[1]))
    o.write(f"{names[i]} mean {mean_[i]:.2f}: " + ", ".join(f"{c} {z:+.1f}" for c, z in row[:5]) + "\n")
# DC 제거 전후 모양: 정상 평균 0 파일과 -1 g 파일의 DC 이외 통계
o.write("정상 평균 |mean| 분위: " + str(np.percentile(abs(mean_[nor]), [50, 90, 99]).round(3)) + "\n")
o.write("DC 큰 파일 로그 PSD 1-2 kHz 구간 중앙값(참고)\n")
o.close()
