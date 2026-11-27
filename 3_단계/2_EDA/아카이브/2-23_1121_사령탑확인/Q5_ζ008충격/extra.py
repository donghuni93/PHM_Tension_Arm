# Q5 보충: fit이 높은(깨끗한 울림) 충격만으로 '빨리 꺼지는 충격 비율'을 만들면? (run.py가 만든 impacts.csv를 읽음)
# 파일별 비율 = (조건을 만족하는 충격 수) / (그 파일의 전체 충격 수). AUC는 사후 관찰
import os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
D = pd.read_csv(os.path.join(HERE, "impacts.csv"))
y = D.groupby("file").y.first()
out = ["파일별 비율 후보 (분모 = 파일의 전체 충격 수, 충격 있는 파일 %d개: 정상 %d, 고장 %d)" % (len(y), (y == 0).sum(), (y == 1).sum()), ""]
cands = [("ζ_B >= 0.08 (fit 제한 없음)", D.zB >= .08 - 1e-9),
         ("ζ_B >= 0.08 & fit_B >= 0.6", (D.zB >= .08 - 1e-9) & (D.fitB >= .6)),
         ("ζ_B >= 0.08 & fit_B >= 0.7", (D.zB >= .08 - 1e-9) & (D.fitB >= .7)),
         ("tau_B < 0.5 ms (fit 제한 없음)", D.tauB < .5),
         ("tau_B < 0.5 ms & fit_B >= 0.6", (D.tauB < .5) & (D.fitB >= .6)),
         ("tau_B < 0.5 ms & fit_B >= 0.7", (D.tauB < .5) & (D.fitB >= .7)),
         ("fit_B < 0.6 (깨끗한 울림 아님)", D.fitB < .6),
         ("fit_B < 0.5", D.fitB < .5)]
for tag, c in cands:
    g = D.assign(c=c.astype(float)).groupby("file").c.mean(); yy = y[g.index].values
    out.append(f"{tag}: AUC {roc_auc_score(yy, g.values):.2f} | 평균 정상 {g[yy==0].mean():.1%} 고장 {g[yy==1].mean():.1%} | 충격 단위 비율 정상 {c[D.y==0].mean():.1%} 고장 {c[D.y==1].mean():.1%}")
out.append("")
out.append("ζ_B >= 0.08 충격의 fit 구간별 충격 수 (정상/고장): " + ", ".join(
    f"{lo}-{hi}: {int(((D.zB>=.08-1e-9)&(D.fitB>=lo)&(D.fitB<hi)&(D.y==0)).sum())}/{int(((D.zB>=.08-1e-9)&(D.fitB>=lo)&(D.fitB<hi)&(D.y==1)).sum())}" for lo, hi in ((0, .5), (.5, .6), (.6, .7), (.7, 1.01))))
open(os.path.join(HERE, "result_extra.txt"), "w", encoding="utf-8").write("\n".join(out)); print("\n".join(out))
