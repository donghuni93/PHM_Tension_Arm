# F step7: 정상 증거 주장 점검 (새 충격 정의, 라벨은 보고용). 가벼운 계산.
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__))
IMP = pd.read_csv(os.path.join(HERE, "impacts_new.csv"), encoding="utf-8-sig")
NI = pd.read_csv(os.path.join(HERE, "nimp.csv"), encoding="utf-8-sig")
TONE = pd.read_csv(os.path.join(HERE, "tone.csv"), encoding="utf-8-sig").set_index("file")
out = open(os.path.join(HERE, "result_F7.txt"), "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")
files = NI.file.tolist(); y = NI.y.values; ntr = 150
g = {k: v for k, v in IMP.groupby("file")}
P("== A) 8.3k 순음: 돌출 >= 1.0 dex 파일 (train) : 파일 | 라벨 | 돌출 | 폭 Hz")
T = TONE.loc[files]
for f, yy in zip(files[:ntr], y[:ntr]):
    if T.loc[f, "tone_prom"] >= 1.0: P(f"  {f} y={yy} prom {T.loc[f,'tone_prom']:.2f} f {T.loc[f,'tone_f']:.0f} w {T.loc[f,'tone_w']:.0f}")
sel = T.tone_prom.values[:ntr] >= 1.0
P("  돌출>=1.0: 고장", int(sel[y[:ntr] == 1].sum()), "/30 정상", int(sel[y[:ntr] == 0].sum()), "/120 val", int((T.tone_prom.values[ntr:] >= 1.0).sum()), "/40")
sel2 = (T.tone_prom.values >= 1.0) & (T.tone_w.values >= 150)
P("  돌출>=1.0 & 폭>=150: 고장", int(sel2[:ntr][y[:ntr] == 1].sum()), " 정상", int(sel2[:ntr][y[:ntr] == 0].sum()), " val", int(sel2[ntr:].sum()))
for c in ("tone_prom", "tone_w"):
    P(f"  {c} 단변량 AUC(150): {roc_auc_score(y[:ntr], T[c].values[:ntr]):.3f}")
P("\n== B) 고주파(>=8.0k) 충격이 있는 파일의 '한 사건 몫'(높이 합 중 최대 비율) ")
for f, yy in zip(files, y):
    if f in g:
        d = g[f]; h = d[d.f >= 8000]
        if len(h):
            if files.index(f) < ntr: P(f"  {f} y={yy} n_hi={len(h)} top1_share={h.height_g.max()/h.height_g.sum():.2f} (f: {sorted(set(h.f.astype(int)))})")
            else: P(f"  {f} val n_hi={len(h)} top1_share={h.height_g.max()/h.height_g.sum():.2f}")
P("\n== C) 본체(5.5-7.9k) 충격 몫과 '충격 >= 3, 몫 >= 0.9' (EXP-034 정상 증거의 새 정의 판) ")
def body(f):
    if f not in g: return np.nan, 0
    d = g[f]; return float(((d.f >= 5500) & (d.f < 8000)).mean()), len(d)
for thr_n in (3, 4):
    a = [(f, yy) for f, yy in zip(files[:ntr], y[:ntr]) if body(f)[1] >= thr_n and body(f)[0] >= 0.9]
    P(f"  충격>={thr_n} & 본체 몫>=0.9: 정상 {sum(1 for _, v in a if v==0)}/120 고장 {sum(1 for _, v in a if v==1)}/30 val {sum(1 for f in files[ntr:] if body(f)[1]>=thr_n and body(f)[0]>=0.9)}/40")
FA = ["nor_" + k for k in ("065", "116", "037", "110", "095", "059", "055", "025", "097", "101", "120")]
P("  오경보 정상 11개의 (충격 수, 본체 몫, 고주파 충격 수):")
for f in FA:
    d = g.get(f); n = 0 if d is None else len(d)
    P(f"    {f}: n={n} body={body(f)[0]:.2f} hi={0 if d is None else int((d.f>=8000).sum())} mid(2-3.4k)={0 if d is None else int(((d.f>=2000)&(d.f<3375)).sum())}")
P("\n== D) 정상 5-95% 밖 정도: 정상 파일 중 충격 수 >=3 인 개수와 그 본체 몫 분포")
bs = [body(f)[0] for f, yy in zip(files[:ntr], y[:ntr]) if yy == 0 and body(f)[1] >= 3]
P(f"  정상 충격>=3 파일 {len(bs)}개, 본체 몫 중앙 {np.nanmedian(bs):.2f}, 몫<0.9 인 개수 {int((np.array(bs)<0.9).sum())}")
bsf = [body(f)[0] for f, yy in zip(files[:ntr], y[:ntr]) if yy == 1 and body(f)[1] >= 3]
P(f"  고장 충격>=3 파일 {len(bsf)}개, 본체 몫 중앙 {np.nanmedian(bsf):.2f}, 몫>=0.9 인 개수 {int((np.array(bsf)>=0.9).sum())}")
bsv = [body(f)[0] for f in files[ntr:] if body(f)[1] >= 3]
P(f"  val 충격>=3 파일 {len(bsv)}개, 본체 몫 중앙 {np.nanmedian(bsv):.2f}, 몫>=0.9 인 개수 {int((np.array(bsv)>=0.9).sum())}")
