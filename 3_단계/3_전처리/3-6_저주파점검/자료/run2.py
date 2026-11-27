# %% 2-37 (2) AE 없이 저주파형 고장이 잡히나 + 저주파 피처 후보(크기 무관 위주)
# 후보는 물리로 먼저 정했다(라벨로 대역 · 문턱을 고르지 않음). 라벨은 설명용 AUC · 꼬리 확인에만.
import glob, os, sys
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import mannwhitneyu, spearmanr, kurtosis
from sklearn.linear_model import LogisticRegression
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.evaluate import best_f2_threshold
FIG = os.path.join(HERE, "figs"); FS = 25000
fl, grp = [], []
for g, p in (("normal", "TeamProject_train_변경/normal"), ("abnormal", "TeamProject_train_변경/abnormal"), ("val", "TeamProject_val/val")):
    s = sorted(glob.glob(os.path.join(ROOT, "2_data", p, "*.csv"))); fl += s; grp += [g] * len(s)
names = np.array([os.path.basename(f)[:-4] for f in fl]); grp = np.array(grp)
X = np.stack([np.loadtxt(f) for f in fl]); Xc = X - X.mean(1, keepdims=True)
N, A, V, T = grp == "normal", grp == "abnormal", grp == "val", grp != "val"
sos = lambda kind, f: butter(4, f, kind, fs=FS, output="sos")
cut = slice(2500, -2500)                       # 양 끝 0.1 s 제외(양방향 필터 가장자리)
lo20 = sosfiltfilt(sos("lowpass", 20), Xc, axis=1)[:, cut]
b200 = sosfiltfilt(sos("bandpass", [20, 200]), Xc, axis=1)[:, cut]
hp1k = sosfiltfilt(sos("highpass", 1000), Xc, axis=1)[:, cut]
F = pd.DataFrame(index=names)
# 물리 후보 (모두 무차원)
F["L_struct"] = np.log10(b200.std(1) / lo20.std(1))            # 저주파 구조 응답비: 20–200 Hz ÷ 하중(0–20 Hz). 체결 이완이면 암이 하중에 비해 크게 떤다(L4)
F["L_lowshare"] = np.log10((lo20.var(1) + b200.var(1)) / Xc[:, cut].var(1))   # 0–200 Hz 몫(log). 1에 가까울수록 저주파만 있는 파일
F["L_kurt200"] = np.log10(kurtosis(b200, axis=1) + 3)          # 20–200 Hz 첨도: 저주파 '쿵'(출렁임 끝단 타격, L2)
F["L_crest200"] = np.log10(np.abs(b200).max(1) / b200.std(1))  # 20–200 Hz 파고율
F["L_crest20"] = np.log10(np.abs(lo20).max(1) / lo20.std(1))   # 하중 파고율: 하중이 한 번 크게 튀나
env1k = np.abs(hilbert(hp1k, axis=1))
F["L2_hfl(지금 H18)"] = np.log10(np.median(env1k, 1) / lo20.std(1))
auc = lambda v: mannwhitneyu(v[A], v[N]).statistic / (A.sum() * N.sum())
L = ["후보 저주파 피처 (무차원). AUC(고장>정상) · 고장 중 정상 99 % 초과 · 정상 1 % 미만 · val 이동 AUC · ρ(하중) 정상"]
load = lo20.std(1)
for c in F.columns:
    v = F[c].values; hi, lo = np.percentile(v[N], [99, 1])
    L.append(f"{c:18s} AUC {auc(v):.2f} | 고장 > 정상 99 %: {', '.join(names[A & (v > hi)]) or '-'} | 고장 < 정상 1 %: {', '.join(names[A & (v < lo)]) or '-'} | val 범위 밖(위/아래) {int((V & (v > hi)).sum())}/{int((V & (v < lo)).sum())} | val 이동 {mannwhitneyu(v[V], v[N]).statistic/(V.sum()*N.sum()):.2f} | ρ(하중) {spearmanr(v[N], load[N])[0]:+.2f}")
L.append("\n대상 파일의 정상 내 백분위 (0–100)")
pct = lambda c, f: (F[c].values[N] < F.loc[f, c]).mean() * 100
L.append("파일     " + " | ".join(f"{c[:12]:>12s}" for c in F.columns))
for f in ("abn_015", "abn_016", "abn_028", "abn_022", "abn_005", "abn_014", "abn_017", "val_023"):
    L.append(f"{f}  " + " | ".join(f"{pct(c, f):12.0f}" for c in F.columns))
L.append("상관(정상, Spearman): " + ", ".join(f"{a}~{b} {spearmanr(F[a][N], F[b][N])[0]:+.2f}" for i, a in enumerate(F.columns) for b in F.columns[i+1:] if abs(spearmanr(F[a][N], F[b][N])[0]) > 0.6))

# %% AE 없이(ET + 박자)의 결합 여유: J-축소 학습 OOF 성분 재사용
from src.final import j_tables, JShrink
y = A[T].astype(int); core = JShrink().fit(j_tables(X[T]), y); Z = core.Z_
for nm, cols in (("J-축소(ET+박자+AE)", [0, 1, 2]), ("AE 없음(ET+박자)", [0, 1])):
    c = LogisticRegression(C=1.0, class_weight="balanced").fit(Z[:, cols], y); s = c.decision_function(Z[:, cols]); t = best_f2_threshold(y, s)
    L.append(f"\n{nm}: 문턱 {t:.2f}, 결합 OOF 입력 재대입 — 여유(점수 − 문턱): " + ", ".join(f"{f} {s[list(names[T]).index(f)] - t:+.2f}" for f in ("abn_015", "abn_016", "abn_028", "abn_022", "abn_005", "abn_014")))
    L.append(f"   고장 중 문턱 아래: {', '.join(names[T][(y == 1) & (s < t)]) or '없음'} | 정상 중 문턱 위 {int(((y == 0) & (s >= t)).sum())}개")
open(os.path.join(FIG, "수치2.txt"), "w", encoding="utf-8").write("\n".join(L)); F.assign(group=grp).to_csv(os.path.join(FIG, "저주파후보.csv"))
print("done")
