# %% 2-37 저주파 전처리: ① 0–2 Hz는 무엇이고, 하중을 2–20 Hz로 바꾸면 H18 · E2가 어떻게 되나 ② abn_015를 J가 무엇으로 잡나 ③ 저주파 봉우리(암의 강체 모드) 탐색
# 라벨은 집단 구분 · 설명용 AUC에만. 문턱 · 대역을 라벨로 고르지 않는다. val은 라벨 없이.
import glob, os, sys
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, welch
from scipy.stats import mannwhitneyu, spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True); FS = 25000
fl, grp = [], []
for g, p in (("normal", "TeamProject_train_변경/normal"), ("abnormal", "TeamProject_train_변경/abnormal"), ("val", "TeamProject_val/val")):
    s = sorted(glob.glob(os.path.join(ROOT, "data", p, "*.csv"))); fl += s; grp += [g] * len(s)
names = np.array([os.path.basename(f)[:-4] for f in fl]); grp = np.array(grp)
X = np.stack([np.loadtxt(f) for f in fl]); Xc = X - X.mean(1, keepdims=True)
N, A, V, T = grp == "normal", grp == "abnormal", grp == "val", grp != "val"
y = A[T].astype(int)
auc = lambda v: mannwhitneyu(v[A], v[N]).statistic / (A.sum() * N.sum())
sos = lambda kind, f: butter(4, f, kind, fs=FS, output="sos")
L = []

# %% ① 0–2 Hz의 정체
b02 = sosfiltfilt(sos("lowpass", 2), Xc, axis=1); b220 = sosfiltfilt(sos("bandpass", [2, 20]), Xc, axis=1); b020 = sosfiltfilt(sos("lowpass", 20), Xc, axis=1)
t = np.arange(X.shape[1]) / FS
slope = np.polyfit(t, Xc.T, 1)[0]                                      # 2초 동안의 선형 추세(g/s)
trend_share = np.array([np.var(np.polyval(np.polyfit(t, b, 1), t)) / (np.var(b) + 1e-20) for b in b02])   # 0–2 Hz 성분 중 직선 추세 몫
r02, r220, r020 = b02.std(1), b220.std(1), b020.std(1)
L.append("① 0–2 Hz 성분 (중앙값 [IQR])")
for nm, v in (("0–2 Hz RMS (g)", r02), ("2–20 Hz RMS (g)", r220), ("0–2 ÷ 0–20 파워 몫", r02 ** 2 / r020 ** 2), ("0–2 Hz 중 직선 추세 몫", trend_share)):
    L.append(f"{nm:22s} 정상 {np.median(v[N]):.4f} [{np.percentile(v[N],25):.4f}, {np.percentile(v[N],75):.4f}] | 고장 {np.median(v[A]):.4f} | val {np.median(v[V]):.4f} | AUC(고장>정상) {auc(v):.2f}")
L.append(f"ρ(0–2 Hz RMS, 2–20 Hz RMS) 정상 {spearmanr(r02[N], r220[N])[0]:+.2f} — 같이 움직이면 같은 하중, 따로면 다른 원인")
L.append(f"ρ(0–2 Hz RMS, |DC|) 정상 {spearmanr(r02[N], np.abs(X.mean(1))[N])[0]:+.2f}")
# 2초 파일에서 0–2 Hz는 최대 4주기: 진동인지 추세인지를 몫으로 판단
# %% H18 변형: 분모 = 0–20 Hz(지금) vs 2–20 Hz
from src.impacts import all_impacts
imps = all_impacts(X)
env_med = np.array([np.median(np.abs(hilbert(im["hp"]))) for im in imps]); nimp = np.array([len(im["t"]) for im in imps])
H18 = {"0–20 Hz (지금)": np.log10(env_med / r020), "2–20 Hz": np.log10(env_med / r220), "분자만(포락선 중앙)": np.log10(env_med)}
L.append("\nH18 변형: AUC(고장>정상) · 하중 끌림 ρ(H18, 하중) 정상 · val 이동 AUC(val>정상) · E2(정상 99 % 분위 & 충격 ≥ 1, Train 전체 정상 기준 재대입)")
for nm, h in H18.items():
    q = np.percentile(h[N], 99); e2 = (h > q) & (nimp >= 1)
    L.append(f"{nm:18s} AUC {auc(h):.2f} | ρ(·, 0–20 Hz RMS) {spearmanr(h[N], r020[N])[0]:+.2f} · ρ(·, 2–20 Hz RMS) {spearmanr(h[N], r220[N])[0]:+.2f} | val 이동 AUC {mannwhitneyu(h[V], h[N]).statistic/(V.sum()*N.sum()):.2f} | E2 고장 {', '.join(names[A & e2])} | 정상 {(N & e2).sum()} | val {', '.join(names[V & e2]) or '없음'}")

# %% ② abn_015 (그리고 저주파형 고장)는 무엇으로 잡히나: J-축소 학습 OOF 성분
from src.final import j_tables, JShrink
Tb = j_tables(X[T]); core = JShrink().fit(Tb, y); P = core.parts(Tb)
Z = pd.DataFrame(core.Z_, columns=["ET_oof_logit", "beat_z", "AE_z"], index=names[T])
sc = core.c_.decision_function(core.Z_)
L.append(f"\n② J-축소(학습 150) 안쪽 OOF 성분과 결합 점수(문턱 {core.tc_:.2f}), 규칙(재대입)")
b20_200 = sosfiltfilt(sos("bandpass", [20, 200]), Xc, axis=1).std(1)
lowshare = (r020 ** 2 + b20_200 ** 2) / Xc.var(1)
for f in ("abn_015", "abn_028", "abn_022", "abn_017"):
    i = list(names[T]).index(f); r = P.iloc[i]
    L.append(f"{f}: ET {Z.ET_oof_logit[f]:+.2f} · 박자 z {Z.beat_z[f]:+.2f} · AE z {Z.AE_z[f]:+.2f} → 결합 {sc[i]:+.2f} ({'≥' if sc[i] >= core.tc_ else '<'} 문턱) | E2 {bool(r.E2)} · 8.3k {bool(r.rule83)} | 0–200 Hz 몫 {lowshare[names == f][0]:.2f} · 20–200 Hz RMS {b20_200[names == f][0]:.4f} g")
L.append(f"0–200 Hz 몫: 정상 중앙 {np.median(lowshare[N]):.2f}, 99 % {np.percentile(lowshare[N],99):.3f}, 최대 {lowshare[N].max():.3f} | 20–200 Hz RMS 정상 99 % {np.percentile(b20_200[N],99):.4f} g, 최대 {b20_200[N].max():.4f}")
L.append("20–200 Hz RMS가 정상 최대를 넘는 파일: " + ", ".join(f"{names[i]}({b20_200[i]:.3f})" for i in np.where(b20_200 > b20_200[N].max())[0]))

# %% ③ 저주파 봉우리(2–200 Hz): 암 · 부싱 강체 모드가 보이나, 고장에서 옮겨 가나
f, Pw = welch(Xc, fs=FS, nperseg=25000, detrend="constant")       # 1 Hz 분해능, 2초에 구간 3개(50 % 겹침)
m = (f >= 3) & (f <= 200); LPw = np.log10(Pw[:, m] + 1e-30); fm = f[m]
med = lambda k: np.median(LPw[k], 0)
from scipy.signal import find_peaks
def pk(curve, prom=0.15):
    i, pr = find_peaks(curve, prominence=prom); return ", ".join(f"{fm[j]:.0f}" for j in i)
L.append("\n③ 3–200 Hz 중앙 log PSD 봉우리(Hz, 돌출 ≥ 0.15 dec): 정상 " + pk(med(N)) + " | 고장 " + pk(med(A)) + " | val " + pk(med(V)))
pkf = fm[np.argmax(LPw[:, (fm >= 5) & (fm <= 60)], 1) + np.argmax((fm >= 5))]
L.append(f"파일별 5–60 Hz 최대 봉우리 위치(Hz): 정상 {np.median(pkf[N]):.0f} [{np.percentile(pkf[N],25):.0f}, {np.percentile(pkf[N],75):.0f}] | 고장 {np.median(pkf[A]):.0f} [{np.percentile(pkf[A],25):.0f}, {np.percentile(pkf[A],75):.0f}] | val {np.median(pkf[V]):.0f} | AUC {auc(pkf):.2f}")
for a_, b_ in ((3, 10), (10, 30), (30, 60), (60, 120), (120, 200)):
    k = (fm >= a_) & (fm < b_); sh = Pw[:, m][:, k].sum(1) / Pw[:, m].sum(1)
    L.append(f"{a_}–{b_} Hz 몫(3–200 Hz 안): 정상 {np.median(sh[N]):.3f} 고장 {np.median(sh[A]):.3f} val {np.median(sh[V]):.3f} | AUC {auc(sh):.2f}")
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L))
pd.DataFrame(dict(file=names, group=grp, r02=r02, r220=r220, r020=r020, trend02=trend_share, H18_020=H18["0–20 Hz (지금)"], H18_220=H18["2–20 Hz"], nimp=nimp, low200share=lowshare, b20_200=b20_200, lowpk=pkf)).to_csv(os.path.join(FIG, "files.csv"), index=False)
np.savez(os.path.join(FIG, "lowpsd.npz"), f=fm, LP=LPw, grp=grp, names=names)
print("done")
