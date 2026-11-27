# %% [markdown]
# V-5 효섭 피처로 본 고립 강타 (11-27)
# - 질문: 효섭 모델(RF 배깅, 11-19 최종)은 우리가 놓친 val 012 · 032 · 034 · 036을 고장으로 봤다(확률 0.46–0.68).
#         어떤 피처 때문인가? 그 피처는 train의 고립 강타 정상(nor_046 · nor_060)도 올리나? 우리 충격 검출과 무엇이 다른가?
# - 효섭 코드(1_참고자료/팀원 공유 정보/신효섭/1119_최종/7_코드/최종모델.py)를 고치지 않고 불러 쓴다
#   conda run -n qe-phm python 5_VAL_단계/V-3_고립강타_구간첨도/3-2_효섭피처진단/자료/run.py

# %% 설정
import os, sys, importlib.util
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, joblib, json
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.data import load_dataset, load_folder
from src.exp import md_table

spec = importlib.util.spec_from_file_location("hy", os.path.join(ROOT, "1_참고자료", "팀원 공유 정보", "신효섭", "1119_최종", "7_코드", "최종모델.py"))
hy = importlib.util.module_from_spec(spec); spec.loader.exec_module(hy)

cache = os.path.join(ROOT, "4_outputs", "cache", "val_eda_hy.joblib")
if os.path.exists(cache):
    Xtr_h, ACtr, r83tr, Xva_h, ACva, r83va, y, ntr, nva = joblib.load(cache)
else:
    Xtr, y, ntr = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경"))
    Xva, nva = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
    Xtr_h, ACtr, r83tr, _, _ = hy.tables(Xtr, ntr); Xva_h, ACva, r83va, _, _ = hy.tables(Xva, nva)
    joblib.dump((Xtr_h, ACtr, r83tr, Xva_h, ACva, r83va, y, ntr, nva), cache)

# AR 잔차 · 결측: 효섭 prep와 같게, 학습 전체(train 150)를 기준으로
XA = pd.concat([Xtr_h, Xva_h], ignore_index=True); ACA = np.vstack([ACtr, ACva]); yA = np.r_[y, np.zeros(len(nva), int)]
tr = np.arange(len(y)); XA = hy.prep(XA, ACA, yA, tr)
Htr, Hva = XA.iloc[tr].reset_index(drop=True), XA.iloc[len(y):].reset_index(drop=True)
cand = hy.candidates(Htr)

# %% ① 효섭 모델 재현: train 전체로 배깅 → val 확률 (제출 상세표와 같은지)
p_va = hy.bag(Htr, y, Hva, 9, cand)
sub = pd.read_csv(os.path.join(ROOT, "1_참고자료", "팀원 공유 정보", "신효섭", "1119_최종", "6_최종·제출", "제출", "submission_val_9조_상세.csv"))
same = float(np.corrcoef(p_va, sub.set_index("file").loc[nva, "prob"].values)[0, 1])

# ② train 5-fold OOF 확률 (seed 1회): 고립 강타 정상 · 고장의 확률
oof = np.zeros(len(y))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Htr, y)):
    Xi = hy.prep(Xtr_h, ACtr, y, a)
    oof[b] = hy.bag(Xi.iloc[a], y[a], Xi.iloc[b], k, cand)

# %% ③ 피처별: train AUC · 강건 z (train 정상 기준) — 후보 · 고립 강타 정상 · 잡은 val 고장
FINAL = ["A_segk", "b25_5_segk", "D_segk", "I_density", "D_nimp", "SK_200_3k", "b25_5_kurt", "A_kurt",
         "R_frac_2_5k", "R_ring_25_58", "R_ring_811_58", "R_811_over_25", "AR_resid"]
nm = y == 0; med = Htr[nm].median(); mad = 1.4826 * (Htr[nm] - med).abs().median(); mad[mad < 1e-9] = Htr[nm].std()[mad < 1e-9] + 1e-9
z = lambda D: (D - med) / mad
Ztr, Zva = z(Htr), z(Hva)
SHOW = {"val 후보(우리 놓침)": ["val_012", "val_032", "val_034", "val_036"], "train 고립강타 정상": ["nor_046", "nor_060", "nor_006", "nor_038", "nor_055", "nor_089"],
        "val 잡은 고장": ["val_039", "val_016", "val_004"], "train 고장 중앙": []}
rows = []
for c in FINAL:
    r = dict(피처=c, AUC=round(roc_auc_score(y, Htr[c]), 3))
    for n in SHOW["val 후보(우리 놓침)"] + SHOW["val 잡은 고장"]:
        r[n[4:]] = round(float(Zva.loc[nva.index(n), c]), 1)
    for n in SHOW["train 고립강타 정상"]:
        r[n.replace("nor_", "N")] = round(float(Ztr.loc[ntr.index(n), c]), 1)
    r["train고장중앙"] = round(float(Ztr[~nm][c].median()), 1)
    rows.append(r)
FZ = pd.DataFrame(rows)

# ④ 충격 검출 비교: 효섭 D_nimp(AR 백색화 8σ) vs 우리 충격 수(포락선/바탕 ≥ 4.5, 1–11 kHz)
Itr, Iva = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_imps.joblib"))
def ours(n):
    im = Itr[ntr.index(n)] if n in ntr else Iva[nva.index(n)]; return len(im["idx"])
def hyv(n, c):
    return float((Htr.loc[ntr.index(n), c]) if n in ntr else Hva.loc[nva.index(n), c])
det = pd.DataFrame([dict(file=n, 우리_충격수=ours(n), 효섭_D_nimp=round(np.exp(hyv(n, "D_nimp"))), 효섭_I_density=round(np.exp(hyv(n, "I_density")), 3),
                         A_segk=round(np.exp(hyv(n, "A_segk")), 1),
                         확률=round(float(p_va[nva.index(n)] if n in nva else oof[ntr.index(n)]), 2))
                    for n in SHOW["val 후보(우리 놓침)"] + SHOW["train 고립강타 정상"] + SHOW["val 잡은 고장"]])
# 효섭 모델이 놓친 우리 판정(013 · 017 · 024 · 025)
miss = pd.DataFrame([dict(file=n, 효섭확률=round(float(p_va[nva.index(n)]), 2)) for n in ["val_013", "val_017", "val_024", "val_025"]])
# train OOF 요약
thr = 0.435
oof_sum = dict(정상_오경보=[ntr[i] for i in np.where(nm & (oof >= thr))[0]], 고장_놓침=[ntr[i] for i in np.where(~nm & (oof < thr))[0]])

with open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8") as fh:
    fh.write(f"재현 확인: 효섭 val 확률과 상관 {same:.3f}\n\n")
    fh.write(f"train 5-fold OOF (seed 1회, 문턱 {thr}): 오경보 {oof_sum['정상_오경보']} · 놓침 {oof_sum['고장_놓침']}\n\n")
    fh.write("## ③ 최종 13개 피처: train AUC · train 정상 기준 강건 z (log 눈금 피처는 log 값 기준)\n\n" + md_table(FZ) + "\n\n")
    fh.write("## ④ 충격 검출 비교 · 확률 (val = train 전체 학습, train = OOF)\n\n" + md_table(det) + "\n\n")
    fh.write("## 효섭 모델이 놓친 우리 판정\n\n" + md_table(miss) + "\n")
json.dump(dict(재현상관=same, oof=oof_sum, val확률=dict(zip(nva, np.round(p_va, 3).tolist()))),
          open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done")

# %% ⑤ 그림: 두 검출기가 잡은 충격 비교 (val 012 · 032 · 034 · 036, 대조 nor_046 · nor_060, 기준 고장 val_039)
import src.viz  # noqa: F401
import matplotlib.pyplot as plt
from scipy import signal as ss
from src.impacts import detect_impacts, highpass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
RAW = {}
Xtr_raw, _, _ = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경")); Xva_raw, _ = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
def raw(n):
    return Xtr_raw[ntr.index(n)] if n in ntr else Xva_raw[nva.index(n)]
def hy_imps(x):
    """효섭 features와 같은 처리: 평균 제거 → 10 Hz 고역 → AR(40) 백색화 → 양끝 500점 자름 → |w| > 8σ(MAD), 간격 10 ms"""
    x = x - x.mean(); xh = ss.sosfiltfilt(hy.SOS["hp10"], x); w = hy.ar_whiten(xh)[hy.TR:-hy.TR]
    sig = np.median(np.abs(w)) / 0.6745; pk, pr = ss.find_peaks(np.abs(w), height=8 * sig, distance=250)
    xs = xh[hy.TR:-hy.TR]; fr = np.fft.rfftfreq(1024, 1 / hy.FS); fp = []
    for p0 in pk:
        seg = xs[p0:p0 + 300]
        if len(seg) < 300:
            fp.append(np.nan); continue
        Sp = np.abs(np.fft.rfft(seg * np.hanning(300), 1024)) ** 2; mk = (fr >= 1000) & (fr < 12000); fp.append(fr[mk][Sp[mk].argmax()])
    return pk + hy.TR + hy.P, pr["peak_heights"] / sig, np.array(fp), w, sig     # 원신호 표본 번호로 (자른 500 + AR 앞 40점)
FILES = ["val_012", "val_032", "val_034", "val_036", "val_039", "nor_046", "nor_060"]
fig, axs = plt.subplots(len(FILES), 1, figsize=(13, 1.5 * len(FILES)), sharex=True)
t = np.arange(50000) / 25000; rows5 = []
for a, n in zip(axs, FILES):
    x = raw(n); xh = highpass(x, fmax=11000)
    ours_ = detect_impacts(x, _hp=xh); pk, amp, fp, w, sig = hy_imps(x)
    e = ss.sosfiltfilt(ss.butter(4, 1000, "lowpass", fs=25000, output="sos"), np.abs(ss.hilbert(xh)))
    from src.impacts import _baseline
    hx_at = (e / _baseline(e))[np.clip(pk, 0, len(e) - 1)]                      # 효섭 충격 위치에서 우리 기준 높이(포락선/바탕)
    a.plot(t, xh, lw=.3, color="0.35")
    top = np.abs(xh).max()
    a.plot(ours_["t"], np.full(len(ours_["t"]), top * 1.08), "v", ms=4, color="C3", label="우리 검출(포락선/바탕 ≥ 4.5)")
    c = np.where((fp >= 2000) & (fp < 5000), "C1", "C0")
    a.scatter(pk / 25000, np.full(len(pk), -top * 1.08), marker="^", s=14, c=c, label="효섭 검출(백색화 8σ) · 주황 = 2–5 kHz 울림")
    a.set_yticks([]); a.margins(x=0)
    a.set_title(f"{n} · 우리 {len(ours_['t'])}개 · 효섭 {len(pk)}개 (2–5 kHz 울림 {int(((fp >= 2000) & (fp < 5000)).sum())}) · "
                f"효섭 충격 위치의 우리 높이 중앙 {np.median(hx_at):.1f}배", fontsize=8.5, loc="left", pad=2)
    rows5.append(dict(file=n, 우리=len(ours_["t"]), 효섭=len(pk), 효섭_2_5k울림=int(((fp >= 2000) & (fp < 5000)).sum()),
                      효섭충격의_우리높이_중앙=round(float(np.median(hx_at)), 1), 효섭충격의_우리높이_최소=round(float(hx_at.min()), 1) if len(hx_at) else None,
                      효섭충격_백색화배수_중앙=round(float(np.median(amp)), 1)))
axs[0].legend(fontsize=7.5, loc="upper right", ncol=2); axs[-1].set_xlabel("초 (1–11 kHz 신호)")
fig.tight_layout(h_pad=.3); fig.savefig(os.path.join(FIG, "검출기_비교.png"), dpi=130); plt.close(fig)

# 확대: val_012의 효섭 충격 중 우리가 못 잡은 것 6개
x = raw("val_012"); xh = highpass(x, fmax=11000); pk, amp, fp, w, sig = hy_imps(x); ours_t = detect_impacts(x, _hp=xh)["idx"]
far = [p for p in pk if np.min(np.abs(ours_t - p)) > 250][:8]
fig, axs = plt.subplots(2, 4, figsize=(13, 4.6))
for a, p in zip(axs.flat, far):
    seg = xh[p - 50:p + 300]; a.plot((np.arange(len(seg)) - 50) / 25, seg, lw=.6, color="C1")
    k = list(pk).index(p); a.set_title(f"t={p / 25000:.3f}s · 백색화 {amp[k]:.0f}σ · 울림 {fp[k] / 1000:.2f} kHz", fontsize=8); a.tick_params(labelsize=7)
fig.supxlabel("ms (val_012에서 효섭만 잡은 충격)", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(FIG, "val012_작은충격.png"), dpi=130); plt.close(fig)
with open(os.path.join(HERE, "결과.md"), "a", encoding="utf-8") as fh:
    fh.write("\n## ⑤ 두 검출기 비교 (효섭 충격 위치에서 우리 기준 높이 = 포락선/100 ms 바탕, 우리 문턱 4.5배)\n\n" + md_table(pd.DataFrame(rows5)) + "\n")
print("fig done")

# %% ⑥ 효섭 검출 사건 중 '작은 사건'(우리 기준 4.5배 미만) · '울림 없는 작은 사건'(뒤 1–7 ms / 앞 −7–−1 ms 에너지 < 2) 수: train 정상 · 고장 · val (11-27 추가)
#   자기상관만 FFT로 계산(효섭 ar_whiten과 같은 식). 원신호 캐시 4_outputs/cache/val_eda_raw.npz
import time
from scipy import linalg
from joblib import Parallel, delayed
from src.impacts import _baseline
ss = ss; R = ROOT
LOG=open(os.path.join(HERE, "spk_log.txt"),"w",encoding="utf-8"); T0=time.time()
def log(m): LOG.write(f"{time.time()-T0:6.1f}s {m}\n"); LOG.flush()
C=R+r"\4_outputs\cache\val_eda_raw.npz"
if os.path.exists(C):
    z=np.load(C,allow_pickle=True); Xtr,y,ntr,Xva,nva=z["Xtr"],z["y"],list(z["ntr"]),z["Xva"],list(z["nva"])
else:
    from src.data import load_dataset, load_folder
    Xtr,y,ntr=load_dataset(R+r"\2_data\TeamProject_train_변경"); log("train 읽음"); Xva,nva=load_folder(R+r"\2_data\TeamProject_val\val"); log("val 읽음")
    np.savez(C,Xtr=Xtr,y=y,ntr=np.array(ntr),Xva=Xva,nva=np.array(nva))
log("데이터 준비")
FS,TR,P=25000,500,40
hp10=ss.butter(4,10,"highpass",fs=FS,output="sos"); lp=ss.butter(4,1000,"lowpass",fs=FS,output="sos")
def ar_whiten(x):                       # 효섭 ar_whiten과 같은 식, 자기상관만 FFT로
    s=x[:20000]; F=np.fft.rfft(s,65536); r=np.fft.irfft(F*np.conj(F))[:P+1]
    a=linalg.solve_toeplitz(r[:P],r[1:P+1]); return ss.lfilter(np.r_[1,-a],[1],x)[P:]
def per_file(x):
    x=x-x.mean(); xh=ss.sosfiltfilt(hp10,x); w=ar_whiten(xh)[TR:-TR]
    sig=np.median(np.abs(w))/0.6745; pk,_=ss.find_peaks(np.abs(w),height=8*sig,distance=250); pk=pk+TR+P
    b=highpass(x,fmax=11000); e=ss.sosfiltfilt(lp,np.abs(ss.hilbert(b))); hx=e/_baseline(e)
    small=noring=0
    for p in pk:
        if p<300 or p+300>len(b): continue
        r=np.mean(b[p+25:p+175]**2)/(np.mean(b[p-175:p-25]**2)+1e-20)
        if hx[p]<4.5: small+=1; noring+= r<2
    return len(pk),small,noring
A=np.array(Parallel(n_jobs=-1)(delayed(per_file)(x) for x in Xtr)); log("train 끝")
V=np.array(Parallel(n_jobs=-1)(delayed(per_file)(x) for x in Xva)); log("val 끝")
o=open(os.path.join(HERE, "작은사건.txt"),"w",encoding="utf-8")
for lab,M in [("train 정상",A[y==0]),("train 고장",A[y==1]),("val",V)]:
    print(f"{lab}: 효섭검출 중앙 {np.median(M[:,0]):.0f} | 작은 사건(우리 4.5배 미만) 중앙 {np.median(M[:,1]):.0f} · 90% {np.percentile(M[:,1],90):.0f} | 그중 울림 없음 중앙 {np.median(M[:,2]):.0f} · 90% {np.percentile(M[:,2],90):.0f}",file=o)
print("train AUC: 효섭검출수",round(roc_auc_score(y,A[:,0]),3)," 작은 사건 수",round(roc_auc_score(y,A[:,1]),3)," 울림없는 작은 사건 수",round(roc_auc_score(y,A[:,2]),3),file=o)
for n in ["val_012","val_032","val_034","val_036","val_009","val_029","val_038","val_039","val_016"]:
    i=nva.index(n); print(n,"효섭",*V[i],file=o)
for k,lab in [(0,"정상"),(1,"고장")]:
    print(f"train {lab} 중 울림없는 작은 사건 ≥10:",[ntr[i] for i in np.where((y==k)&(A[:,2]>=10))[0]],file=o)
print("val 중 울림없는 작은 사건 ≥10:",[nva[i] for i in np.where(V[:,2]>=10)[0]],file=o)
log("완료")

# %% ⑦ V 피처 후보 표 (11-27): 우리 모델(H 15열 + 박자 + AE)은 그대로 두고 더할 후보. 실험 아님(EDA)
#   지표: train AUC(방향) · H 15열과 최대 |ρ| · ET OOF와 ρ · ET OOF에 더한 로지스틱(5-fold × 20) AUC · PR-AUC 변화
#         train 정상 기준 강건 z: val 후보 012 · 032 · 034 · 036 · train 고립 강타 정상 046 · 060 · train 고장 중앙 · val 전체 중앙
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score
m_ = joblib.load(os.path.join(ROOT, "3_단계", "7_최종·제출", "제출_EXP-062", "모델", "final_model_062.joblib")); core_ = m_.core_
Ttr_, Tva_, y_, ntr_, nva_ = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_tabs.joblib"))
Etr_, _ = core_._et_input(Ttr_); Hcols = list(core_.cols) + list(core_.peak); oof_et = core_.Z_[:, 0]
Itr_, Iva_ = joblib.load(os.path.join(ROOT, "4_outputs", "cache", "val_eda_imps.joblib"))
hxm = lambda I: np.log([max(im["height_x"]) if len(im["height_x"]) else 1.0 for im in I])
CANDV = {   # 이름: (train 값, val 값, 출처 · 뜻)
    "spike_noring": (np.log1p(A[:, 2]), np.log1p(V[:, 2]), "효섭 검출 사건 중 우리 기준 4.5배 미만 · 뒤 울림 없음(뒤/앞 에너지 < 2) 수"),
    "spike_small": (np.log1p(A[:, 1]), np.log1p(V[:, 1]), "효섭 검출 사건 중 우리 기준 4.5배 미만 수"),
    "D_nimp": (Htr["D_nimp"].values, Hva["D_nimp"].values, "효섭: AR(40) 백색화 > 8σ 사건 수 (log)"),
    "A_segk": (Htr["A_segk"].values, Hva["A_segk"].values, "효섭: 10 Hz 고역 신호 0.1초 구간 첨도 평균 (log)"),
    "D_segk": (Htr["D_segk"].values, Hva["D_segk"].values, "효섭: 백색화 신호 구간 첨도 평균 (log)"),
    "b25_5_segk": (Htr["b25_5_segk"].values, Hva["b25_5_segk"].values, "효섭: 2.5–5 kHz 대역 구간 첨도 (log)"),
    "b25_5_kurt": (Htr["b25_5_kurt"].values, Hva["b25_5_kurt"].values, "효섭: 2.5–5 kHz 대역 첨도 (log)"),
    "SK_200_3k": (Htr["SK_200_3k"].values, Hva["SK_200_3k"].values, "효섭: STFT 스펙트럼 첨도 200 Hz–3 kHz"),
    "I_density": (Htr["I_density"].values, Hva["I_density"].values, "효섭: 포락선 > 중앙 6배 구간 비율 (log)"),
    "AR_resid": (Htr["AR_resid"].values, Hva["AR_resid"].values, "효섭: train 정상 AR(40)로 예측하고 남는 몫 (log)"),
    "R_frac_2_5k": (Htr["R_frac_2_5k"].values, Hva["R_frac_2_5k"].values, "효섭: 큰 사건 20개 중 2–5 kHz 울림 비율"),
    "hx_max": (hxm(Itr_), hxm(Iva_), "우리 검출: 가장 큰 충격 배수 (log, V-2)"),
}
cvr = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
def cvs(Xm):
    au, ap = [], []
    for a_, b_ in cvr.split(Xm, y_):
        p_ = LogisticRegression(C=1.0, class_weight="balanced").fit(Xm[a_], y_[a_]).decision_function(Xm[b_])
        au.append(roc_auc_score(y_[b_], p_)); ap.append(average_precision_score(y_[b_], p_))
    return np.mean(au), np.mean(ap)
base_au, base_ap = cvs(oof_et[:, None])
def rz(v_tr, v):
    n_ = v_tr[y_ == 0]; md = np.median(n_); sc = 1.4826 * np.median(np.abs(n_ - md))
    sc = sc if sc > 1e-9 else (n_.std() + 1e-9); return (np.asarray(v) - md) / sc
rows7 = []
for k_, (vt, vv, desc) in CANDV.items():
    vt = np.nan_to_num(vt, nan=np.nanmedian(vt)); vv = np.nan_to_num(vv, nan=np.nanmedian(vt))
    a_ = roc_auc_score(y_, vt); rh = [abs(spearmanr(vt, Etr_[:, j]).correlation) for j in range(Etr_.shape[1])]; j_ = int(np.argmax(rh))
    au_, ap_ = cvs(np.column_stack([oof_et, vt])); zt, zv = rz(vt, vt), rz(vt, vv)
    r_ = dict(V후보=k_, AUC=round(max(a_, 1 - a_), 3), 방향="+" if a_ >= .5 else "−", H최대ρ=f"{max(rh):.2f} ({Hcols[j_]})",
              ET_OOFρ=round(spearmanr(vt, oof_et).correlation, 2), ΔAUC=round(au_ - base_au, 4), ΔPRAUC=round(ap_ - base_ap, 4))
    for n_ in ["val_012", "val_032", "val_034", "val_036"]:
        r_[n_[4:]] = round(float(zv[nva_.index(n_)]), 1)
    r_["N046"] = round(float(zt[ntr_.index("nor_046")]), 1); r_["N060"] = round(float(zt[ntr_.index("nor_060")]), 1)
    r_["train고장중앙"] = round(float(np.median(zt[y_ == 1])), 1); r_["val중앙"] = round(float(np.median(zv)), 1)
    rows7.append(r_)
VT = pd.DataFrame(rows7)
# 후보끼리 상관 (train 150, spearman)
keys = list(CANDV); M = np.array([[spearmanr(np.nan_to_num(CANDV[a][0]), np.nan_to_num(CANDV[b][0])).correlation for b in keys] for a in keys])
VC = pd.DataFrame(np.round(M, 2), index=keys, columns=keys)
pd.DataFrame({k_: np.r_[np.nan_to_num(v[0]), np.nan_to_num(v[1])] for k_, v in CANDV.items()}, index=list(ntr_) + list(nva_)).to_csv(
    os.path.join(HERE, "V피처후보_값.csv"), encoding="utf-8-sig")
with open(os.path.join(HERE, "V피처후보.md"), "w", encoding="utf-8") as fh:
    fh.write(f"기준: ET OOF logit만 로지스틱 AUC {base_au:.4f} · PR-AUC {base_ap:.4f} (5-fold × 20)\n\n")
    fh.write(md_table(VT) + "\n\n## 후보끼리 상관 (train, spearman)\n\n" + md_table(VC.reset_index()) + "\n\n## 뜻\n\n" +
             "\n".join(f"- **{k_}**: {v[2]}" for k_, v in CANDV.items()) + "\n")
print("V table done")

# %% ⑧ 우리 전처리(1–11 kHz)로 다시 정의한 V 후보 · 백색화가 필요한가 (11-27, src/featV.py)
from src.featV import v_row
VR_tr = pd.DataFrame(Parallel(n_jobs=-1)(delayed(v_row)(im["hp"]) for im in Itr_))
VR_va = pd.DataFrame(Parallel(n_jobs=-1)(delayed(v_row)(im["hp"]) for im in Iva_))
VR_tr.index, VR_va.index = list(ntr_), list(nva_)
rows8 = []
OURS = {"segk": "구간 첨도 (백색화 없음)", "segk_w": "구간 첨도 (AR 백색화)", "spike_noring": "울림 없는 작은 사건 수 (백색화 없이 검출)",
        "spike_noring_w": "울림 없는 작은 사건 수 (백색화 후 검출)", "spike_all": "짧게 튀는 사건 수 (백색화 없음)", "spike_all_w": "짧게 튀는 사건 수 (백색화)"}
for k_, desc in OURS.items():
    vt, vv = VR_tr[k_].values, VR_va[k_].values
    a_ = roc_auc_score(y_, vt); rh = [abs(spearmanr(vt, Etr_[:, j]).correlation) for j in range(Etr_.shape[1])]; j_ = int(np.argmax(rh))
    au_, ap_ = cvs(np.column_stack([oof_et, vt])); zt, zv = rz(vt, vt), rz(vt, vv)
    r_ = dict(후보=k_, 뜻=desc, AUC=round(max(a_, 1 - a_), 3), 방향="+" if a_ >= .5 else "−", H최대ρ=f"{max(rh):.2f} ({Hcols[j_]})",
              ΔAUC=round(au_ - base_au, 4), ΔPRAUC=round(ap_ - base_ap, 4))
    for n_ in ["val_012", "val_032", "val_034", "val_036"]:
        r_[n_[4:]] = round(float(zv[nva_.index(n_)]), 1)
    r_["N046"] = round(float(zt[ntr_.index("nor_046")]), 1); r_["N060"] = round(float(zt[ntr_.index("nor_060")]), 1)
    r_["train고장중앙"] = round(float(np.median(zt[y_ == 1])), 1); r_["val중앙"] = round(float(np.median(zv)), 1)
    rows8.append(r_)
V8 = pd.DataFrame(rows8)
rho_pairs = {f"{a} ~ {b}": round(spearmanr(VR_tr[a], VR_tr[b]).correlation, 2) for a, b in
             [("segk", "segk_w"), ("segk", "spike_noring"), ("segk_w", "spike_noring_w"), ("spike_noring", "spike_noring_w")]}
rho_hy = {f"{a} ~ 효섭 {b}": round(spearmanr(VR_tr[a], CANDV[b][0]).correlation, 2) for a, b in
          [("segk", "A_segk"), ("segk_w", "D_segk"), ("spike_noring_w", "spike_noring")]}
pd.concat([VR_tr, VR_va]).to_csv(os.path.join(HERE, "V피처_우리전처리_값.csv"), encoding="utf-8-sig")
with open(os.path.join(HERE, "V피처_우리전처리.md"), "w", encoding="utf-8") as fh:
    fh.write(f"우리 전처리(1–11 kHz 대역통과, 양끝 20 ms 자름). 기준: ET OOF만 로지스틱 AUC {base_au:.4f} · PR-AUC {base_ap:.4f}\n\n")
    fh.write(md_table(V8) + "\n\n후보끼리 상관(train): " + " · ".join(f"{k} {v}" for k, v in rho_pairs.items()) +
             "\n\n효섭 원래 정의와 상관(train): " + " · ".join(f"{k} {v}" for k, v in rho_hy.items()) + "\n")
print("V ours done")
