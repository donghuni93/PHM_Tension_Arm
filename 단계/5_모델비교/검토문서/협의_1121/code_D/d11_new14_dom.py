# D2: new14 위에 dom/n_ev(사건 구조)를 더하면 새 정보가 있나 + new14 열의 val 이동. quick_cv 수준(탐색용). 결과는 파일로.
import os, sys
FD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_F"); FD = os.path.abspath(FD)
src = open(os.path.join(FD, "step5_final.py"), encoding="utf-8").read()
cut = src.index("RANDOM = (fault.sum()")
pre = src[:cut].replace("result_F5", "result_D11_pre").replace("figs_F5", "figsD11pre").replace("F5_diag", "D11pre_diag")
pre = pre.replace("def P(*a):\n    s = \" \".join(str(x) for x in a); print(s, flush=True); OUT.write(s + \"\n\"); OUT.flush()", "def P(*a):\n    s = \" \".join(str(x) for x in a); OUT.write(s + \"\n\"); OUT.flush()")
__file__ = os.path.join(FD, "step5_final.py")
exec(compile(pre, "step5_pre", "exec"))
HD = os.path.dirname(os.path.abspath(__file__))
O = open(os.path.join(HD, "d11_log.txt"), "w", encoding="utf-8")
def Q(*a):
    O.write(" ".join(str(x) for x in a) + "\n"); O.flush()
# 사건 구조: 50 ms 안 충격 묶음, 사건 높이 = 묶음 최대 height_x
dom = np.ones(N); nev = np.zeros(N)
for i, f in enumerate(files):
    d = IMP[IMP.file == f].sort_values("t")
    if len(d) == 0: continue
    ev = []; cur = None
    for t, h in zip(d.t.values, d.height_x.values):
        if cur is not None and t - cur[0] <= 0.05: cur[1] = max(cur[1], h); cur[0] = t
        else:
            if cur is not None: ev.append(cur[1])
            cur = [t, h]
    ev.append(cur[1]); ev = np.array(ev); dom[i] = ev.max() / ev.sum(); nev[i] = len(ev)
Fidx = np.arange(ntr)[:, None].astype(float)
def mk(extra):
    def build(sp, tr):
        return np.hstack([NEW16] + [e.reshape(N, -1) for e in extra])
    return build
class SpET2(BaseEstimator):
    def __init__(self, ex=None): self.ex = ex
    def fit(self, Fi, yy):
        tr = Fi[:, 0].astype(int); self.M_ = np.hstack([NEW16] + [e.reshape(N, -1) for e in EXTRAS[self.ex]])
        self.m_ = make_et(seed=0)().fit(self.M_[tr], yy); return self
    def predict_proba(self, Fi): return self.m_.predict_proba(self.M_[Fi[:, 0].astype(int)])
EXTRAS = {"new14": [], "+dom": [dom], "+n_ev": [np.log1p(nev)], "+dom,n_ev": [dom, np.log1p(nev)]}
Q("공간 | PR-AUC ± sd | 위험 정상 | 오경보 FA 점수백분위 | 고장 5-NN 순도(폴드밖) | FA 정상 5-NN 순도")
for k in EXTRAS:
    q = quick_cv(Fidx, y, np.arange(ntr), lambda k=k: SpET2(k), seeds=range(5))
    oof = q["oof"]; fa = []
    for s_ in range(oof.shape[0]):
        pn = oof[s_, nor]
        for i in fa_idx: fa.append(np.mean(pn <= oof[s_, i]))
    risk = np.mean([(oof[s_, nor] > np.percentile(oof[s_, fault], 10)).sum() for s_ in range(oof.shape[0])])
    pv, pf = [], []
    M = np.hstack([NEW16] + [e.reshape(N, -1) for e in EXTRAS[k]])
    for s in range(5):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx_all, y, idx_all):
            Zf = np.clip(standardize(M, tr), -10, 10)
            who = te[fault[te]]; wn = np.array([i for i in te if isFA[i]])
            if len(who): pv.append(purity(Zf, who))
            if len(wn): pf.append(npurity(Zf, wn))
    Q(f"{k:10s} | {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std(ddof=1):.3f} | {risk:.1f} | {np.mean(fa):.3f} | {np.mean(pv):.3f} | {np.mean(pf):.3f}")
# new14 열별 val 이동: train 정상 중앙/IQR 대비 val 중앙(단위 IQR), train 정상 5-95% 밖 비율
Q("\n== new14 열 val 이동 (val 중앙 - 정상 중앙)/정상IQR, val 중 정상 5-95% 밖 비율 ==")
nm = FINAL16
Mn = NEW16[:ntr][nor[:ntr]] if len(nor) == ntr else NEW16[:ntr][nor]
Mv = NEW16[ntr:]
for j, c in enumerate(nm):
    q1, md, q3 = np.percentile(Mn[:, j], [25, 50, 75]); lo, hi = np.percentile(Mn[:, j], [5, 95])
    iqr = max(q3 - q1, 1e-9)
    Q(f"{c:16s} shift {(np.median(Mv[:, j]) - md) / iqr:+.2f}  outside {np.mean((Mv[:, j] < lo) | (Mv[:, j] > hi)):.2f}")
Q("dom/n_ev val shift: dom 정상중앙", np.median(dom[:ntr][nor]) if nor.dtype == bool and len(nor)==ntr else "", "val", np.median(dom[ntr:]), "n_ev val", np.median(nev[ntr:]))
