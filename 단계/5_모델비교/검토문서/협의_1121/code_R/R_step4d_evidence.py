"""R 4d: 정상 증거 연속 피처 (본체 모드 울림 · 한 사건 집중 · 좁은 순음) + 오경보 정상 위치 + C4 (고장이 같이 내려가는가) + 로지스틱 CV + t-SNE 정상 순도.
정의는 기전으로 먼저 고정(아래), 라벨은 보고·평가에만 쓴다. 출력 out/R4d.txt"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
s4b = open(os.path.join(here, "R_step4b_select.py"), encoding="utf-8").read().split("# ================= 1.")[0]
exec(compile(s4b, "R4b_head", "exec"))
s4c = open(os.path.join(here, "R_step4c_margin.py"), encoding="utf-8").read()
exec(compile("import itertools\n" + s4c[s4c.index("def pick_ring_m"):s4c.index('p("\\n[B]')], "R4c_defs", "exec"))
buf = io.StringIO()
def p(*a): print(*a, file=buf)

# ---------------- 정상 증거 피처 정의 (기전, 라벨 없이 정의) ----------------
BODY = (5500, 7750)   # 본체 모드 대역 (EXP-034, 결정.md)
HF = 9250
def evid(i):
    im = imps[i]; r = im["rings"]; h = np.asarray(im["height_x"], float); k = len(h)
    fr_, fz_, ft_ = (r[:, 0], r[:, 1], r[:, 2]) if k else (np.zeros(0),) * 3
    inb = (fr_ >= BODY[0]) & (fr_ < BODY[1]) & (ft_ >= 0.5)
    body_cnt = np.log1p(inb.sum()); body_frac = inb.mean() if k else 0.0
    hs = np.sort(h)[::-1]
    one_share = hs[0] / hs.sum() if k else 0.0
    gap12 = np.log10(hs[0] / hs[1]) if k > 1 else (1.0 if k == 1 else 0.0)
    hh = h[(fr_ >= HF)] if k else np.zeros(0)
    hf_conc = (hh.max() / hh.sum()) if len(hh) else 0.0
    hf_n = len(hh)
    return body_cnt, body_frac, one_share, gap12, hf_conc, hf_n
EV = np.array([evid(i) for i in range(n_all)])
evn = ["body_cnt", "body_frac", "one_share", "gap12", "hf_conc", "hf_n"]
P83, W83 = R["P83"], R["W83"]
tone = P83 / np.maximum(W83, 50) * 100
EV = np.column_stack([EV, tone]); evn.append("tone83")
nor_all = np.zeros(n_all, bool); nor_all[TRI] = y == 0
fau_all = np.zeros(n_all, bool); fau_all[TRI] = y == 1
FA = ["N116", "N065", "N059", "N037", "N110", "N095", "N055", "N025", "N097", "N101", "N120", "N019", "N092", "N103", "N001", "N048"]
fa_i = [sn.index(s) for s in FA]

p("[1] 정상 증거 피처: 단변량 AUC(고장=1, 정상 증거면 AUC<0.5), 중앙값 정상/고장/val, val 이동")
for j, nm_ in enumerate(evn):
    v = EV[:, j]; auc = roc_auc_score(y, v[TRI])
    qn = np.percentile(v[nor_all], [5, 50, 95]); mv = np.median(v[VAL]); frac_lo = np.mean(v[VAL] < qn[0]); frac_hi = np.mean(v[VAL] > qn[2])
    p(f"   {nm_:>10}: AUC {auc:.3f} | 중앙 정상 {np.median(v[nor_all]):.2f} 고장 {np.median(v[fau_all]):.2f} val {mv:.2f} | 정상 5/50/95% {qn.round(2)} | val이 정상 5% 아래 {frac_lo:.2f} 95% 위 {frac_hi:.2f}")
p("\n[2] 자주 오경보 정상의 증거 값 (정상 안 백분위, 높을수록 정상 쪽 증거)")
p("     " + " ".join(f"{s:>8}" for s in ["file"] + evn))
for s, i in zip(FA, fa_i):
    p(f"     {s:>6}" + " ".join(f"{(np.mean(EV[nor_all, j] <= EV[i, j])):8.2f}" for j in range(len(evn))) + "   | 원값 " + " ".join(f"{EV[i,j]:.2f}" for j in range(len(evn))))
p("\n[3] C4: 증거가 높아 같이 내려갈 고장 (오경보 정상군 중앙값 이상 / 정상 중앙 이상인 고장)")
for j, nm_ in enumerate(evn):
    thr = np.median(EV[fa_i, j]); hi = [sn[i] for i in np.flatnonzero(fau_all) if EV[i, j] >= thr]
    thr2 = np.percentile(EV[nor_all, j], 50); hi2 = [sn[i] for i in np.flatnonzero(fau_all) if EV[i, j] >= thr2]
    p(f"   {nm_:>10}: 오경보군 중앙 {thr:.2f} 이상 고장 {len(hi)}개 ({' '.join(hi[:12])}) | 정상 중앙 {thr2:.2f} 이상 고장 {len(hi2)}개 | val {int(np.sum(EV[VAL, j] >= thr))}개가 오경보군 중앙 이상")
b3 = (np.expm1(EV[:, 0]).round() >= 3) & (EV[:, 1] >= 0.9)
p(f"\n   [EXP-034 이진형 재확인: 본체 울림 >=3 이고 비율>=0.9] 정상 {int(b3[nor_all].sum())} 고장 {int(b3[fau_all].sum())} val {int(b3[VAL].sum())} ({' '.join(sn[i] for i in VAL if b3[i])}); FA: " + " ".join(f"{s}:{int(b3[i])}" for s, i in zip(FA, fa_i)))
Xtr, ytr, ntr = load_train(); mean_ = Xtr.mean(1)
low = [i for i in range(len(ntr)) if abs(mean_[i]) > 0.5]
p("\n[4] 평균 |DC|>0.5 g 파일: " + " ".join(f"{sn[i]}({mean_[i]:.2f})" for i in low))
Tz = T2.iloc[TRI].reset_index(drop=True)
for i in low:
    z = {}
    for c in Tz.columns:
        v = Tz[c].values.astype(float); ref = v[nor]; mad = 1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-9
        z[c] = (v[i] - np.median(ref)) / mad
    top = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:3]
    p(f"   {sn[TRI[i]]}: T2에서 가장 어긋난 열 " + ", ".join(f"{k} z={v:.1f}" for k, v in top))

# ---------------- 로지스틱 CV: 폴드 안 u(m=0.2) + 증거 축 ----------------
FMj = joblib.load(os.path.join(root, "outputs", "final", "final_model.joblib")); Zoof = np.asarray(FMj.core_.oof_z_)
vz = pd.read_csv(os.path.join(here, "..", "code_D", "val_A_parts.csv"), encoding="utf-8-sig"); Zall = np.vstack([Zoof, vz[["ae_z", "et_z"]].values])
M = 0.2; SEEDS = range(5)
folds = [(s, tr, te) for s in SEEDS for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(n), y)]
fsel = {}
for s, tr, te in folds:
    sel = {"ring": pick_ring_m(tr, M, True), "spec": pick_spec_m(tr, M), "J": pick_J_m(tr, M), "L": pick_L_m(tr, M)}
    fsel[(s, tuple(te[:3]))] = (sel, u_all(sel, tr))
def evz(tr):
    ref = tr[nor[tr]]; Zc = np.empty_like(EV)
    for j in range(EV.shape[1]): Zc[:, j] = np.clip(robust_z(EV[TRI][ref, j])(EV[:, j]), -5, 5)
    return Zc
def feats(kind, U, Zc):
    cl = lambda ks: np.max(np.column_stack([np.clip(U[k], 0, 3) for k in ks]), 1)
    ek = {"b": [0, 1], "o": [2, 3], "t": [6], "all": [0, 2, 6]}
    if kind == "M0": return Zall
    A3 = cl(("ring", "spec", "L"))
    if kind == "M1": return np.column_stack([Zall, A3])
    e = kind.split("+")[1]
    return np.column_stack([Zall, A3, Zc[:, ek[e]]])
kinds = ["M0", "M1", "M1+b", "M1+o", "M1+t", "M1+all"]
res = {}
for kd in kinds:
    aps, oofs = [], []
    for seed in SEEDS:
        oof = np.zeros(n)
        for s, tr, te in [x for x in folds if x[0] == seed]:
            sel, U = fsel[(s, tuple(te[:3]))]; F = feats(kd, U, evz(tr))
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(F[TRI][tr], y[tr]); oof[te] = m.decision_function(F[TRI][te])
        aps.append(average_precision_score(y, oof)); oofs.append(oof)
    res[kd] = (np.mean(aps), np.std(aps), np.mean(oofs, 0), oofs)
fa_tr = [i for i in fa_i if i < len(TRI)]
p("\n[5] 로지스틱 CV (탐색용, 5겹x5 seed, 선택·z 기준 모두 폴드 안; AE z, ET z = 최종 모델 OOF). PR-AUC | 정상 OOF 95% 분위 임계에서 고장 재현 | FA 16개 평균 위쪽 백분위 | FA 중 임계 초과 수")
for kd, (ap, sd, mo, oofs) in res.items():
    rec, fp_, pct = [], [], []
    for oof in oofs:
        thr = np.percentile(oof[nor], 95); rec.append((oof[~nor] > thr).mean()); fp_.append(np.sum(oof[fa_tr] > thr))
        pct.append(np.mean([(oof[nor] >= oof[i]).mean() for i in fa_tr]))
    p(f"   {kd:>8}: PR-AUC {ap:.3f} ± {sd:.3f} | 재현@정상95% {np.mean(rec):.3f} | FA 평균 위쪽 백분위 {np.mean(pct):.3f} | FA 임계 초과 {np.mean(fp_):.1f}")
p("   핵심 고장 순위 (고장보다 점수 높은 정상 비율, 낮을수록 좋음):")
keyf = ["A002", "A003", "A006", "A010", "A029", "A030", "A014", "A017", "A004", "A022", "A005", "A001", "A007", "A009"]
for kd, (ap, sd, mo, oofs) in res.items():
    rk = [(mo[nor] >= mo[sn.index(k)]).mean() for k in keyf]; p(f"   {kd:>8}: " + " ".join(f"{k}:{r:.2f}" for k, r in zip(keyf, rk)))
p("   FA 정상 평균 OOF의 정상 안 위쪽 비율 (작을수록 정상 무리 안쪽):")
for kd, (ap, sd, mo, oofs) in res.items():
    p(f"   {kd:>8}: " + " ".join(f"{sn[i]}:{(mo[nor] >= mo[i]).mean():.2f}" for i in fa_tr))
trall = np.arange(n); sel = {"ring": pick_ring_m(trall, M, True), "spec": pick_spec_m(trall, M), "J": pick_J_m(trall, M), "L": pick_L_m(trall, M)}; U = u_all(sel, trall); Zc = evz(trall)
p("\n[6] val (전체 train 학습, 경계 = 학습 정상 점수 95% 분위): 경계 ±sd 안 val 수 · val>=경계 수 · 계수")
for kd in kinds:
    F = feats(kd, U, Zc); m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500).fit(F[TRI], y)
    s_tr, s_va = m.decision_function(F[TRI]), m.decision_function(F[VAL]); thr = np.percentile(s_tr[nor], 95); sd = s_tr[nor].std()
    p(f"   {kd:>8}: ±0.25sd {int((abs(s_va-thr)<.25*sd).sum()):2d} ±0.5sd {int((abs(s_va-thr)<.5*sd).sum()):2d} ±1sd {int((abs(s_va-thr)<sd).sum()):2d} | val>=경계 {int((s_va>=thr).sum())} | 계수 {np.round(m.coef_[0], 2)}")
# ---------------- t-SNE 정상 순도 ----------------
Tt, Lt, fr = cached_feat2("train", Xtr)
FJ = json.load(open(os.path.join(root, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
pk = PeakRef(float(fr[1] - fr[0])).fit(Lt[nor]).transform(Lt); Tt = Tt.copy(); Tt["H16_peak_h"], Tt["H17_peak_w"] = pk[:, 0], pk[:, 1]
BASE = Tt[list(FJ["final"])].values
def zs(A, ref):
    Z = np.empty_like(A, float)
    for j in range(A.shape[1]): Z[:, j] = np.clip(robust_z(A[ref, j])(A[:, j]), -8, 8)
    return Z
def nn(Z, idx):
    return NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z[idx], return_distance=False)[:, 1:]
fa_idx = np.flatnonzero(~nor); fa_nor_idx = [i for i in fa_tr if y[i] == 0]
p("\n[7] 입력 공간별 (전체 train 정의, in-sample, 참고용): 고장 5-NN 순도 | 오경보 정상 5-NN 중 정상 비율 (높을수록 정상 무리 안)")
Utr = {k: np.clip(U[k][TRI], 0, 3) for k in EXPK}; A3tr = np.max(np.column_stack([Utr[k] for k in ("ring", "spec", "L")]), 1)[:, None]
Ztr = zs(EV[TRI], np.flatnonzero(nor)); B = zs(BASE, np.flatnonzero(nor))
spaces = {"base16": B, "base16+A3": np.hstack([B, A3tr]), "base16+증거(body,one,tone)": np.hstack([B, Ztr[:, [0, 2, 6]]]),
          "base16+A3+증거": np.hstack([B, A3tr, Ztr[:, [0, 2, 6]]])}
for nm_, Z in spaces.items():
    pf = y[nn(Z, fa_idx)].mean(); pn = (1 - y[nn(Z, fa_nor_idx)]).mean()
    p(f"   {nm_:>26}: 고장 순도 {pf:.3f} | 오경보 정상 {len(fa_nor_idx)}개 정상비율 {pn:.3f}")
open(os.path.join(here, "out", "R4d.txt"), "w", encoding="utf-8").write(buf.getvalue()); print("done")
