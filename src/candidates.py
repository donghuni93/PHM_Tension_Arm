"""EXP-019 스크리닝 후보 모델 → 전문가 함수 (tr, te) → (te, 1) 이상 점수

src/experts.py와 같은 규칙: 학습 파일 tr 안에서만 fit, 디스크 캐시(joblib_experts)로 분할 재사용.
ScoreStack((ae_mean, et_logit, 새 전문가), combine="lr")에 그대로 끼워 보완성을 본다.
하이퍼파라미터는 문헌 기본값·실행 전 고정값 (스크리닝에서 튜닝하지 않음, 결정.md '모델 탐색').
딥러닝(1D-CNN·합성곱 AE·Deep SAD·2D-CNN)은 GPU에서 순서대로(warm_cache n_jobs=1) 돌린다.

후보 (회색지대 정보를 새로 볼 가능성 순)
 1 minirocket_ridge   원신호 → MiniRocket(커널 10k) → 릿지
 2 cnn1d_mil          원신호 0.2초 창 → 1D-CNN(WDCNN류) → 파일 = 상위 3창 평균 (MIL)
 3 convae_stft        log STFT 구간 → 2D 합성곱 AE(정상만) → 파일 = 오차 상위 3구간 평균
 4 deepsad_stft       log STFT 구간 → Deep SAD(정상 + 고장 라벨) → 파일 = 거리 상위 3구간 평균
 5 cnn2d_mil          log STFT 구간 → 2D-CNN → 파일 = 상위 3구간 평균 (MIL)
 6 ar_resid_lr        정상으로 AR(32) 학습 → 잔차 충격 피처 5개 → 로지스틱
 7 tabpfn_ext         ext150 → TabPFN  (미사용: 가중치에 외부 API 키 필요, 사용자 결정 11-14)
 8 mspc_ext           ext150 → 정상 PCA(분산 90%) → T² + SPE 결합 지수
 9 catboost_ext       ext150 → CatBoost(500회, 클래스 균형)
"""
import os
import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, lfilter
from scipy.stats import kurtosis
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.exp import CACHE
from src.experts import X, y, E, mem, FS


def _hp200(X):
    """평균 제거 → 200 Hz 고역통과 (EXP-012 AE와 같은 전처리, 라벨 안 씀)"""
    Xc = X - X.mean(1, keepdims=True)
    return sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1).astype(np.float32)


_XH = None


def xh():
    global _XH
    if _XH is None:
        _XH = _hp200(X)
    return _XH


def _logit(p, n):
    p = np.clip(p, 1 / (2 * n), 1 - 1 / (2 * n))
    return np.log(p / (1 - p))


# ---------- 1. MiniRocket + 릿지 (Dempster 2021 기본값) ----------
@mem.cache
def minirocket_feats(tr, seed=0):
    """MiniRocket을 학습 파일 tr로 fit(편향 = tr 합성곱 출력의 분위수, 라벨 안 씀) → 150개 전체 변환 (150, ~10k)
    기본값: 커널 10,000개, 팽창 최대 32. 원신호 50,000점 그대로 (8 kHz대 정보를 버리지 않게 다운샘플 안 함)"""
    from aeon.transformations.collection.convolution_based import MiniRocket
    tr = np.array(tr)
    Xh = xh()[:, None, :]
    m = MiniRocket(random_state=seed).fit(Xh[tr])
    return np.asarray(m.transform(Xh), dtype=np.float32)


@mem.cache
def minirocket_ridge(tr, te):
    """MiniRocket 특징(PPV: 패턴이 나타나는 비율) → 표준화 → RidgeClassifierCV(α 1e-3–1e3, 원 논문 기본) → te 결정값 (te, 1)"""
    tr, te = np.array(tr), np.array(te)
    F = minirocket_feats(tuple(tr.tolist()))
    m = make_pipeline(StandardScaler(), RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced"))
    m.fit(F[tr], y[tr])
    return m.decision_function(F[te])[:, None]


# ---------- 딥러닝 공용 ----------
def _torch():
    import torch
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    return torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")


NPER, HOP, CH, CHOP = 512, 256, 32, 16      # STFT 512점(48.8 Hz) · 홉 256 → 약 197프레임. 구간 32프레임(0.33초), 홉 16


def _stft_all():
    """log10 STFT 파워 (150, 256, 프레임). Nyquist bin 하나는 버려 256칸. 라벨 안 씀, 디스크 캐시"""
    p = os.path.join(CACHE, "stft_hp200_512.npy")
    if os.path.exists(p):
        return np.load(p)
    _, _, Z = stft(xh(), fs=FS, nperseg=NPER, noverlap=NPER - HOP, boundary=None, padded=False, axis=-1)
    S = np.log10(np.abs(Z[:, :256, :]) ** 2 + 1e-12).astype(np.float32)
    np.save(p, S)
    return S


def _chunks(S):
    """(n, 256, T) → (n, k, 1, 256, 32) 겹치는 구간"""
    starts = range(0, S.shape[-1] - CH + 1, CHOP)
    return np.stack([S[:, :, s:s + CH] for s in starts], 1)[:, :, None]


def _stft_std(tr):
    """학습 정상 기준으로 주파수 bin별 표준화한 구간 (150, k, 1, 256, 32)"""
    S = _stft_all()
    n = S[np.array(tr)[y[np.array(tr)] == 0]]
    mu, sd = n.mean((0, 2))[None, :, None], n.std((0, 2))[None, :, None] + 1e-6
    return _chunks((S - mu) / sd).astype(np.float32)


def _topk_mean(v, k=3):
    return v.topk(min(k, v.shape[1]), dim=1).values.mean(1)


def _train_mil(net, inst_fn, tr, epochs, torch, dev, bs=16):
    """파일 라벨 MIL 학습: 파일마다 인스턴스(창·구간) 로짓 → 상위 3개 평균 = 파일 로짓 → BCE(고장 가중 = 정상/고장 수)"""
    tr = np.array(tr); yt = y[tr].astype(np.float32)
    pw = torch.tensor((yt == 0).sum() / max(1, (yt == 1).sum()), device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // bs)):
            xb = inst_fn(tr[b], train=True)                    # (B, k, ...)
            B, k = xb.shape[:2]
            z = net(xb.reshape(B * k, *xb.shape[2:])).reshape(B, k)
            loss = lossf(_topk_mean(z), torch.tensor(yt[b], device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def _score_mil(net, inst_fn, te, torch):
    net.eval(); out = []
    with torch.no_grad():
        for b in np.array_split(np.array(te), max(1, len(te) // 16)):
            xb = inst_fn(b, train=False)
            B, k = xb.shape[:2]
            out.append(_topk_mean(net(xb.reshape(B * k, *xb.shape[2:])).reshape(B, k)).cpu().numpy())
    return np.concatenate(out)[:, None]


# ---------- 2. 1D-CNN + 창 MIL ----------
WIN, WHOP, KTR = 5000, 2500, 16            # 0.2초 창, 평가 홉 0.1초(19창), 학습은 파일마다 무작위 시작 16창


def _net1d(torch):
    nn = torch.nn
    def blk(i, o): return [nn.Conv1d(i, o, 3, padding=1), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
    return nn.Sequential(nn.Conv1d(1, 16, 64, stride=8, padding=28), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                         *blk(16, 32), *blk(32, 64), *blk(64, 64),
                         nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveMaxPool1d(1),
                         nn.Flatten(), nn.Linear(64, 1), nn.Flatten(0))


@mem.cache
def cnn1d_mil(tr, te, epochs=40):
    """WDCNN류 1D-CNN (첫 층 넓은 커널 64, stride 8). 입력 = 200 Hz 고역통과 원신호 / 학습 정상 표준편차 중앙값(진폭 정보 유지)"""
    torch, dev = _torch()
    tr, te = np.array(tr), np.array(te)
    Xh = xh(); s = np.median(Xh[tr[y[tr] == 0]].std(1))
    T = torch.tensor(Xh / s, device=dev)
    starts_ev = np.arange(0, Xh.shape[1] - WIN + 1, WHOP)
    rng = np.random.default_rng(0)

    def inst(idx, train):
        st = rng.integers(0, Xh.shape[1] - WIN, (len(idx), KTR)) if train else np.tile(starts_ev, (len(idx), 1))
        ii = torch.tensor(st, device=dev)[..., None] + torch.arange(WIN, device=dev)
        return T[torch.tensor(idx, device=dev)[:, None, None], ii][:, :, None, :]   # (B, k, 1, WIN)

    net = _train_mil(_net1d(torch).to(dev), inst, tr, epochs, torch, dev)
    return _score_mil(net, inst, te, torch)


# ---------- 2D 공용 (log STFT 구간 1×256×32) ----------
def _enc2d(torch, bias=True, bn=True, act=None):
    nn = torch.nn
    act = act or nn.ReLU
    L = []
    for i, o in ((1, 16), (16, 32), (32, 32), (32, 32)):              # 256×32 → 16×2 (×32ch = 1,024, 입력의 1/8)
        L += [nn.Conv2d(i, o, 3, stride=2, padding=1, bias=bias)] + ([nn.BatchNorm2d(o, affine=bias)] if bn else []) + [act()]
    return nn.Sequential(*L)


def _instance_2d(Q, dev, torch):
    def inst(idx, train):
        return Q[torch.tensor(np.asarray(idx), device=dev)]
    return inst


# ---------- 3. 합성곱 AE (정상만) ----------
@mem.cache
def convae_stft(tr, te, epochs=60):
    """정상 구간만으로 2D 합성곱 AE 학습(MSE) → 구간별 복원 오차 → 파일 = 상위 3구간 평균 → log10.
    EXP-012 AE(1초 PSD 평균)와 달리 0.33초 시간 해상도 + 48.8 Hz 주파수 해상도"""
    torch, dev = _torch(); nn = torch.nn
    tr, te = np.array(tr), np.array(te)
    Q = torch.tensor(_stft_std(tr), device=dev)                       # (150, k, 1, 256, 32)
    dec = nn.Sequential(*[m for i, o in ((32, 32), (32, 32), (32, 16)) for m in
                          (nn.ConvTranspose2d(i, o, 4, stride=2, padding=1), nn.BatchNorm2d(o), nn.ReLU())],
                        nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1))
    net = nn.Sequential(_enc2d(torch), dec).to(dev)
    N = Q[torch.tensor(tr[y[tr] == 0], device=dev)].reshape(-1, 1, 256, CH)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(N), generator=g).split(64):
            loss = ((net(N[b.to(dev)]) - N[b.to(dev)]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Z = Q[torch.tensor(te, device=dev)]
        B, k = Z.shape[:2]
        err = ((net(Z.reshape(B * k, 1, 256, CH)) - Z.reshape(B * k, 1, 256, CH)) ** 2).mean((1, 2, 3)).reshape(B, k)
        return torch.log10(_topk_mean(err)).cpu().numpy()[:, None]


# ---------- 4. Deep SAD (Ruff 2020, η = 1) ----------
@mem.cache
def deepsad_stft(tr, te, epochs=50):
    """편향 없는 인코더 φ → 중심 c(초기 φ의 정상 평균). 정상 구간은 ‖φ−c‖²를 줄이고 고장 파일 구간은 역수를 줄임
    (고장 파일의 모든 구간을 고장으로 보는 근사). 파일 = 거리 상위 3구간 평균 → log10"""
    torch, dev = _torch(); nn = torch.nn
    tr, te = np.array(tr), np.array(te)
    Q = torch.tensor(_stft_std(tr), device=dev)
    net = nn.Sequential(_enc2d(torch, bias=False, bn=False, act=nn.LeakyReLU), nn.Flatten(),
                        nn.Linear(32 * 16 * 2, 32, bias=False)).to(dev)
    Xtr = Q[torch.tensor(tr, device=dev)]
    k = Xtr.shape[1]
    Xtr = Xtr.reshape(-1, 1, 256, CH)
    lab = torch.tensor(np.repeat(y[tr], k), device=dev).float()
    with torch.no_grad():
        c = net(Xtr[lab == 0]).mean(0)
        c[(c.abs() < 0.1) & (c < 0)], c[(c.abs() < 0.1) & (c >= 0)] = -0.1, 0.1
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6)
    g = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(epochs):
        for b in torch.randperm(len(Xtr), generator=g).split(128):
            b = b.to(dev)
            d = ((net(Xtr[b]) - c) ** 2).sum(1)
            loss = torch.where(lab[b] == 0, d, 1.0 / (d + 1e-6)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Z = Q[torch.tensor(te, device=dev)]
        B = Z.shape[0]
        d = ((net(Z.reshape(B * k, 1, 256, CH)) - c) ** 2).sum(1).reshape(B, k)
        return torch.log10(_topk_mean(d) + 1e-12).cpu().numpy()[:, None]


# ---------- 5. 2D-CNN + 구간 MIL ----------
@mem.cache
def cnn2d_mil(tr, te, epochs=40):
    """log STFT 구간 → 2D-CNN(인코더 + 전역 최대 풀링 + 선형) → 파일 = 상위 3구간 평균 로짓"""
    torch, dev = _torch(); nn = torch.nn
    tr, te = np.array(tr), np.array(te)
    Q = torch.tensor(_stft_std(tr), device=dev)
    net = nn.Sequential(_enc2d(torch), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(32, 1), nn.Flatten(0)).to(dev)
    inst = _instance_2d(Q, dev, torch)
    net = _train_mil(net, inst, tr, epochs, torch, dev)
    return _score_mil(net, inst, te, torch)


# ---------- 6. AR 예측 잔차 ----------
AR_P = 32


@mem.cache
def ar_feats(tr):
    """학습 정상 파일(앞 10,000점씩)로 AR(32) 최소제곱 → 150개 전체 잔차 → 충격 피처 5개 (150, 5)
    잔차 = 정상 진동으로 예측 못 하는 부분(충격). 피처: log 첨도, log(잔차 RMS/원신호 RMS),
    10 ms 창 최댓값 > 8σ(MAD) 창 수(log1p), 두 번째/첫 번째 창 최댓값, log 파고율"""
    tr = np.array(tr); Xh = xh().astype(np.float64)
    rows, tgt = [], []
    for x in Xh[tr[y[tr] == 0]]:
        x = x[:10000] / x[:10000].std()
        rows.append(np.lib.stride_tricks.sliding_window_view(x[:-1], AR_P)[:, ::-1]); tgt.append(x[AR_P:])
    a = np.linalg.lstsq(np.vstack(rows), np.concatenate(tgt), rcond=None)[0]
    F = []
    for x in Xh:
        e = lfilter(np.r_[1, -a], [1], x)[AR_P:]
        sig = 1.4826 * np.median(np.abs(e - np.median(e))) + 1e-12
        wm = np.abs(e[: len(e) // 250 * 250]).reshape(-1, 250).max(1)
        top = np.sort(wm)[-2:]
        F.append([np.log(kurtosis(e) + 3), np.log(e.std() / x.std()), np.log1p((wm > 8 * sig).sum()),
                  top[0] / top[1], np.log(np.abs(e).max() / e.std())])
    return np.array(F)


@mem.cache
def ar_resid_lr(tr, te):
    tr, te = np.array(tr), np.array(te)
    F = ar_feats(tuple(tr.tolist()))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced")).fit(F[tr], y[tr])
    return m.decision_function(F[te])[:, None]


# ---------- 7. TabPFN ----------
@mem.cache
def tabpfn_ext(tr, te):
    """ext150 → TabPFNClassifier 기본값(GPU) → logit 확률"""
    from tabpfn import TabPFNClassifier
    tr, te = np.array(tr), np.array(te)
    m = TabPFNClassifier(device="cuda", random_state=0).fit(E[tr], y[tr])
    return _logit(m.predict_proba(E[te])[:, 1], len(tr))[:, None]


# ---------- 8. PCA-MSPC ----------
@mem.cache
def mspc_ext(tr, te):
    """학습 정상으로 표준화·PCA(누적 분산 90%) → T²(주성분 공간 안 거리) + SPE(주성분 공간 밖 잔차).
    결합 지수 = T²/정상 중앙값 + SPE/정상 중앙값 → log10. 정상만 학습"""
    from sklearn.decomposition import PCA
    tr, te = np.array(tr), np.array(te)
    N = E[tr[y[tr] == 0]]
    sc = StandardScaler().fit(N); Zn = sc.transform(N)
    p = PCA(0.9, random_state=0).fit(Zn)

    def stat(Z):
        t = p.transform(Z)
        return (t ** 2 / p.explained_variance_).sum(1), ((Z - p.inverse_transform(t)) ** 2).sum(1)
    t2n, spen = stat(Zn)
    t2, spe = stat(sc.transform(E[te]))
    return np.log10(t2 / np.median(t2n) + spe / np.median(spen))[:, None]


# ---------- 정상만 학습하는 국소 이상 모델 (EXP-023: 회색 고장 014·017·022) ----------
def _rep(name, tr):
    """표현(150, d): 'mr' MiniRocket PPV(tr로 fit) · 'ar' AR 잔차 피처 5 · 'ext' ext150"""
    if name == "mr":
        return minirocket_feats(tuple(np.asarray(tr).tolist()))
    if name == "ar":
        return ar_feats(tuple(np.asarray(tr).tolist()))
    return E


@mem.cache
def oneclass(tr, te, rep="mr", model="knn"):
    """학습 정상만으로 표준화(+ 10차원 넘으면 PCA 누적 분산 95%) → 정상만 학습한 이상 점수 (te, 1). 클수록 이상
    knn: 가장 가까운 정상 5개까지 평균 거리 · lof: LocalOutlierFactor(novelty, k=10) · iforest: IsolationForest(300그루)
    다수 고장과 닮았는지가 아니라 '정상 무리에서 얼마나 떨어졌나'만 본다 → 유형이 다른 소수 고장용"""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    tr, te = np.array(tr), np.array(te)
    F = _rep(rep, tr)
    N = F[tr[y[tr] == 0]]
    sc = StandardScaler().fit(N)
    Zn, Zt = sc.transform(N), sc.transform(F[te])
    if Zn.shape[1] > 10:
        p = PCA(0.95, random_state=0).fit(Zn)
        Zn, Zt = p.transform(Zn), p.transform(Zt)
    if model == "knn":
        return NearestNeighbors(n_neighbors=5).fit(Zn).kneighbors(Zt)[0].mean(1)[:, None]
    if model == "lof":
        return -LocalOutlierFactor(n_neighbors=10, novelty=True).fit(Zn).score_samples(Zt)[:, None]
    return -IsolationForest(n_estimators=300, random_state=0).fit(Zn).score_samples(Zt)[:, None]


def oc(rep, model):
    """oneclass를 (tr, te) 전문가 함수로 (이름 유지)"""
    f = lambda tr, te: oneclass(tr, te, rep, model)
    f.__name__ = f"oc_{rep}_{model}"
    return f


# ---------- 9. CatBoost ----------
@mem.cache
def catboost_ext(tr, te):
    """ext150 → CatBoostClassifier(500회, 클래스 균형, 나머지 기본값) → logit 확률"""
    from catboost import CatBoostClassifier
    tr, te = np.array(tr), np.array(te)
    m = CatBoostClassifier(iterations=500, auto_class_weights="Balanced", random_seed=0, verbose=0,
                           thread_count=4, allow_writing_files=False).fit(E[tr], y[tr])
    return _logit(m.predict_proba(E[te])[:, 1], len(tr))[:, None]


# ---------- EXP-033 딥러닝 검토: 입력별 지도 신경망 (설정은 실행 전 고정, 튜닝 없음) ----------
@mem.cache
def mlp_psd(tr, te, epochs=300):
    """MLP (GPU). 입력 = AE와 같은 스펙트럼(200 Hz 고역통과, 1초 구간 8개 log10 Welch PSD 1025칸)의 파일 평균.
    학습 정상 기준 칸별 표준화 → 1025-128-32-1 (ReLU, dropout 0.2) → BCE(고장 가중 = 정상/고장 수), AdamW(1e-3, wd 1e-2), 전체 배치.
    AE는 정상만 복원(비지도)하고, 이 MLP는 같은 입력에서 고장/정상 경계를 지도 학습한다 → '입력은 같고 학습 방식만 다를 때' 비교"""
    from src.experts import LP
    torch, dev = _torch(); nn = torch.nn
    tr, te = np.array(tr), np.array(te)
    A = LP.mean(1)
    n = A[tr[y[tr] == 0]]
    A = ((A - n.mean(0)) / (n.std(0) + 1e-6)).astype(np.float32)
    net = nn.Sequential(nn.Linear(A.shape[1], 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 32), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(32, 1), nn.Flatten(0)).to(dev)
    Xt, yt = torch.tensor(A[tr], device=dev), torch.tensor(y[tr].astype(np.float32), device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        net.train(); loss = lossf(net(Xt), yt); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(A[te], device=dev)).cpu().numpy()[:, None]


@mem.cache
def cnn2d_full(tr, te, epochs=60):
    """2D-CNN (GPU). 입력 = 2초 전체 log STFT 한 장 (256칸 × 약 194프레임, 48.8 Hz × 10 ms). 구간으로 자르지 않는다.
    EXP-019 cnn2d_mil은 0.33초 구간 MIL이라 0.9초 간격 같은 '사건 사이 관계'를 볼 수 없었다(딥러닝_검토 2절 추정) → 이것을 바꾼 판.
    학습 정상 기준 주파수 칸별 표준화 → 합성곱 4층(stride 2, 256×194 → 16×13) → 주파수 평균 → 시간축 1D 합성곱(커널 5, 팽창 2,
    약 1.3초 시야) → 시간 최대 풀링 → 선형. 학습 때만 시간축 원형 이동(정상 상태 신호라 시작 시점은 임의) 증강"""
    torch, dev = _torch(); nn = torch.nn
    tr, te = np.array(tr), np.array(te)
    S = _stft_all()
    n = S[tr[y[tr] == 0]]
    mu, sd = n.mean((0, 2))[None, :, None], n.std((0, 2))[None, :, None] + 1e-6
    Q = torch.tensor(((S - mu) / sd)[:, None].astype(np.float32), device=dev)    # (150, 1, 256, T)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = _enc2d(torch)
            self.t = nn.Sequential(nn.Conv1d(32, 32, 5, dilation=2, padding=4), nn.BatchNorm1d(32), nn.ReLU())
            self.out = nn.Linear(32, 1)
        def forward(self, x):
            h = self.enc(x).mean(2)                       # (B, 32, T')
            return self.out(self.t(h).amax(2)).squeeze(1)

    net = Net().to(dev)
    yt = y[tr].astype(np.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((yt == 0).sum() / max(1, (yt == 1).sum()), device=dev))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // 16)):
            xb = torch.roll(Q[torch.tensor(tr[b], device=dev)], int(rng.integers(Q.shape[-1])), dims=-1)
            loss = lossf(net(xb), torch.tensor(yt[b], device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(Q[torch.tensor(te, device=dev)]).cpu().numpy()[:, None]
