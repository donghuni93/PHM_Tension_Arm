"""충격 울림의 정밀 추정 (11-27, 5_VAL_단계 V-3에서 검증, V-4에서 사용)

한 모드 모델 x(t) = e^{−ζωt}(a sin ω_d t + b cos ω_d t). 진폭 · 위상(a, b)은 최소제곱으로 바로 풀고 f · ζ만 찾는다.
- est_grid: 제출 모델과 같은 사전(1–12 kHz 250 Hz × ζ 4칸, 8 ms). 처음 위치 찾기용 (ζ 자체는 오차 50–100 %)
- est_vp:   VP(변수 투영) 연속 최적화. 합성 검증 ζ 오차 1.5 % (잡음비 30) · 5 % (잡음비 10)
- est_mp:   Matrix Pencil. 그리드 없이 극점을 직접 계산, 2모드로 맥놀이 분리
- zeta_drift: 창 시작을 늦췄을 때 ζ가 얼마나 바뀌나 ln(ζ_late / ζ_early). 지수 감쇠 하나면 0
fit = 정규화 조각을 모드 기저에 투영한 크기 (0–1, 기존 impact_rings_v2의 fit과 같은 정의)
"""
import numpy as np
from scipy.optimize import least_squares

from src.final import _ring_atoms

FS = 25000
N8 = int(0.008 * FS)                    # 8 ms
OFF = int(0.0002 * FS)                  # 봉우리 + 0.2 ms (양방향 필터 앞울림 회피)
_A, _PAR = None, None
ZF = np.geomspace(0.003, 0.3, 40)


def basis(f, z, n=N8):
    """감쇠 사인파 한 모드의 2열 기저 (sin · cos)"""
    t = np.arange(n) / FS; w = 2 * np.pi * f
    a = np.exp(-z * w * t); wd = w * np.sqrt(max(1 - z * z, 1e-6))
    return np.column_stack([a * np.sin(wd * t), a * np.cos(wd * t)])


def proj_fit(B, s):
    Q, _ = np.linalg.qr(B)
    return float(np.linalg.norm(Q.T @ (s / np.linalg.norm(s))))


def est_grid(s):
    global _A, _PAR
    if _A is None:
        _A, _PAR = _ring_atoms()
    s = s[:N8] / np.linalg.norm(s[:N8]); fit = np.linalg.norm(np.einsum("s,ast->at", s, _A), axis=1); k = fit.argmax()
    return dict(f=_PAR[k, 0], z=_PAR[k, 1], fit=float(fit[k]))


def est_vp(s, n=None, f_init=None):
    """VP: 사전(또는 f_init) → ±300 Hz(20 Hz) × ζ 40칸 → (f, ln ζ) least_squares. n = 창 길이(점, 기본 8 ms)"""
    n = N8 if n is None else n
    f0 = est_grid(s[:N8])["f"] if f_init is None else f_init
    s = s[:n] / np.linalg.norm(s[:n])
    best = max(((f, z) for f in np.arange(max(600, f0 - 300), f0 + 301, 20) for z in ZF), key=lambda p: proj_fit(basis(*p, n), s))

    def res(p):
        B = basis(p[0], np.exp(p[1]), n); c = np.linalg.lstsq(B, s, rcond=None)[0]; return s - B @ c
    r = least_squares(res, [best[0], np.log(best[1])], bounds=([300, np.log(1e-3)], [12500, np.log(0.5)]), x_scale=[50, .1])
    f, z = r.x[0], float(np.exp(r.x[1]))
    return dict(f=f, z=z, fit=proj_fit(basis(f, z, n), s))


def est_mp(s, modes=2, n=None):
    """Matrix Pencil: 양의 주파수 모드 (f, ζ, 진폭) 큰 순. 2모드 fit = 두 모드 기저 투영"""
    n = N8 if n is None else n
    s = s[:n] / np.linalg.norm(s[:n]); L = n // 3; M = 2 * modes
    Y = np.lib.stride_tricks.sliding_window_view(s, L + 1)
    _, _, Vh = np.linalg.svd(Y, full_matrices=False); V = Vh[:M].T
    zs = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    zs = zs[(np.angle(zs) > 2 * np.pi * 300 / FS) & (np.abs(zs) > 1e-3)]
    if len(zs) == 0:
        return dict(f1=np.nan, z1=np.nan, f2=np.nan, z2=np.nan, ratio=np.nan, fit2=np.nan)
    lam = np.log(zs) * FS; f = np.abs(lam.imag) / (2 * np.pi); z = np.clip(-lam.real / np.abs(lam), 1e-4, 0.99)
    Vd = zs[None, :] ** np.arange(n)[:, None]
    amp = np.abs(np.linalg.lstsq(np.hstack([Vd, Vd.conj()]), s.astype(complex), rcond=None)[0][:len(zs)])
    o = np.argsort(-amp); f, z, amp = f[o], z[o], amp[o]
    B = np.hstack([basis(fi, zi, n) for fi, zi in zip(f[:modes], z[:modes])])
    return dict(f1=f[0], z1=z[0], f2=f[1] if len(f) > 1 else np.nan, z2=z[1] if len(f) > 1 else np.nan,
                ratio=amp[1] / amp[0] if len(f) > 1 else 0.0, fit2=proj_fit(B, s))


def zeta_drift(x, p, late_ms=1.2, early_ms=0.2, win_ms=8.0):
    """충격 봉우리 p(표본)에서 시작 early_ms · late_ms 두 창의 VP → dict(f, z_early, z_late, fit, drift = ln(z_late / z_early))
    late 창은 early 창의 주파수로 시작(같은 모드를 따라간다). 신호가 모자라면 None"""
    n = int(win_ms * FS / 1000); e0, l0 = p + int(early_ms * FS / 1000), p + int(late_ms * FS / 1000)
    if l0 + n > len(x):
        return None
    a = est_vp(x[e0:e0 + max(n, N8)], n=n)
    b = est_vp(x[l0:l0 + n], n=n, f_init=a["f"])
    return dict(f=a["f"], z_early=a["z"], z_late=b["z"], fit=a["fit"], fit_late=b["fit"], drift=float(np.log(b["z"] / a["z"])))


# ---- 뷰어 · 표용 ζ 신뢰 판정 (11-27, V-3 검증 결과를 규칙으로) ----
# 확실 = (1) VP 한 모드 fit ≥ 0.9 (V-3: fit ≥ 0.9에서만 ζ ±10–15 %)
#        (2) 독립 방법 Matrix Pencil(1모드)과 |ln 비| ≤ 0.15 (V-3 fit ≥ 0.9 중앙 0.06)
#        (3) 창 시작 +0.6 ms로 늦춰도 |ln 비| ≤ 0.20 (지수 감쇠 하나면 0)
#        (4) 5 kHz 이상에서 ζ ≥ 0.08이 아님 (τ ≈ 0.3 ms라 +0.2 ms 창으로는 잴 수 없음, V-3 현실 합성 −10 – −50 %)
# 기준은 모두 라벨 없이 V-3 합성 · 재현성 검증에서 정했다
SURE_FIT, SURE_MP, SURE_SHIFT = 0.90, 0.15, 0.20


def zeta_check(xh, p, f_grid=None):
    """1 kHz 고역 신호 xh, 봉우리 표본 p → dict(f, z, fit, z_mp, z_shift, d_mp, d_shift, sure, why). 창 = p + 0.2 ms부터 8 ms.
    신호가 모자라면 None"""
    p = int(p); s0, s1 = p + OFF, p + OFF + int(0.0004 * FS)          # +0.2 ms · +0.6 ms
    if s1 + N8 > len(xh):
        return None
    v = est_vp(xh[s0:s0 + N8], f_init=f_grid)
    m = est_mp(xh[s0:s0 + N8], modes=1)
    w = est_vp(xh[s1:s1 + N8], f_init=v["f"])
    d_mp = abs(np.log(m["z1"] / v["z"])) if np.isfinite(m["z1"]) else np.inf
    d_sh = abs(np.log(w["z"] / v["z"]))
    why = []
    if v["fit"] < SURE_FIT: why.append(f"fit {v['fit']:.2f} < {SURE_FIT}")
    if d_mp > SURE_MP: why.append(f"MP와 {np.exp(d_mp) - 1:.0%} 차이")
    if d_sh > SURE_SHIFT: why.append(f"시작 +0.6 ms에서 {np.exp(d_sh) - 1:.0%} 변함")
    if v["f"] >= 5000 and v["z"] >= 0.08: why.append("5 kHz 이상 · ζ ≥ 0.08 = 너무 빨리 죽어 잴 수 없음")
    return dict(f=float(v["f"]), z=float(v["z"]), fit=float(v["fit"]), z_mp=float(m["z1"]), z_shift=float(w["z"]),
                d_mp=float(d_mp), d_shift=float(d_sh), sure=not why, why=" · ".join(why))
