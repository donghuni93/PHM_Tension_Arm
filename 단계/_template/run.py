# %% [markdown]
# EXP-nnn 제목
# - 가설: H? (단계/1_기준·기록/물리가설.md) — 이 실험이 확인하려는 물리 변화와 예상 신호 징후
# - 바꾼 것: 이전 최고 대비 무엇을 바꾸는가 (하나만)
# 사용: 단계/5_모델비교/스크리닝/EXP-nnn_이름/ 으로 폴더째 복사 → VS Code에서 셀 단위 실행(Shift+Enter), 또는
#   conda run -n qe-phm python 단계/5_모델비교/스크리닝/EXP-nnn_이름/run.py
# 결과가 나오면 폴더를 5_모델비교/유의미/ 또는 기각/ 으로 옮긴다 (루트는 위로 올라가며 src/를 찾으므로 깊이와 무관)
# 기록 후에는 이 파일을 고치지 않는다. 바꾸려면 새 EXP 폴더.

# %% 설정
import os, sys
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *

# member·model·changes = 노션 카드 속성 (팀원·모델·변경 사항). F2 점수는 main=True 구성에서 자동
ex = Experiment(__file__, "EXP-nnn", "제목", ["H?"],
                member="이동훈", model="ExtraTrees", changes="이전 대비 바꾼 것 한 줄")
X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X)
g = np.arange(len(y))

# %% 그림 1: 놓친 파일 vs 정상 5–95% (매 실험 기본)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_focus",
            "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")

# %% 그림 2: t-SNE (매 실험 기본). 이 실험이 쓰는 입력 공간마다 패널 하나. score= 를 주면 이상 점수 색 줄 추가
fig, _ = viz.tsne({"ext150 피처": E.values}, y, names, mark=ex.focus)
ex.save_fig(fig, "tsne", "입력 공간에서 고장이 정상과 따로 모이는가, 놓친 파일은 어디에 있는가")

# %% 그림 3: 핵심 그림 (key=True)
# README에서 해석 첫 문단(결론 요약) 바로 아래에 자동으로 놓인다. 그 아래에 '그림 읽는 법' → 상세 설명을 손으로 쓴다
# 결합 모델 실험은 logistic_plane이 필수 핵심 그림 (결정.md 기본 그림, 11-16):
#   AE z × ET z 평면(OOF seed 0) + 로지스틱 경계(0 실선, ±1 점선), 기준 | 새 구성 나란히, × = 자주 틀린 파일
#   규칙으로 잡힌 고장은 이 평면에서 뺀다(뺀 목록은 상자로 표시), 규칙이 켜진 정상은 남긴다. t-SNE에만 모든 파일
#   예시 코드: 단계/5_모델비교/유의미/EXP-029_1116_큰타격수·사건울림·주기잠금/figs_기본.py
# 그 밖의 기본 그림: hard_files(어려운 파일 위험도, 열 = 구성) · psd_missed(맨 끝)
# fig = ...; ex.save_fig(fig, "이름", "무엇을 보는 그림인지", key=True)

# %% 중첩 CV
et = make_et()
ex.record("기준 ext150", nested_cv(E.values, y, g, et), names, main=True)
