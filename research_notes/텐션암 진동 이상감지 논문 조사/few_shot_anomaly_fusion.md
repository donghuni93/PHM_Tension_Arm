# 고장 샘플이 적은 진동 이상감지 + 다중 검출기 점수 결합(Score Fusion)

> 범위: 팀 목록(Zhang 2020, Hoang & Kang 2019, Balaji 2023, Karthikeyan 2024, Shah 2024, Sai 2023, Jardine 2006, Antoni 2006/2007) 제외.
> 검증 상태 표기: **verified** = 서지(저자·연도·venue·DOI/arXiv)를 검색 결과의 1차/공식 페이지로 확인, **partially** = 서지는 확인했지만 본문 수치·세부식은 초록/요약 수준 또는 기억 기반, **unverified** = 확인 못 함.
> 우리 상황 요약: 전문가 A(ExtraTrees, ~150 피처, 지도) + 전문가 B(정상만 학습한 AE, 1025-bin log-PSD, 평균 재구성오차) → 각각 정상 OOF 대비 z-score → 2입력 L2 로지스틱(≈1:1). 문제 ① 선형결합이 "한 전문가만 강하게 본 고장"을 희석 ② max(OR)는 한쪽만 이상한 정상의 오경보 증가 ③ AE 평균오차는 원인 비특이적.

---

## Q1. 이상점수 정규화와 결합 규칙 (Kriegel 2011, Aggarwal & Sathe, Zimek, Fisher/Stouffer/Tippett, min-p, 스태킹)

### Takeaway
"평균 vs 최대"는 편향-분산 트레이드오프이며(Aggarwal & Sathe), 통계적으로는 **"한 출처만 강한 신호" 대안에는 Tippett(min-p)가, "여러 출처의 중간 신호" 대안에는 Fisher/Stouffer가 강하다**(Heard & Rubin-Delanchy). 따라서 각 전문가 점수를 **정상 OOF 기준 conformal p-value로 바꾼 뒤 min-p(Bonferroni 보정)와 Fisher를 같은 중첩 CV로 비교**하는 것이 우리 문제(희석 vs 오경보)에 정확히 대응하는, 학습 파라미터 0개짜리 처방이다.

### Cited Findings

**[1] Kriegel, H.-P., Kröger, P., Schubert, E., Zimek, A. (2011). Interpreting and Unifying Outlier Scores. Proc. SIAM SDM 2011, pp. 13–24. DOI: 10.1137/1.9781611972818.2** — verified(서지), partially(세부식)
- 서로 다른 검출기의 점수는 척도·범위·의미가 달라 비교/결합이 어렵고, 저자들은 점수를 [0,1]의 "이상 확률"로 바꾸는 정규화를 제안하며 이 통일이 outlier ensemble 성능을 높인다고 보고 — [researchr/TU Dortmund 요약](https://researchr.org/publication/KriegelKSZ11-0), [프리프린트 PDF](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2011/SDM11-outlier-preprint.pdf)
- (기억 기반, 본문 재확인 권장) 제안 방식 중 Gaussian scaling은 `max(0, erf((s−μ)/(σ√2)))` 형태로, **평균 이하(정상 쪽) 점수를 0으로 잘라** 정상 쪽 잡음이 결합에 끼어들지 않게 한다.
- 코드: ELKI(자바)에 구현되어 있다고 알려짐(미확인).
- **적용**: 지금은 z-score를 그대로 LR에 넣으므로 한 전문가의 **음수 z(정상보다 더 정상)가 다른 전문가의 양수 z를 상쇄**한다. 이것이 "희석"의 직접 원인 중 하나. z를 `max(0, ·)`로 자르거나 Gaussian scaling 후 결합하면 "한쪽만 강한 고장"이 덜 희석된다. 파라미터가 없어 과적합 위험이 낮다.

**[2] Aggarwal, C. C., Sathe, S. (2015). Theoretical Foundations and Algorithms for Outlier Ensembles. ACM SIGKDD Explorations 17(1): 24–47. DOI: 10.1145/2830544.2830549 (DOI는 집계 사이트 기준, ACM DL 재확인 권장)** — partially
- 분류의 편향-분산 분해를 이상탐지에 옮기고, subsampling·feature bagging 개선과 함께 **평균(averaging)과 최대(maximization) 결합 함수의 트레이드오프**를 분석 — [Bohrium 초록](https://www.bohrium.com/paper-details/theoretical-foundations-and-algorithms-for-outlier-ensembles/814590890851958785-18201), [dblp](https://dblp.dagstuhl.de/rec/journals/sigkdd/AggarwalS15.html)
- 단행본 확장: Aggarwal & Sathe, *Outlier Ensembles: An Introduction*, Springer 2017, DOI 10.1007/978-3-319-54765-7 — [Springer](https://link.springer.com/book/10.1007%2F978-3-319-54765-7)
- (기억 기반) 평균은 분산을 줄이고 max는 편향을 줄이지만 분산을 키운다는 해석, 그리고 그 절충으로 "평균 후 최대/최대 후 평균"(AOM/MOA) 같은 하이브리드가 논의됨.
- **적용**: 우리 관찰(평균=희석, max=오경보)은 이 논문이 말하는 **편향(평균) vs 분산(max)** 그 자체다. 전문가가 2개뿐이므로 AOM보다는 "max에 대한 분산 감소" 수단(예: 각 전문가 점수를 여러 seed/부트스트랩으로 평균한 뒤 max)이 실용적. 즉 **max의 오경보는 각 전문가 점수의 seed 잡음 때문일 수 있으니, 먼저 전문가 내부를 평균해 안정화한 뒤 전문가 간 max**를 시험.

**[3] Zimek, A., Campello, R. J. G. B., Sander, J. (2013). Ensembles for Unsupervised Outlier Detection: Challenges and Research Questions. ACM SIGKDD Explorations 15(1): 11–22.** — verified(서지, 연도는 2013; 일부 집계는 2014로 표기)
- 이상탐지 앙상블은 분할이 아니라 **순위(rank)를 결합**하는 문제라 클러스터링 앙상블과 다른 고유한 난제(정규화, 다양성, 결합)를 가진다는 position paper — [dblp](https://dblp.org/rec/journals/sigkdd/ZimekCS13), [JCU 저장소](https://researchonline.jcu.edu.au/46774/)
- **적용**: 결합 전 "정규화 → 다양성(두 전문가가 서로 다른 파일을 놓치는가) → 결합"을 분리해 점검하라는 체크리스트로 사용. 두 전문가의 놓친 고장 집합의 교집합/차집합을 표로 남기면 결합 이득의 상한을 미리 알 수 있다.

**[4] Heard, N. A., Rubin-Delanchy, P. (2018). Choosing between methods of combining p-values. Biometrika 105(1): 239–246. DOI: 10.1093/biomet/asx076. arXiv: 1707.06897** — verified
- Edgington·Fisher·George·Pearson·Stouffer·Tippett 6개 결합법을 각각 **우도비 검정으로 재해석**해, 어떤 대안가설에서 어느 결합법이 가장 강력한지 제시(Birnbaum 1954: 모든 합리적 결합법은 어떤 대안에서는 최적) — [arXiv](https://arxiv.org/abs/1707.06897), [Bristol 저장소](https://research-information.bris.ac.uk/en/publications/choosing-between-methods-of-combining-subipisub-values/)
- (기억 기반 해석, 본문 확인 권장) Tippett(min-p)은 신호가 소수 출처에 몰린 경우, Fisher는 일부가 강한 경우에도 비교적 견고, Stouffer는 신호가 고르게 약한 경우에 유리.
- **적용**: 우리 문제를 정확히 이론화한다. "한 전문가만 강하게 본 고장"이 주된 실패라면 **Tippett/min-p 계열**, 둘 다 약하게 보는 고장이 많다면 Stouffer(=z 합 ≈ 현재 LR)가 맞다. → 놓친 고장들이 어느 유형인지(한쪽만 vs 둘 다 약함) 세면 결합법이 정해진다. Fisher는 둘의 절충으로 첫 후보.

**[5] Vovk, V., Wang, R. (2020). Combining p-values via averaging. Biometrika 107(4): 791–808. DOI: 10.1093/biomet/asaa027. arXiv: 1212.4966** — verified(서지), partially(본문)
- 의존 구조를 전혀 가정하지 않고 p-value를 결합: 산술평균×2는 항상 유효한 p-value이며 2보다 작은 계수는 일반적으로 불가, 기하평균은 ×e, 조화평균은 ×약 log K — [Biometrika PDF(저자)](https://math.uwaterloo.ca/~wang/papers/2019Vovk-Wang-BIOM.pdf), [arXiv](https://arxiv.org/abs/1212.4966)
- **적용**: 두 전문가는 같은 파일을 보므로 p-value가 **독립이 아니다** → Fisher/Stouffer의 명목 오경보율은 깨진다(순위만 쓰면 무관하지만 임계값을 α로 정하면 문제). 오경보율을 이론적으로 보장하려면 **Bonferroni: 2·min(p_A, p_B) ≤ α** (OR 규칙의 정식 버전, 의존성 무관)를 쓴다. K=2라 손실이 작다.

**[6] Bates, S., Candès, E., Lei, L., Romano, Y., Sesia, M. (2023). Testing for outliers with conformal p-values. Annals of Statistics 51(1): 149–178. arXiv: 2104.08279. (DOI 10.1214/22-AOS2244는 집계 사이트 기준, 미확인)** — verified(arXiv·게재지), partially(DOI)
- 임의의 이상점수를 **정상 보정(calibration) 집합 대비 순위로 바꾼 conformal p-value**는 주변(marginal) 유효하고, 테스트 점들 사이에 양의 의존성이 있어 BH로 FDR 제어가 가능; 학습 데이터 조건부로 유효한 보정 p-value 방법도 제시 — [arXiv](https://arxiv.org/abs/2104.08279), [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.08279)
- **적용(핵심 레시피)**: 이미 갖고 있는 "정상 OOF 점수"가 보정 집합이다. 전문가 k의 p-value = `(1 + #{정상 OOF 점수 ≥ s_k}) / (n_정상 + 1)`. z-score 대신 이걸 쓰면 (i) 분포 가정 없음 (ii) 두 전문가 단위가 자동 통일 (iii) 결합 규칙(min-p/Fisher/Stouffer)을 바로 적용. 주의: 정상 OOF가 fold당 ~96개(120×4/5)라 **최소 p ≈ 1/97**로 해상도가 거칠다 → 극단 고장끼리 동점이 생김. 동점 해소용으로 z 크기를 보조 키로 쓰거나 Kriegel식 연속 정규화와 병행.

### Inferences
- 현재 2입력 L2-LR(≈1:1)은 사실상 **Stouffer(z 합)**에 가깝다. Heard & Rubin-Delanchy 관점에서 이는 "둘 다 약하게 보는 고장"에 최적화된 규칙이라 "한쪽만 강한 고장"에 약한 것이 이론적으로 예상된다.
- 파라미터 없는 대안 3개를 같은 중첩 CV로 비교하는 것이 과적합이 가장 적다: ① 2·min(p_A,p_B)(Bonferroni-OR) ② Fisher −2Σlog p ③ 음수 절단 z 합 `max(0,z_A)+max(0,z_B)`. ③은 Kriegel의 아이디어를 가장 단순화한 형태로, 정상 쪽 상쇄만 제거한다.
- "한쪽만 이상한 정상" 오경보는 conformal p의 해상도 한계(1/97)와도 관계: 정상 120개 중 각 전문가에서 상위 1–2개는 정의상 p≈0.01이므로, OR 결합은 **정상 오경보율 ≈ 2α**를 피할 수 없다. 이 비용이 F2에서 이득인지가 결정 기준.
- 스태킹(LR 가중 학습)은 양성 24개(30×4/5) 내부 fold로 가중치를 추정하므로 분산이 크다. 가중치가 ≈1:1로 나온 것 자체가 "고정 규칙으로 대체해도 된다"는 신호.

### Gaps
- Aggarwal & Sathe의 AOM/MOA·thresholded 결합의 정확한 정의와 실험 수치는 본문을 열어 확인하지 못함(기억 기반).
- Kriegel 2011의 Gaussian scaling 정확한 식은 프리프린트를 직접 열지 못해 기억 기반으로 표기.
- "few positives 스태킹"에 대한 전용 실증 논문은 이번 조사에서 찾지 못함.

---

## Q2. 고장 소수(30개) 환경의 준지도/few-shot: OCSVM, IForest, Deep SVDD, Deep SAD, DevNet, PU

### Takeaway
대규모 벤치마크(ADBench)는 **이상 라벨이 1%만 있어도 대부분의 준지도 방법이 최고의 비지도 방법을 앞선다**고 보고하며, 완전 지도 방법은 비슷한 성능에 약 10% 라벨이 필요했다. 우리는 고장 30/150(20%)이라 지도(ExtraTrees)가 강한 것이 자연스럽고, **미지 고장 대비용으로 정상-only 전문가를 유지하되, 라벨 고장을 "밀어내는" 준지도 점수(Deep SAD/DevNet 계열)를 세 번째 후보로** 시험할 가치가 있다. 다만 딥 모델은 150 파일로 과적합 위험이 크므로 얕은 형태로 줄여야 한다.

### Cited Findings

**[7] Han, S., Hu, X., Huang, H., Jiang, M., Zhao, Y. (2022). ADBench: Anomaly Detection Benchmark. NeurIPS 2022 Datasets & Benchmarks. arXiv: 2206.09426** — verified(서지), partially(수치)
- 30개 알고리즘(비지도 14·준지도 7·지도 9) × 57개 데이터셋 비교 — [arXiv](https://arxiv.org/abs/2206.09426v2)
- **이상 라벨 1%만으로도 대부분의 준지도 방법이 최고 비지도 방법을 능가**, 지도 방법은 비슷한 성능에 약 10% 라벨 필요, 비지도 중 통계적으로 압도적인 방법은 없음 — [NeurIPS 슬라이드](https://NeurIPS.cc/media/neurips-2022/Slides/55709.pdf), [arXiv](https://arxiv.org/abs/2206.09426v2)
- 코드: GitHub `Minqi824/ADBench`(공식으로 알려짐, 미확인), PyOD 기반 구현 다수.
- **적용**: 비지도 방법 간 우열이 없다는 결과 → AE를 IForest/OCSVM/kNN으로 바꾼다고 큰 이득을 기대하기 어렵다. 대신 "라벨 고장 30개를 정상-only 전문가에도 약하게 주입"하는 준지도 방향이 근거가 있다.

**[8] Ruff, L., Vandermeulen, R. A., Görnitz, N., Binder, A., Müller, E., Müller, K.-R., Kloft, M. (2020). Deep Semi-Supervised Anomaly Detection (Deep SAD). ICLR 2020. arXiv: 1906.02694** — verified
- 기존 준지도 이상탐지는 대부분 라벨된 정상만 쓰고 라벨된 이상은 거의 쓰지 않는다는 문제의식에서, Deep SVDD를 일반화해 **라벨 이상은 중심에서 멀어지도록(역거리 항)** 학습하는 end-to-end 방법; 소량 라벨에서도 성능 향상 보고 — [arXiv](https://arxiv.org/abs/1906.02694v2), [ar5iv](https://ar5iv.labs.arxiv.org/html/1906.02694)
- 선행: Ruff et al. (2018) Deep One-Class Classification (Deep SVDD), ICML 2018, PMLR 80 — partially(기억 기반 서지)
- 코드: 저자 GitHub `lukasruff/Deep-SAD-PyTorch`(알려진 저장소, 미확인)
- **적용**: 우리 AE 잠재공간 또는 band-pooled log-PSD에 Deep SAD 손실을 적용하면 "정상 구 바깥 + 알려진 고장은 더 바깥"이 되어 **정상-only 전문가의 미지 고장 탐지력은 유지하면서 알려진 고장 방향을 강화**. 과적합 방지: 작은 MLP(은닉 1층), 라벨 고장 가중치 η를 작게, 반드시 중첩 CV 내부에서만 라벨 사용. 주의: 라벨 고장이 이질적이라 "중심에서 멀리"라는 단일 목적이 다른 유형 간 간섭은 적다(방향을 강제하지 않음) — 이 점이 이질 고장에 유리.

**[9] Pang, G., Shen, C., van den Hengel, A. (2019). Deep Anomaly Detection with Deviation Networks (DevNet). KDD 2019, pp. 353–362. DOI: 10.1145/3292500.3330871. arXiv: 1911.08623** — verified
- 전체 학습 데이터의 0.005%–1%에 불과한 소수 라벨 이상으로 이상점수를 end-to-end 학습. 가우시안 사전분포 + **Z-score 기반 deviation loss**: 정상 점수는 기준값 근처로, 라벨 이상 점수는 기준에서 크게(예: z>5) 벗어나도록 → 점수가 해석 가능 — [arXiv](https://arxiv.org/abs/1911.08623), [Adelaide 저장소](https://eres.library.adelaide.edu.au/dspace/handle/2440/122748)
- 코드: 저자 GitHub `GuansongPang/deviation-network`(알려진 저장소, 미확인)
- **적용**: DevNet의 출력은 정의상 "정상 대비 z-score"라 **우리 결합 틀(정상 OOF z)과 단위가 같다**. 즉 전문가 A·B의 피처(또는 두 점수 + band별 AE 오차 몇 개)를 입력으로 하는 작은 DevNet은 "학습된 결합기"이면서 출력 척도가 고정되어 임계값 해석이 쉽다. 단, 파라미터가 늘어나는 만큼 고정 규칙(Q1)보다 우선순위는 낮음.

### Inferences
- 우리 비율(고장 20%)은 ADBench의 "지도 학습이 따라잡는 구간(~10%)"을 넘으므로, 알려진 고장에 대해서는 ExtraTrees가 준지도보다 나을 가능성이 높다. 준지도/정상-only 전문가의 존재 이유는 **미지 고장 대비**이며, 평가도 "고장 유형 하나를 통째로 빼는 leave-one-fault-type-out" 방식으로 보완해야 그 가치가 드러난다(고장 유형 라벨이 없으면 클러스터로 대체).
- Positive-Unlabelled(PU) 설정은 val/test가 라벨 없는 혼합이라 형식상 맞지만, 40개 파일로 transductive 학습을 하면 누수·과적합 위험이 커서 이번 과제에는 비권장(추론).

### Gaps
- 150개 규모 진동 데이터에서 Deep SAD/DevNet을 쓴 실증 사례는 찾지 못함. 벤치마크는 대부분 표 형식·이미지 데이터.
- PU 학습의 진동 응용 논문은 이번 조사 범위(15회 호출)에서 확인하지 못함.

---

## Q3. 스펙트럼 오토인코더: 대역별 재구성오차, 오차 정규화, 집계(max vs mean)

### Takeaway
DCASE 계열 기준선은 **재구성오차 벡터 자체를 정상 오차의 공분산으로 정규화(Mahalanobis)** 하는 쪽으로 진화했다(평균 MSE → Selective Mahalanobis). 우리 AE에 그대로 옮기면 "어느 주파수에서 얼마나 정상보다 큰가"가 반영되어 평균오차의 비특이성(문제 ③)이 줄어든다. 1025 bin은 정상 120개로 공분산 추정이 불가하므로 **대역 풀링(20–40대역) + 축소(shrinkage) 공분산**이 필요하다.

### Cited Findings

**[10] Harada, N., Niizumi, D., Ohishi, Y., Takeuchi, D., Yasuda, M. (2023). First-shot anomaly sound detection for machine condition monitoring: A domain generalization baseline. arXiv: 2303.00455 (EUSIPCO 2023 게재로 보이는 PDF 존재)** — verified(arXiv), partially(게재지)
- DCASE 2023 Task 2 기준선. **MSE 모드**: 점수 = ‖x − x̂‖ 평균. **Selective Mahalanobis 모드**: 재구성 차이 벡터(x − x̂)에 대해 source 공분산·target 공분산 각각으로 Mahalanobis 거리를 계산하고 **둘 중 최솟값**을 점수로 사용. 공분산은 마지막 epoch 후 정상 학습 샘플의 재구성 차이로 도메인별 추정 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2303.00455), [arXiv](https://arxiv.org/abs/2303.00455)
- 3 seed 평균 조화평균 총점 0.5903(Selective Mahalanobis) vs 0.5599(MSE), 산술평균 0.6300 vs 0.5880 — 소폭 개선 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2303.00455)
- "first-shot" 조건: 기계별 하이퍼파라미터 튜닝과 정답 기반 앙상블 선택을 금지 — [arXiv](https://arxiv.org/abs/2303.00455)
- 코드: https://github.com/nttcslab/dcase2023_task2_baseline_ae (논문 내 링크로 확인)
- **적용**: (a) 1025 bin을 로그 간격 32대역 정도로 풀링한 재구성오차 벡터 e(32차원)를 만들고, 정상 OOF의 e로 Ledoit-Wolf 공분산을 추정해 Mahalanobis 점수 산출. (b) 진단용으로 **대역별 표준화 오차 z_b = (e_b − μ_b)/σ_b** 와 그 max·argmax를 기록 → "어느 대역 때문에 높은가"가 남아 오류 분석·물리가설(H…) 연결이 쉬워짐. (c) "selective min"은 도메인(작업조건·게인) 군집이 여러 개일 때 가장 가까운 정상 조건 기준으로 판단하는 장치 → 정상 파일이 조건별로 군집되어 있다면 군집별 공분산의 min을 쓰는 변형도 가능. "first-shot" 원칙은 우리 원칙 3(라벨 보고 대역 고르지 않기)과 같은 정신.

**[11] Suefusa, K., Nishida, T., Purohit, H., Tanabe, R., Endo, T., Kawaguchi, Y. (2020). Anomalous sound detection based on interpolation deep neural network (IDNN). ICASSP 2020. arXiv: 2005.09234** — verified(서지), partially(저자 순서: 출처마다 다름)
- 비정상(non-stationary) 기계음에서는 이상 여부와 무관하게 AE 재구성오차가 커지는 문제(특히 가장자리 프레임 예측이 어려움)를 지적하고, **가운데 프레임을 빼고 주변 프레임으로 보간 예측**하는 방식으로 표준 AUC 27% 개선, 특히 비정상음에서 효과 — [arXiv](https://arxiv.org/abs/2005.09234), [DeepAI](https://deepai.org/publication/anomalous-sound-detection-based-on-interpolation-deep-neural-network)
- **적용**: 텐션암 신호가 충격성(비정상)이라면 세그먼트별 log-PSD의 평균 오차는 충격이 우연히 많은 정상 세그먼트에서도 커진다 → 오경보 원인 후보. 우리는 세그먼트 PSD를 쓰므로, 시간축 보간 대신 **세그먼트 오차의 분위수(예: 상위 10%)·max vs 평균**을 비교하는 것이 같은 문제의 저비용 점검이다.

### Inferences
- 평균오차는 "많은 bin의 작은 차이"와 "몇 bin의 큰 차이"를 구분하지 못한다(문제 ③). 대역별 표준화 후 (i) 평균(≈ 분산형 고장) (ii) max(≈ 국소 대역 고장) (iii) Mahalanobis(상관 고려) 세 집계를 **각각 별도 점수로 conformal p-value화**해 Q1의 결합 틀에 넣으면, AE 하나가 "서로 다른 이유"를 구분하는 다중 전문가가 된다. 단, 점수 수가 늘면 OR(min-p)의 오경보가 늘어나므로 Bonferroni 계수 K도 같이 늘어난다.
- 대역별 오차의 max를 쓸 때 대역 수가 많을수록 정상의 max도 커지므로(극값 효과) **반드시 대역별 표준화 후 max**를 취해야 한다.

### Gaps
- DCASE 2022 Task 2 설명 논문(Dohi et al., arXiv 2206.05876)의 기준선 세부는 직접 확인하지 못함(여러 기술보고서가 이 arXiv 번호로 인용하는 것만 확인 — [Wei et al. 보고서](https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Wei_22_t2.pdf)).
- 주파수 bin별 오차 정규화를 진동 PSD AE에 직접 적용한 MSSP/TII 논문은 이번 호출 예산 안에서 찾지 못함.

---

## Q4. 산업 음향/진동 이상탐지 벤치마크(DCASE Task 2, MIMII, ToyADMOS): 우승 방법과 도메인/게인 이동 교훈

### Takeaway
DCASE Task 2는 "정상만 학습, 미지 이상 탐지"라는 우리 전문가 B의 문제와 같은 설정이며, 최근 흐름은 **임베딩 + kNN 거리 점수 + 도메인별 점수 정규화**(z-score, 국소 밀도 정규화 LDN)다. 게인·조건 이동이 있을 때 **점수를 조건(도메인)별로 정규화**하는 것이 반복적으로 효과를 보였다.

### Cited Findings

**[12] Koizumi, Y., Kawaguchi, Y., Imoto, K., Nakamura, T., Nikaido, Y., Tanabe, R., Purohit, H., Suefusa, K., Endo, T., Yasuda, M., Harada, N. (2020). Description and Discussion on DCASE2020 Challenge Task2: Unsupervised Anomalous Sound Detection for Machine Condition Monitoring. DCASE 2020 Workshop. arXiv: 2006.05822** — verified(arXiv), partially(저자 목록은 기억 기반)
- 정상 샘플만 학습에 제공되어 **들어본 적 없는 이상을 탐지**해야 하는 과제; ToyADMOS·MIMII 일부(toy car, toy conveyor, valve, pump, fan, slide rail) 사용; AUC·pAUC 평가; 40팀 117개 제출 — [arXiv](https://arxiv.org/abs/2006.05822)
- 데이터 원전: ToyADMOS (Koizumi et al., WASPAA 2019, arXiv 1908.03299), MIMII (Purohit et al., DCASE 2019 Workshop, arXiv 1909.09347) — partially(기억 기반 ID)
- **적용**: pAUC(저 FPR 구간 AUC)는 "오경보를 적게 유지하면서 얼마나 잡는가"를 보는 지표로, 우리 OR 결합의 오경보 문제를 평가할 보조 지표로 쓸 만하다(예: FPR≤10% 구간 pAUC).

**[13] Wilkinghoff, K. 외 (2025). Local Density-Based Anomaly Score Normalization for Domain Generalization. arXiv: 2509.10951** — partially(검색 스니펫 기반, 저자 전체 목록 미확인)
- 거리 기반 이상음 탐지에서 **각 참조 샘플의 국소 밀도로 kNN 거리 점수를 정규화(LDN)** 하면 도메인 이동 하에서 여러 임베딩 기반 시스템이 개선되고 기존 점수 정규화법보다 낫다고 보고 — [arXiv HTML](https://arxiv.org/html/2509.10951v1)
- 후속 "Mind the Gap"(Wilkinghoff, Wichern, Le Roux, Tan)은 LDN이 매우 작은 이웃 크기에서만 쓰이며 K를 키우면 국소성 가정이 깨질 수 있음을 다룸 — [arXiv PDF 2602.18777](https://arxiv.org/pdf/2602.18777) (partially)
- 실제 시스템 사례: GenRep(Saengthong & Shinozaki) — 고정 사전학습 임베딩 + kNN 점수, DCASE 2025에서 도메인별 z-score/LDN/도메인별 LDN 비교 — [DCASE2025 기술보고서](https://dcase.community/documents/challenge2025/technical_reports/DCASE2025_Saengthong_18_t2.pdf), [arXiv 2409.05035](https://arxiv.org/html/2409.05035v1)
- **적용**: 정상 파일들이 게인/하중 조건별로 군집되어 있다면, 전역 μ·σ로 z-score를 내면 "밀도가 낮은 정상 군집"의 파일이 오경보가 된다(문제 ②의 후보 원인). **kNN 거리 ÷ 해당 이웃의 kNN 거리(LDN 아이디어)** 는 파라미터가 K 하나뿐이라 우리 규모에서도 시도 가능. K는 1–3 정도로 작게.

### Inferences
- DCASE 결과는 "점수 결합 규칙"보다 **"점수 정규화를 정상 분포의 국소 구조에 맞추는 것"**이 도메인 이동에서 더 큰 이득을 준다는 방향을 시사. 우리 val/test에 조건 이동(장착·게인 차)이 있다면, 결합 전에 각 전문가 점수를 정상 국소 밀도로 정규화하는 것이 우선일 수 있다.
- val 40개가 공개되면 **라벨 없이도** val 점수 분포가 학습 정상 OOF 분포와 같은지(예: AE 점수 중앙값 이동)를 보는 것은 정당한 도메인 이동 점검이다(라벨을 쓰지 않으므로 누수 아님).

### Gaps
- DCASE 2020–2025 상위 팀이 **이종 검출기 점수를 어떤 규칙으로 앙상블했는지**(평균/순위평균/가중합) 정리된 1차 자료는 이번 호출 예산 안에서 확보하지 못함. 기술보고서 개별 열람 필요.
- 진동(가속도) 데이터 전용 few-shot 벤치마크는 찾지 못함.

---

## Q5. 소규모 데이터 평가 함정: 중첩 CV, F-β 임계값, 작은 검증셋 분산, 누수

### Takeaway
모델 선택 기준의 **분산** 자체가 선택 과적합을 만들며, 그 손실은 알고리즘 간 차이와 같은 크기일 수 있다(Cawley & Talbot). 중첩 CV는 이 편향을 크게 줄인다(Varma & Simon). 따라서 결합 규칙 후보는 **소수(3–4개)로 미리 고정**하고 같은 중첩 CV로 비교해야 하며, F2 임계값은 내부 fold에서 정해야 한다.

### Cited Findings

**[14] Cawley, G. C., Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR 11(70): 2079–2107.** — verified
- 모델 선택 기준의 분산이 편향만큼 중요하며, 분산이 크면 선택 단계에서 과적합; 그 성능 손실은 학습 알고리즘 간 성능 차이와 비슷한 크기일 수 있고, 흔한 평가 관행이 이 선택 편향으로 신뢰할 수 없게 됨 — [JMLR](https://www.jmlr.org/papers/v11/cawley10a.html)
- **적용**: 결합 규칙·정규화·집계 조합을 여러 개 시험하면 그 자체가 하이퍼파라미터 탐색이다. 중첩 CV 외곽 점수에서 "최고"를 고르면 다시 선택 편향이 생기므로, **비교 후보 수를 사전에 제한하고 seed sd를 함께 보고, 차이가 sd 이내면 단순한 쪽**을 택한다.

**[15] Varma, S., Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 7:91. DOI: 10.1186/1471-2105-7-91** — verified
- CV 오차를 최소화하도록 파라미터를 고른 뒤 그 CV 오차를 보고하면 독립 테스트 오차를 낙관적으로 편향 추정; null 데이터 시뮬레이션으로 입증, **중첩 CV는 편향을 크게 줄여 독립 테스트셋과 비슷한 추정**을 줌(SVM·Shrunken Centroids) — [PMC 전문](https://pmc.ncbi.nlm.nih.gov/articles/PMC1397873)
- **적용**: 현재 5×4 중첩 CV 설계의 정당성 근거. 결합기(LR)와 정규화 기준(정상 OOF μ·σ 또는 conformal 보정집합)도 **외곽 fold의 학습부에서만** 계산해야 한다(정상 OOF 통계를 전체 120개로 한 번에 구하면 미세 누수).

**[16] Lipton, Z. C., Elkan, C., Narayanaswamy, B. (2014). Optimal Thresholding of Classifiers to Maximize F1 Measure. ECML PKDD 2014, LNCS 8725, pp. 225–239. DOI: 10.1007/978-3-662-44851-9_15. arXiv: 1402.1892 (arXiv 제목 "Thresholding Classifiers to Maximize F1 Score")** — verified
- 실수 출력 분류기에서 최적 F1과 그 임계값의 관계를 유도; **보정된 확률이면 F1 최적 임계값 = 최적 F1의 절반**; 정보 없는 분류기의 F1 최적 행동은 "모두 양성" — [arXiv](https://arxiv.org/abs/1402.1892), [Springer PDF](https://link.springer.com/content/pdf/10.1007/978-3-662-44851-9_15.pdf)
- **적용**: F1 결과를 F-β로 일반화하면(추론, 아래) 최적 임계값 ≈ F*_β/(1+β²). F2에서 F*≈0.8이면 보정 확률 임계값 ≈ 0.16 → **F2 최적 임계값은 낮고, 분류기가 약할수록 "전부 고장" 쪽으로 쏠린다.** 이 관계는 prevalence(고장 비율)에 의존하므로 val/test 고장 비율이 학습(20%)과 다르면 학습에서 고른 확률 임계값이 어긋난다. → 확률 임계값 대신 **정상 기준 분위수(예: 정상 OOF 점수의 상위 x%)로 임계값을 정의**하면 prevalence 변화에 덜 민감(정상 분포만 쓰므로).

### Inferences
- F-β 일반화(Lipton 식의 확장)는 본 조사에서 원문 확인을 못 했으므로 추론으로 표기: Fβ = (1+β²)TP / ((1+β²)TP + β²FN + FP) 에서 같은 논리를 적용하면 임계값 = F*/(1+β²).
- 작은 검증셋 분산(추론, 이항 근사): val 40개 중 고장이 10개라면 고장 1개를 놓칠 때마다 재현율이 10%p 변한다. 따라서 val 점수 한 번으로 결합 규칙을 바꾸는 것은 Cawley & Talbot이 경고한 선택 과적합에 해당. val은 "최종 고정 모델의 확인용"으로만 쓰는 것이 원칙과 일치.
- 누수 체크리스트: (i) 정상 OOF 기준 통계·conformal 보정집합을 외곽 fold 안에서만 계산 (ii) AE 학습 정상과 보정 정상은 분리(같은 파일이면 점수가 낙관적으로 작아져 p-value가 과소평가 → 오경보 증가) (iii) 대역 풀링 경계는 라벨 없이 고정(로그 간격 등).

### Gaps
- F-β 최적 임계값의 원 논문 수준 유도(β≠1)는 확인 못 함.
- prevalence 이동 하에서 F-β 임계값 보정에 관한 전용 논문은 이번 예산 안에서 찾지 못함.

---

## 부록: "평균은 희석, OR은 오경보" 문제에 대한 저과적합 실험 순서(근거 연결)

1. **진단**: 놓친 고장을 (a) 한 전문가만 강함 (b) 둘 다 약함으로 분류, 오경보 정상을 (c) 한 전문가만 높음으로 분류 → 결합 규칙 선택의 근거 [3][4].
2. **정규화 교체(파라미터 0)**: z → conformal p-value(정상 OOF 보정집합, 외곽 fold 내부) [6][15].
3. **결합 후보 4개 고정 후 중첩 CV 비교**: 현재 LR(≈Stouffer) / 음수 절단 z 합 [1] / Fisher [4] / Bonferroni-OR 2·min p [5]. 추가 후보는 만들지 않음 [14].
4. **AE 비특이성**: 대역 풀링 오차 벡터 → 대역별 표준화 max + Ledoit-Wolf Mahalanobis를 별도 점수로 [10][11].
5. **조건 이동 대비**: 정상 국소 밀도 정규화(LDN, K 작게) [13], val 라벨 없는 분포 비교.
6. (선택) 정상-only 전문가에 라벨 고장 약하게 주입: Deep SAD / DevNet 소형 [7][8][9].
7. 임계값: 정상 OOF 분위수 기반으로 정의해 prevalence 이동에 강하게 [16].

### 코드 요약
| 항목 | 코드 | 상태 |
|---|---|---|
| DCASE2023 AE 기준선(Selective Mahalanobis) | github.com/nttcslab/dcase2023_task2_baseline_ae | 논문 내 링크로 확인 |
| Deep SAD | github.com/lukasruff/Deep-SAD-PyTorch | 미확인(기억) |
| DevNet | github.com/GuansongPang/deviation-network | 미확인(기억) |
| ADBench | github.com/Minqi824/ADBench | 미확인(기억) |
| Kriegel 정규화 | ELKI | 미확인(기억) |
| conformal p / Ledoit-Wolf | 직접 구현 수 줄 / `sklearn.covariance.LedoitWolf` | 표준 |
