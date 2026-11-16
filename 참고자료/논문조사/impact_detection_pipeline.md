# 비주기 간헐 충격(clunk·rattle·looseness) 검출 신호처리 파이프라인 — SK/kurtogram 이후 문헌

> 조사일 2026-11-14. 웹 검색 17회. 검증 상태 표기: **verified** = 서지·DOI를 검색 결과에서 확인, **partially** = 서지는 확인했지만 DOI나 세부 내용은 확인하지 못함, **unverified** = 기억에 의존했거나 2차 출처만 있음.
> 팀 목록(Antoni 2006·2007, Jardine 2006, Zhang 2020, Hoang & Kang 2019, Balaji 2023, Karthikeyan 2024, Andreaus 2007, Shah 2024, Sai 2023)은 제외했다.
> 대부분 원문 PDF는 읽지 못했고 초록·서지 레코드·2차 서술을 근거로 정리했다. 수식·성능 수치를 인용하려면 원문을 확인해야 한다.

### 요약 표

| # | 문헌 | DOI / URL | 상태 | OA·코드 |
|---|---|---|---|---|
| 1 | Antoni 2016, Infogram, MSSP 74:73–94 | 10.1016/j.ymssp.2015.04.034 | verified | HAL 사본 있음(hal-01285393, 봇 차단으로 열람 실패) |
| 2 | Moshrefzadeh & Fasana 2018, Autogram, MSSP 105:294–318 | 10.1016/j.ymssp.2017.12.009 | verified | MATLAB FileExchange 65413, PoliTo IRIS 사본 |
| 3 | Barszcz & Jabłoński, Protrugram, MSSP (2011 추정) | Scribd 사본만 확인 | partially | — |
| 4 | Tse & Wang 2013, Sparsogram Part 1, MSSP 40(2):520–544 | DOI 확인 못 함 | partially | — |
| 5 | Miao, Zhao, Lin 2017, Gini index로 kurtosis 계열 gram 개선, Meas. Sci. Technol. 28(12) | DOI 확인 못 함 | partially | Gini 관련 MATLAB FileExchange 97572 |
| 6 | Wang 2018, 스펙트럼 kurtosis·L2/L1·smoothness·Gini 비교, MSSP | DOI 확인 못 함 | partially | — |
| 7 | Wang 외 2020, 가중 정규화 squared envelope의 합(SWNSE), MSSP | 10.1016/j.ymssp.2020.106725 | partially(DOI는 검색 결과로 확인, 서지 세부는 미확인) | — |
| 8 | Endo & Randall 2007, AR+MED, MSSP 21(2):906–919 | 10.1016/j.ymssp.2006.02.005 | verified | — |
| 9 | Sawalhi, Randall, Endo 2007, MED+SK, MSSP | 10.1016/j.ymssp.2006.12.002 | partially | — |
| 10 | McDonald, Zhao, Zuo 2012, MCKD, MSSP 33:237–255 | DOI 확인 못 함 | partially | MATLAB FileExchange 31326 |
| 11 | Crowther, Janello, Singh 2007, 클리어런스 충격원 정량화, JSV 307:428–451 | 10.1016/j.jsv.2007.05.055 | verified | — |
| 12 | Oh & Singh 2005, clunk 비선형 비틀림 모델과 지표, SAE 2005-01-2291 | SAE 페이지 | partially | 유료 |
| 13 | Lee, Kim, Kim 2015, squeak·rattle 자동 검출·구분, SAE 2015-01-2258 | SAE 페이지 | partially | 유료 |
| 14 | Bayani, Wickman, Söderberg, 덜컹임(rattle) 불쾌감 지표, Chalmers | research.chalmers.se/publication/525790 | partially | Chalmers 사본 가능성 |
| 15 | 현가장치(볼조인트·부싱 마모) 진동 + 전이학습(radar plot), Machines 2023 | munin.uit.no/handle 10037/31787 | partially | OA |

---

## Q1. 과도 신호용 대역 선택법(protrugram, sparsogram, infogram, autogram, spectral negentropy, Gini) 중 비주기 충격에 맞는 것은?

### Takeaway
비주기·불규칙 충격에는 **포락선 스펙트럼(SES) 쪽 지표**(protrugram, sparsogram, infogram의 SES 버전, autogram)가 근거가 약하다. 이들은 결함 주파수 같은 반복성을 전제한다. **시간 영역 지표**(SK, infogram의 SE 버전 = squared envelope의 spectral negentropy, Gini index)가 맞는다. 다만 Wang(2018)에 따르면 이 지표들은 모두 최대 squared envelope에 단조 증가한다. 그래서 "정상 파일의 큰 충격 하나"에 끌려가는 우리 오경보 문제를 그대로 안고 있다.

### Cited Findings
- **Infogram (Antoni 2016, verified, DOI 10.1016/j.ymssp.2015.04.034)**: 반복 과도를 "평형 상태로부터의 이탈"로 보고 squared envelope(SE)와 squared envelope spectrum(SES)의 negentropy를 각각 계산해 kurtogram을 확장했다. 두 infogram을 함께 쓰면 임펄스성 잡음이 있거나 과도가 반복 간격에 비해 빨리 감쇠할 때 kurtogram보다 낫다고 보고한다. MSSP 74:73–94. — [peeref 레코드](https://peeref.com/works/2818032), [BL 카탈로그](https://eld.bl.uk/catalog/100031025006_0x000001), [HAL](https://hal.archives-ouvertes.fr/hal-01285393)
  - **적용**: 1.25 kHz 고정 격자 대신 1/3-이진 트리(fast kurtogram 구조)로 SE-negentropy("time-domain infogram")를 계산해 파일별 최대 정보 대역과 그 값을 피처로 쓴다. SES 버전은 우리 데이터에서 주기성이 없으므로 대조군으로만 쓴다. **라벨을 보고 대역을 고르는 것이 아니라** 파일마다 비지도로 고르므로 CLAUDE.md 원칙 3에 맞는다.
- **Autogram (Moshrefzadeh & Fasana 2018, verified, DOI 10.1016/j.ymssp.2017.12.009)**: 필터링한 신호의 kurtosis 대신 squared envelope의 불편(unbiased) 자기상관의 kurtosis를 쓴다. 근거는 2차 사이클로정상 신호의 자기공분산이 주기적이라는 점이다. Upper/Lower Autogram과 Combined SES 지표를 제안했다. 대상은 구름베어링이다. MSSP 105:294–318. MATLAB 코드가 공개돼 있다. — [polito IRIS](https://iris.polito.it/handle/11583/2696265), [MathWorks FileExchange 65413](https://www.mathworks.com/matlabcentral/fileexchange/65413-autogram)
  - **적용**: 자기상관의 주기성에 기대므로 불규칙 충격에는 이론적으로 불리하다. 쓸 경우 "충격 간격 규칙성 피처"로만 의미가 있다(정상과 고장의 반복성 차이를 검정할 때).
- **Protrugram (Barszcz & Jabłoński, partially)**: 시간 신호의 kurtosis가 아니라 복조 신호의 포락선 스펙트럼 진폭의 kurtosis로 대역을 고른다. 동기는 과도 반복률이 높아지면 시간 영역 kurtosis가 떨어진다는 점이다. 2차 문헌은 결함 주파수에 대한 사전 지식이 필요하고(blind가 아님), 결함 톤이 약하면 실패한다고 지적한다. — [Scribd 사본 "A Novel Method for the Optimal Band Selection for Vibration Signal Demodulation and Comparison With the Kurtogram"](https://es.scribd.com/document/274252167/2010-A-Novel-Method-for-the-Optimal-Band-Selection-for-Vibration-Signal-Demodulation-and-Comparison-With-the-Kurtogram-Tomasz-Barszcz), [Barszcz Diagnostyka 2010 PDF](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-article-BAR0-0051-0049/c/httpwww_bg_utp_edu_plartdiagnostyka2010barszcz.pdf)
  - **적용**: 회전 주파수가 없는 우리 문제에는 **부적합**하다. 보고서의 "검토했으나 제외" 근거로 쓴다.
- **Sparsogram (Tse & Wang 2013, partially)**: 여러 분해 깊이에서 wavelet packet 계수의 포락선 파워스펙트럼 희소도를 계산하고, 희소도가 가장 큰 노드를 공진 대역으로 고른다. 베어링 시뮬레이션과 실측에서 기존 WP 방법 3개보다 낫다고 보고했다. MSSP 40(2):520–544(2013). 질문의 "2010"과 연도가 다르다. 출처들은 2013으로 일치한다. — [peeref 13365982](https://peeref.com/zh/works/13365982), [Springer 리뷰 s40430-021-03117-y](https://link.springer.com/article/10.1007/s40430-021-03117-y)
  - **적용**: 포락선 *스펙트럼* 희소도이므로 비주기 충격에는 약하다. 대신 WPT 분해 구조(1.25 kHz 고정 격자를 대체하는 이진 대역 분할)는 그대로 차용할 수 있다.
- **Gini index (Miao, Zhao, Lin 2017, partially)**: kurtosis가 임펄스성 잡음에 취약하다는 점을 들어 공진 대역 선택 지표로 Gini index를 제안했다. 고정축 베어링과 유성기어 베어링 신호로 검증했다. Meas. Sci. Technol. 28(12), 2017-11-15. 이후 MSSP 2021 "Practical framework of Gini index…"(Miao 외)에서 랜덤 임펄스 잡음과 디컨볼루션으로 확장했다. — [XJTU 레코드](https://scholar.xjtu.edu.cn/en/publications/improvement-of-kurtosis-guided-grams-via-gini-index-for-bearing-f/), [peeref 23724795](https://www.peeref.com/works/23724795), [MATLAB FileExchange 97572 Gini index 기반 방법](https://www.mathworks.com/matlabcentral/fileexchange/97572-gini-index-based-methods-and-indexes)
- **Spectral Gini / L2/L1 norm (Wang 2018, partially)**: SK, spectral L2/L1 norm, spectral smoothness index의 역수, spectral Gini index가 **모두 최대 squared envelope에 단조 증가**하므로 모두 이상치(outlier)에 민감하다. 같은 저자의 "Spectral L2/L1 norm" 논문(MSSP 104:290–293, 2018)은 SK를 squared envelope과 squared L2/L1 norm으로 분해할 수 있음을 보였다. — [peeref 5667962](https://peeref.com/zh/works/5667962), [SJTU 교수 페이지](https://faculty.sjtu.edu.cn/wangdong1/en/lwcg/220678/content/50601.htm), [BL 카탈로그 L2/L1](https://eld.bl.uk/catalog/100054831704_0x000001)
- **통합 관점 (Wang 외 2020, DOI 10.1016/j.ymssp.2020.106725, partially)**: SK, negentropy, Gini, smoothness index가 모두 "가중 정규화 squared envelope의 합(SWNSE)"이라는 하나의 틀에 들어간다고 주장한다. — [peeref 레코드(검색 결과)](https://www.peeref.com/works/23724795)

### Inferences
- 우리 오경보(정상 파일의 큰 충격 하나)와 Wang 2018의 결론은 같은 현상이다. kurtosis·crest·Gini·negentropy 같은 sparsity 지표는 사건 **하나**의 크기에 지배된다. 그래서 대역 선택 지표를 바꾸는 것만으로는 오경보가 줄지 않을 가능성이 크다. 사건 **개수·분포**(Q3)와 **모양**(감쇠·링잉)을 따로 피처로 두어야 한다.
- 실무 조합 제안: (a) 이진 트리 대역에서 SE-negentropy·Gini·kurtosis를 계산해 "최대 대역 위치 + 최대값"을 피처로 쓴다(H 가설: 고장 시 충격 공진 대역이 이동하거나 넓어진다). (b) 같은 지표를 상위 1개 사건을 제거하고 다시 계산하는 robust 버전을 추가한다. (b)는 문헌 근거가 아니라 Wang 2018의 이상치 민감성에서 나온 추론이다.
- Python 구현: fast kurtogram은 공개 포팅이 여럿 있고, infogram은 kurtogram 코드에서 지표만 바꾸면 된다(추론, 특정 저장소는 확인하지 않음).

### Gaps
- Infogram 원문에서 "SE-infogram은 비반복 단일 과도에도 유효한가"를 직접 다룬 부분은 확인하지 못했다(HAL 봇 차단).
- Protrugram의 정확한 서지(연도·권·쪽·DOI)는 확인하지 못했다. 2010 Diagnostyka 판과 MSSP 판이 따로 있는 것으로 보인다.
- Miao 2017과 Wang 2018의 DOI는 검색 결과에서 확인하지 못했다. 임의로 적지 않았다.

---

## Q2. 과도 검출·특성화(TKEO, CWT/WPT, 정합필터, MED/MCKD)는 언제 포락선 방법보다 나은가?

### Takeaway
**MED는 "주기를 모르는/불규칙한 충격"에 쓸 수 있는 블라인드 디컨볼루션**이다. 전달경로(링잉)를 되돌려 충격을 날카롭게 만든다. 그래서 "링잉이 긴 정상 충격"과 "짧은 고장 충격"을 구분하는 데 쓸 만하다. **MCKD**는 주기 T를 입력으로 받으므로 우리 데이터에는 맞지 않는다. TKEO는 순간 에너지를 추적해 사건 시작점을 잡는 데 유리하지만 잡음에 약하다.

### Cited Findings
- **AR + MED (Endo & Randall 2007, verified, DOI 10.1016/j.ymssp.2006.02.005)**: AR 필터는 자기상관 기반이어서 위상에 둔감하다. MED는 kurtosis 같은 고차 통계를 최대화하는 역필터로 이를 보완해, 기어 이 스폴과 필렛 균열 검출을 개선했다. MSSP 21(2):906–919. — [검색 결과(scilit·peeref 레코드)](https://peeref.com/works/67771483)
  - **적용**: (1) 정상 파일로만 AR 모델을 학습해 예측 잔차를 만든다. 정상 진동 패턴을 지우고 "새로운 것"만 남기는 방식이며, 학습 데이터 누출 없이 fold 안에서 학습한다. (2) 잔차에 MED를 적용하고 사건 피처를 다시 계산한다. 링잉이 긴 정상 충격은 MED 후 단일 스파이크로 압축되므로 링잉 길이 자체가 피처가 된다(추론).
- **MED + SK (Sawalhi, Randall, Endo 2007, partially, DOI 10.1016/j.ymssp.2006.12.002)**: MED로 전달경로 효과를 제거한 뒤 SK를 적용해 베어링 결함을 강화했다. — 같은 검색 결과
- **MCKD (McDonald, Zhao, Zuo 2012, partially)**: "correlated kurtosis"를 최대화하는 FIR 필터를 골라 **주기 T에서 반복되는 충격**을 강조한다. 기어 이 칩 검출에 적용했다. MSSP 33:237–255. MATLAB 코드(`mckd(x,filterSize,termIter,T,M,plotMode)`)가 공개돼 있다. — [MathWorks FileExchange 31326](https://www.mathworks.com/matlabcentral/fileexchange/31326), [Jia 외 2015 Sensors(PMC4701337) 2차 서술](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4701337/)
  - **적용**: 함수 시그니처가 주기 T를 요구한다. 텐션암 충격은 비주기이므로 **부적합**하다. 보고서에 "검토 후 제외"로 남긴다.
- **최대 kurtosis 적응 필터링 (arXiv 1309.5004, "Blind Deconvolution via Maximum Kurtosis Adaptive Filtering")**: MED 계열을 적응 필터로 구현한 프리프린트. 검색 결과로 존재만 확인했다. — [arXiv 1309.5004](https://arxiv.org/pdf/1309.5004)
- **TKEO**: 비선형 미분 연산자로 순간 총에너지를 추정하며, 베어링의 반복 충격 추출에 쓰인다. 흔히 웨이블릿 패킷으로 잡음을 줄이고 Teager 에너지를 FFT해 결함 주파수를 찾는다. EMD+TKEO로 로터-스테이터 부분 마찰(rub)을 검출한 사례가 있다. 2024년 비교 연구는 TKEO가 강한 잡음에서 잡음 강건성과 주파수 추정 정확도가 낮다고 지적한다. **기계적 풀림(looseness)에 직접 적용한 문헌은 이번 검색에서 찾지 못했다.** — [IJ 레코드(Inderscience 106377)](https://www.inderscience.com/filter.php?aid=106377), [Teager-Huang rub 검출(CROSBI)](https://www.bib.irb.hr/959426), [core.ac.uk 206710704](https://core.ac.uk/download/206710704.pdf)
  - **적용**: Ψ[x](n) = x(n)² − x(n−1)x(n+1)은 진폭²·주파수²에 비례한다. 그래서 같은 진폭이라도 고주파 충격에 더 크게 반응한다. 200 Hz HPF 신호에 TKEO를 적용하고 이동평균한 뒤 **현재 8σ(MAD) 임계 검출을 대체하거나 병행**해 사건 시작점을 더 정확히 잡는다. 이렇게 하면 rise time(Q3) 추정이 좋아진다. 우리 신호는 진폭이 크고 SNR이 높은 편이라 잡음 취약성은 덜 문제일 것으로 본다(추론).

### Inferences
- 포락선(Hilbert) 방법이 이기는 경우: 충격이 주기적이어서 포락선 스펙트럼에 선이 생길 때다. 우리 데이터에서는 해당되지 않는다(팀 관찰: 포락선 스펙트럼 피크가 약함). 따라서 포락선은 **시간 영역 포락선의 모양**(Q3)에만 쓰고 포락선 스펙트럼 피처는 줄이는 편이 합리적이다.
- MED/AR 잔차가 이기는 경우: 전달경로 링잉이 사건을 번지게 해 kurtosis를 낮출 때다. "놓친 고장 = 조용하거나 아래쪽으로 벗어남"과 직접 맞지는 않는다. 그 고장은 충격이 아니라 **에너지 결핍/감쇠 증가**(부싱 경화·연화)일 수 있으므로, 충격 검출 강화보다 대역별 RMS의 하향 이탈 피처가 맞을 수 있다(추론, `단계/1_기준·기록/물리가설.md`에 가설로 올릴 것).
- 정합필터: 사건 템플릿(정상 파일 평균 충격 응답)과의 상관 계수를 "정상 충격과 얼마나 닮았나"로 쓰는 방법이다. 직접 근거 문헌은 이번 검색에서 확보하지 못했다.

### Gaps
- MCKD 2012의 DOI는 검색 결과에서 확인하지 못했다.
- TKEO 원 논문(Kaiser 1990)과 looseness 적용 1차 문헌은 확인하지 못했다.
- CWT/WPT 기반 과도 검출을 비주기 충격에 적용해 포락선 방법과 정량 비교한 문헌은 찾지 못했다.

---

## Q3. 충격 모양 피처(rise time, ring-down/감쇠율, 충격 에너지, 사건 간격 통계)와 양성 vs 고장 충격 구분

### Takeaway
학술 1차 문헌은 드물다. 가장 직접적인 근거는 자동차 구동계 clunk 연구(Crowther 외 2007 JSV)의 **충격 수·성격 전역 지표, 충격 전·후 지표, windowed mean-square 가속도**와, 루즈파트 모니터링 특허의 **"급상승 후 지수 감쇠" 판정**이다. 우리 오경보(링잉이 길거나 사건 뒤 배경이 올라가는 정상 충격)를 구분하려면 사건별 감쇠 시상수와 사건 전·후 배경비를 피처로 두는 것이 문헌과 맞다.

### Cited Findings
- **Crowther, Janello, Singh 2007 (verified, DOI 10.1016/j.jsv.2007.05.055)**: 여러 클리어런스를 가진 MDOF 비틀림 계(차량 구동계)에서 충격원을 정량화했다. 충격 수와 성격에 대한 전역 지표, 충격 전·후 지표, 충돌하는 물체의 상대 운동에너지와 가속도를 제안했다. 충격이 여러 번 일어나면 windowed mean-square 가속도가 에너지 성격의 지표가 된다고 보고했다. JSV 307:428–451. — [zhenzhi 레코드](https://data.zhenzhi.com/details/1/874237267), [BL 카탈로그](https://eld.bl.uk/catalog/100138413416_0x000001)
  - **적용**: 지금 쓰는 50 ms windowed RMS가 이 "windowed mean-square" 지표에 해당한다. 보고서 근거로 인용할 수 있다. 여기에 **사건 전 50 ms 대비 사건 후 50–200 ms RMS 비**(pre/post 지표)를 더하면 "사건 후 배경이 올라간 정상 파일" 오경보를 직접 겨냥할 수 있다.
- **루즈파트 모니터링 특허 (US 8,446,287)**: 충격 진동은 가파르게 상승한 뒤 지수적으로 감쇠한다. 그래서 신호의 감쇠 특성을 확인해 진짜 충격인지 판정한다. — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8446287)
- **충격 분석 특허 (US 12,227,137)**: 동적 모드 분해(DMD)로 감쇠 시상수와 진동 주파수를 추출하고, 그 피처를 군집화해 충격 유형을 분류한다. — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12227137)
- **rise time / rise angle (arXiv 2511.01592, 항공 복합재 충격 식별)**: rise time = 시작점에서 피크까지 시간, rise angle = 피크 진폭 / rise time. 이를 시간·주파수·시간-주파수 도메인 후보 피처로 묶고 도메인 지식으로 선택했다. — [arXiv 2511.01592](https://arxiv.org/pdf/2511.01592)
- **다중 임펄스 응답의 감쇠비 추정 (MDPI Signals 6(1):13)**: 웨이블릿 기반으로 감쇠비를 추출한다. 충격 타이밍이 동기화되지 않으면 추정이 불안정해진다고 지적한다. — [MDPI](https://mdpi.com/2624-6120/6/1/13)
- **FETT (Feature Extraction Toolbox for Transients, TUM)**: slope, time above threshold, slope of decay 같은 과도 피처를 계산한다. — [TUM 문서](https://mediatum.ub.tum.de/doc/1357429/229767.pdf)

### Inferences
- 사건 단위 피처 세트 제안(각 사건에서 계산한 뒤 파일 단위로 중앙값·최대·IQR로 집계):
  1. rise time(TKEO 시작점 → |x| 피크), 2. 감쇠 시상수 τ(피크 이후 포락선의 log 선형 회귀 기울기, 또는 −20 dB 도달 시간), 3. 사건 에너지(∫x² over 사건 창), 4. pre/post 배경 RMS 비, 5. 사건 내 주파수 중심(spectral centroid), 6. 사건 간격의 변동계수(CV)와 사건 수.
- 집계에서 **최대값 대신 중앙값·상위 2번째 값**을 함께 두면 "큰 충격 하나" 오경보에 덜 민감해진다(Q1의 Wang 2018 추론과 연결).
- 물리 해석 가설(검증 필요): 부싱 고무가 열화되면 감쇠가 바뀐다(경화 → 짧은 링잉·고주파 / 균열·이탈 → 금속 접촉 → 짧은 rise time과 높은 중심 주파수). 볼조인트 유격이 생기면 반복 접촉으로 사건 수가 늘고 간격이 불규칙해진다. → `단계/1_기준·기록/물리가설.md`에 H로 등록하고 위 피처를 연결할 것.

### Gaps
- 차량 현가 부품에서 "양성 충격(노면·지그 접촉) vs 결함 충격"을 감쇠·rise time으로 구분한 학술 1차 문헌은 찾지 못했다. 근거는 특허와 다른 분야(복합재·구동계)다.
- Crowther 2007의 정확한 지표 수식은 원문을 확인하지 못했다.

---

## Q4. 자동차 NVH: clunk / rattle / squeak(BSR) 검출 논문과 객관 지표

### Takeaway
BSR 분야는 주로 **음향(마이크) + 음질 지표**(sharpness, roughness, fluctuation strength, N10 loudness)를 쓴다. clunk 분야는 **구동계 비틀림 모델 기반 충격 지표**를 쓴다. 가속도 단일 채널 2초 분류에 그대로 옮길 수 있는 것은 "사건 검출 → 사건별 지표 최대값 공간에서 분류"라는 구조다.

### Cited Findings
- **Lee, Kim, Kim — SAE 2015-01-2258 (partially)**: squeak·rattle 사건을 자동으로 검출하고 유형을 구분하는 알고리즘. 사건의 sharpness·roughness·fluctuation strength **최대값**이 이루는 3차원 공간에서 squeak와 rattle을 구분했다. 녹음 86건에 적용해 거의 100% 맞게 식별했다고 보고했다. — [SAE Mobilus](https://saemobilus.sae.org/content/2015-01-2258)
  - **적용**: roughness(15–300 Hz 변조)와 fluctuation strength(약 4 Hz 변조)는 원래 음향 지표다. 하지만 가속도 신호에도 "포락선 변조 스펙트럼의 저주파 대역 에너지"로 비슷한 피처를 만들 수 있다(추론). 사건별 지표의 **최대값 공간**에서 분류하는 구조는 우리 per-event 피처 설계에 참고가 된다.
- **Oh & Singh — SAE 2005-01-2291 (partially)**: FWD 수동변속기 차량의 clunk를 축소 차수 비선형 비틀림 모델로 해석하고, 계산한 시간응답에서 clunk(날카로운 과도) 정량 지표를 제안했다. — [SAE Mobilus](https://saemobilus.sae.org/papers/examination-clunk-phenomena-using-a-non-linear-torsional-model-a-front-wheel-drive-vehicle-manual-transmission-2005-01-2291)
- **Gilbert 외 — SAE 2001-01-1502 (partially)**: 수동변속기 구동계 clunk를 시험과 해석(집중질량 과도 모델)을 함께 써서 정량화했다. — [SAE Mobilus](https://saemobilus.sae.org/content/2001-01-1502)
- **Bayani, Wickman, Söderberg (Chalmers, partially)**: rattle 소리의 불쾌감을 예측하는 새 심리음향 객관 지표를 제안하고 주관 청취 시험으로 검증했다. — [Chalmers research](https://research.chalmers.se/en/publication/525790)
- **N10 loudness 객관 측정 특허 (US 7,437,274)**: 가진기나 4-poster로 차량을 가진하고 마이크로 녹음한 뒤 N10 loudness로 변환해 임계값과 비교한다. — [Google Patents](https://patents.google.com/patent/US7437274)
- **Brines 외 — SAE 2001-01-1554 (partially)**: 차체 직접 가진은 매우 조용해서 시험 장비 소음이 squeak·rattle을 가리지 않으므로 객관 측정에 유리하다고 주장한다. — [SAE Mobilus](https://saemobilus.sae.org/papers/application-direct-body-excitation-toward-developing-a-full-vehicle-objective-squeak-rattle-metric-2001-01-1554)
- **Purdue 2021 사례 (partially)**: 주관 청취 기반 S&R 라인말단(end-of-line) 시험을 마이크·분석기로 대체했다. 16대 파일럿으로 합격/불합격 한계를 설정했다. — [Purdue docs](https://docs.lib.purdue.edu/iclss/2021/sectors/6)
- **현가 부품 고장 진단 (Machines 2023, UiT, partially)**: 가속도계 1개를 쓴 시험대에서 정상 1 + 고장 7 조건(볼조인트 마모, 컨트롤암 부싱 마모 포함)을 다뤘다. 진동을 radar plot으로 바꿔 전이학습으로 분류했다. — [UiT Munin](https://munin.uit.no/bitstream/10037/31787/2/article.pdf)
- **Chalmers 학위논문**: 다물체 시뮬레이션에서 부싱 강성 저하와 볼조인트 유격을 모델링하고, 창 단위 신호에 통계적 검출기를 적용했다. — [Chalmers ODR](https://odr.chalmers.se/bitstreams/4b6cbd6e-a28c-40a1-b8c9-fa449f055d27/download)

### Inferences
- 우리 과제와 가장 가까운 것은 UiT 2023(같은 부품, 가속도 1채널)과 Purdue EOL 사례(라인 합격/불합격 한계 설정)다. 발표 자료의 "도메인 배경" 슬라이드에 쓸 만하다.
- Purdue처럼 "정상 분포 기반 한계(예: 정상 95% 분위)"로 설정하는 방식은 팀의 "정상 5–95% 겹침 그림" 분석과 같은 사고방식이다.

### Gaps
- SAE 논문 본문(지표 정의 수식)은 유료라 확인하지 못했다.
- 현가 부품 EOL 시험의 가속도 기반 BSR 검출 알고리즘을 공개한 1차 학술 문헌은 찾지 못했다.

---

## Q5. 짧은 기록(2 s)·고샘플률 실무: STFT 창 선택, 회전이 없을 때 사이클로정상 방법의 적용성

### Takeaway
이번 검색에서 이 질문에 직접 답하는 1차 문헌은 확보하지 못했다. 아래는 위 문헌에서 끌어낸 추론이다. 사이클로정상 도구(SES, autogram, protrugram, MCKD, infogram-SES)는 **주기성(회전) 전제**가 있어 텐션암에는 맞지 않는다. 시간 영역 지표(SE-negentropy, kurtosis, Gini)와 사건 단위 분석이 맞다.

### Cited Findings
- Autogram은 "2차 사이클로정상 신호의 자기공분산은 주기적"이라는 성질에 기반한다. — [polito IRIS](https://iris.polito.it/handle/11583/2696265)
- Protrugram은 결함 주파수에 대한 사전 지식이 필요하고 결함 톤이 약하면 실패한다. — [Barszcz 2010 Diagnostyka 등 2차 서술 검색 결과](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-article-BAR0-0051-0049/c/httpwww_bg_utp_edu_plartdiagnostyka2010barszcz.pdf)
- MCKD는 주기 T를 입력 인자로 요구한다. — [MathWorks FileExchange 31326](https://www.mathworks.com/matlabcentral/fileexchange/31326)
- Infogram은 과도가 반복 간격에 비해 빨리 감쇠할 때(즉 사건이 드문드문할 때) kurtogram보다 낫다고 보고한다. — [peeref 레코드](https://peeref.com/works/2818032)

### Inferences
- **STFT 창**: 25 kHz에서 50 ms 창은 1,250샘플, 주파수 해상도는 20 Hz다. 충격 링잉(수 ms)보다 길어서 사건 하나가 창 하나에 섞인다. 충격 모양을 보려면 2–5 ms 창(50–125샘플, 해상도 200–500 Hz)과 75% 겹침이 맞다. 대역 에너지 변동성 피처에는 지금의 긴 창을 유지하는 이중 해상도 구성을 제안한다(추론. 시간-주파수 불확정성 원리에서 나온 것이며 특정 논문 근거는 아님).
- **2 s 기록 길이**: 사건이 몇 개뿐이라 SES 해상도(0.5 Hz)는 충분하지만 반복 횟수가 적어 SES 피크 통계가 불안정하다. 이것이 팀이 관찰한 "포락선 스펙트럼 피크 약함"의 원인 중 하나일 수 있다(추론).
- **회전 없는 사이클로정상성**: 시험 지그가 주기적 가진(예: 일정 주파수 가진기)을 쓴다면 그 가진 주기에 대해 사이클로정상 분석이 의미를 가질 수 있다. 가진 조건을 모르면 저주파 sway 성분의 주기를 먼저 확인할 것을 제안한다(추론).

### Gaps
- 비회전 기계에서 사이클로정상 방법의 적용성을 다룬 리뷰, 2 s 기록의 STFT 창 선택 지침 논문은 이번 검색 범위(17회)에서 찾지 못했다. 추가 조사가 필요하다.
