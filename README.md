# Risk-management-LPPL-PPO2-trader

## 강화학습(PPO2 알고리즘)과 위험성회피 전략(LPPL모델, Turblence index)을 적용시킨 트레이더

  - 도구

    [![파이썬 Badge](https://img.shields.io/badge/python-3776AB?style=flat-square&logo=python&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

    [![파이토치 Badge](https://img.shields.io/badge/pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

    [![주피터 Badge](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

  - 목표: 기존 PPO알고리즘 트레이더의 안정성과 수렴성 향상(PPO2) 및 리스크 회피

  - 진행 이유: 논문을 읽던중 Turblence index를 사용하여 위험 회피할수 있다는 내용을 보게됐다. 또한 과거 LPPL 팀프로젝트를 진행한 경험이 있어 강화학습 에이전트와 위험 회피 전략들을 결합시키면 안정적인 트레이딩을 할 수 있을것으로 생각하여 진행.

 
## 기능
  - 크롤링 및 API를 사용한 주가,코인 데이터 수집
  - 여러가지 팩터 사용
  - 차원의저주 문제 완화(PCA)
  - 데이터의 노이즈 완화(Denoise Auto Encoder)
  - 위험성 관리(LPPL모델, Turblence index 사용)
  - PPO 에이전트 개선 (PPO -> PPO2)
  - 백테스팅

## 요약
  - Bitcoin or SPY(sp 500) 호출
  - 여러 지표 계산 (stochastic RSI , Volume ratio, 투자 심리도) 
  - LPPL(위험 회피 모델) 출력후 지표로 사용
  - 모든 지표데이터를 Denoise Auto Encoder 에 통과 시켜 노이즈 제거
  - 주성분 분석을 하여 2차원 데이터로 추출
  - PPO2 에이전트 학습
  - 



## 본론

- ## PPO2 (PPO에서 추가된 점)
   -Value function clipping :implementation instead fits the value network with a PPO-like objective : 
    
   - Reward scaling  : reward 를 scaling 한다(분산 감소)
   - Reward Clipping :The implementation also clips the rewards with in a preset range : reward를 clipping한다(분산감소)

   - Observation Normalization: state s를 0-1로 정규화 시킨다. (분산 감소)
   - Observation Clipping:  state s 를 clipping 한다. (분산 감소)
   - Hyperbolic tan activations : exploration 좀더 잘할수 있도록 한다.
   - Global Gradient Clipping : actor와 critic 의 가중치를 clipping 해서 오버피팅을 방지한다

 
 
## 결론
   -

## 한계 및 개선
  - State의 정의 (차원의 저주 문제로 인해 1개의 feature만 사용)<br/>
      - 팩터들 사이에서 다중공선성 문제가 생길 수 있으므로 Feature Extraction 방법으로 차원의 저주와 높은 상관계수 문제 해결 예정<br/><br/>
  - 시장은 t시점에서 알파를 찾아도 향후 새로운 알파가 생겨난다. <br/> 
      - 더많은 에이전트를 앙상블하거나 MARL(Multi-Agent-Reinforcement-learning) 사용 예정   <br/>
      - 각 에이전트가 알고리즘 자체를 스스로 개선하도록 하여 여러 에이전트들의 전략간 상관계수와 편향을 낮출 예정

    
    


노이즈 트레이더들의 따라하기 행동은 positive feedback process 와 비례한다.
트랜드가 있으면 주가데이터는 정상성을 잃어버릴수 있으므로 price데이터 사용은 합리적이지 않다? = 학습이 price 생으로 사용하고 지표 PCA한거 수렴이 어려웠음
검증 단계에서 turbulence index를 사용하여 risk-aversion을 조정한다
turbulence index는 일별 수익률과 그들의 상호 작용에서 비정상적인 변화의 평균 정도를 포착한다.
 index t는 시장 붕괴에 대한 risk-aversion을 해결하기 위해 reward function과 통합

다중공선성

약한 정상성 :시계열데이터가 약한정상성을 만족하려면
정상성: 모든시점에 대해 일정한 평균을 가진다 . 분산이 일정하다 시계열데이터가 평균,분산이 시간에 의존하지않고 공분산은 시간이 아닌 시차(t의 텀)에만 의존해야함
이로인해 생기는 정상 시계열 모형 = MA, AR모형

[AR(1) 모형]  Yt = ф1Yt-1 + εt / 직전 시점 데이터로만 분석

[AR(2) 모형]  Yt = ф1Yt-1 + ф2Yt-2 + εt / 연속된 3시점 정도의 데이터로 분석
Yt : 현재 시점의 시계열 자료

Yt-1, Yt-2, ..., Yp : 이전, 그 이전 시점 p의 시계열 자료

фp : p시점이 현재에 어느 정도 영향을 주는지를 나타내는 모수

εt : 백색잡음과정(white noise process), 시계열분석에서 오차항을 의미




약한의존성t=> inf로 갈때 상관관계가 0으로 수렴한다면 약한의존성 만족


약한정상성과 약한 의존성을 가진다면 대수의법칙 적용가능

대수의 법칙은 ‘경험적 확률과 수학적 확률과의 관계를 나타내는 정리(定理)’이다. 즉, 표본의 관측대상의 수가 많으면 통계적 추정의 정밀도가 향상된다는 것을 수학적으로 증명한 것이다. 이론적으로는 다음과 같다.

‘무한 모집단에서 무작위로 추출된 확률변수 X가 독립적으로 동일한 분포에 따르고 E(X)=μ, V(X)=σ2인 경우, 표본의 크기가 커짐에 따라 표본 평균 =∑xi/n은 확률적으로 모집단의 실제 평균값 μ에 수속(收束)한다


