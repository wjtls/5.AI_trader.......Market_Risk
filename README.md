# Risk-management-LPPL-PPO2-trader

## 강화학습(PPO2 알고리즘)과 위험성회피 전략(LPPL모델, Turblence index)을 적용시킨 트레이딩 에이전트

  - 도구

    [![파이썬 Badge](https://img.shields.io/badge/python-3776AB?style=flat-square&logo=python&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

    [![파이토치 Badge](https://img.shields.io/badge/pytorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

    [![주피터 Badge](https://img.shields.io/badge/jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white&link=mailto:wjtls01@naver.com)](mailto:wjtls01@naver.com)

  - 목표: 기존 PPO알고리즘 트레이더의 안정성과 수렴성 향상(PPO2) 및 리스크 회피

  - 진행 이유: 논문을 읽던중 Turbulence index를 사용하여 위험 회피할수 있다는 내용을 보게됐다. 또한 과거 LPPL 팀프로젝트를 진행한 경험이 있어 강화학습 에이전트와 위험 회피 전략들을 결합시키면 안정적인 트레이딩을 할 수 있을것으로 생각하여 진행.

<br/><br/><br/><br/>
 
## 기능
  - 크롤링 및 API를 사용한 주가,코인 데이터 수집
  - 데이터의 노이즈 완화(Denoise Auto Encoder)
  - LPPL 모델사용 (위험성 회피 전략)
  - turbulence index 사용 (위험성 회피 전략)
  - PPO2 에이전트 사용 (PPO -> PPO2로 개선)
  - 백테스팅
<br/><br/><br/><br/>
## 요약
  - Bitcoin or SPY(sp 500) 호출
  - 여러 지표 계산 (stochastic RSI , Volume ratio, 투자 심리도) 
  - LPPL(위험 회피 모델) 출력후 지표로 사용 
  - 모든 지표데이터를 Denoise Auto Encoder 에 통과 시켜 노이즈 제거
  - PPO2 에이전트 학습
<br/><br/><br/><br/>

## 본론
- ## PPO2 (PPO에서 추가된 점)
   - 기존 PPO 알고리즘에 여러 기법들을 추가하여 분산을 감소 시키고 학습의 안정성과 수렴성을 높인다.
  
   - 1. Value function clipping :implementation instead fits the value network with a PPO-like objective<br/> 
      ![image](https://user-images.githubusercontent.com/60399060/147339340-637b0940-97e2-4b4a-914c-4e94bf8aecda.png)<br/> 
      PPO에서 surrogate loss를 clipping 한 것과 같이 value function 을 clipping 하여 분산을 감소 시킨다.
     <br/> <br/> 

   - 2. Reward scaling  : reward 를 scaling 한다 (분산 감소)
   - 3. Reward Clipping :The implementation also clips the rewards with in a preset range : reward를 clipping한다 (분산감소)
   - 4. Observation Normalization: state s를 0-1로 정규화 시킨다. (분산 감소)
   - 5. Observation Clipping:  state s 를 clipping 한다. (분산 감소)
   - 6. Global Gradient Clipping : actor와 critic 의 가중치를 clipping (오버피팅 방지)




<br/><br/><br/>

- ## LPPL (위험성 지표)
  - 1. LPPL의 가정
     - LPPL 모델에서는 초기에 배당이 없고 이자율, 위험회피 등이 없는 이상적인 시장을 가정. <br/><br/>
       그래서 이상적인 시장에서는 주가의 펀더멘털 가치(주식의 이론적가치)가 0이다. <br/>
       이런 조건 하에 가격이 펀더멘탈 가치를 뛰어 넘는다면 과열 상태를 의미<br/><br/><br/>

     - LPPL 모델에서는 시장에 노이즈 트레이더, 이성적 트레이더 집단이 있다고 가정.<br/><br/>
       이성적 트레이더 집단은 자신의 투자 철학을 가지고 여러 조건을 참고하여 합리적인 의사 결정을 한다.<br/>
       노이즈 트레이더 집단은 뉴스나 전문가, 이성적 트레이더의 행동을 모방하는 집단이다.<br/>
       (노이즈 트레이더들의 따라하기 행동은 positive feedback process, 합리적 트레이더의 행동은 negative feedback 와 비례 한다.)
 <br/><br/>
  - 2. LPPL 함수
     - ![image](https://user-images.githubusercontent.com/60399060/147334170-7e84add0-f730-4d7f-8dcf-70431a01d7a2.png)
     - (A,B,C,phi = 단위분포이며 어떤 구조적 정보도 제공하지 않는다, 오메가(w)= 버블 진행 시간동안 진동수, tc= 임계 시간)
     - LPPL의 가정을 바탕으로 LPPL 함수를 정의.
     - 함수는 일시적 가격 성장을 설명하며 폭락이 생기기 전 임계시간 t를 가진다.
 <br/><br/>
  - 3. 위험성 예측
     - 구간을 rolling 하여 LPPL함수를 주가에 피팅시키면 시퀀스 마다 tc,m,w,A,B,C,pi 변수들을 구할수 있다.
     - ![image](https://user-images.githubusercontent.com/60399060/147338065-ba857eb0-da17-4a33-9315-8a0894e4476c.png)
     - ![image](https://user-images.githubusercontent.com/60399060/147338349-bba4bd3b-8741-4164-9468-b47b52d7866b.png)
     - O, D를 정의하고 변수들의 범위를 설정
     - positive feedback 계산 : 피팅을해서 구한 변수가 해당 범위안에 들어간다면 버블의 위험성이 존재한다 (카운트 +=1 하여 횟수 저장)
     - negative feedback 계산 : 변수가 해당 범위를 만족하지 않는다면 위험성이 낮다.(합리적 트레이더들이 많은 구간)
     - 
<br/><br/><br/>

 - ## turbulence index (위험성 지표)
   - 1. turbulence index는 일별 수익률과 그 상호 작용에서 비정상적인 변화의 평균 정도를 포착한다.<br/>
        ![image](https://user-images.githubusercontent.com/60399060/147341186-05008cda-c3a6-4b7f-a55d-34173a187869.png)<br/>
        yt : the stock returns for current period t (t 시간동안 return) <br/>
        u  : denotes the average of historical returns (yt의 평균) <br/>
        sigma : #denotes the covariance of historical returns (yt의 공분산) <br/>
        <br/>
   - 2. 검증 단계에서 turbulence index를 사용하여 위험회피를 한다 index t 는 시장 붕괴에 대한 <br/>
        위험회피(risk-aversion)를 해결하기 위해 reward function과 통합.<br/>
        ![image](https://user-images.githubusercontent.com/60399060/147342208-adf94472-8517-4d74-b7e5-7b989f16e5f4.png)<br/>
        turbulence index가 기준 값에 도달하면 전량 매도한다.
         
 <br/><br/><br/><br/>
   

## 결론
   -
<br/><br/><br/><br/>
## 한계 및 개선
  - State의 정의 <br/>
      - 강화학습 MDP에서 State를 제대로 정의하지 못하면 네트워크의 과소적합 또는 차원의 저주로 인한 과적합 발생.  
      - state를 Stochastic RSI, 거래량 비율, 투자 심리도, LPPL모델 , 로그 수익률 등의 지표를 사용할 경우 <br/>
        디노이징과 차원축소를 해도 과적합 발생
      - 이는 State가 강화 학습 에이전트의 목표인 수익률을 잘 설명하지 못한것이 원인.
      - state가 수익률을 제대로 설명 할 수 있게끔 정의해야 한다.  <br/>
  
  - Denoise Auto Encoder를 과적합 시키면 노이즈가 오히려 증가 했다.<br/>
      - 학습 epoch 감소 시키고 적절한 학습률 및 은닉층을 설정<br/>

  - 시장은 t시점에서 알파를 찾아도 향후 새로운 알파가 생겨난다. <br/> 
      - 더많은 에이전트를 앙상블하거나 MARL(Multi-Agent-Reinforcement-learning) 사용 예정   <br/>
      - 각 에이전트가 알고리즘 자체를 스스로 개선하도록 하여 여러 에이전트들의 전략간 상관계수와 편향을 낮출 예정<br/><br/>
  
  
  
    
    
    
트랜드가 있으면 주가데이터는 정상성을 잃어버릴수 있으므로 price데이터 사용은 합리적이지 않다
학습이 price 생으로 사용하고 지표 PCA한거 수렴이 어려웠음 


