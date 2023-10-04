Gaussian Process(1) - Bayesian Analysis
=============

해당 글의 전체적인 내용은 책 Gaussian Process for Machine Learning (Carl Edward Rasmussen)에서 가져왔다.  
Gaussian Process를 이해하기 위해서 간단한 개념부터 공부해보겠다.  
공부하면서 적는 글이라 틀린 부분이 꽤 있을 수 있다. 


## Bayesian Inference

해당 글에서 가장 중요한 개념인 Bayesian inference의 정의는 다음과 같다.  
> Bayesian inferenece is a method of statistical inference in which Bayes' theorem is used to update the probability for a hypothesis as more evidence or information becomes available. (from. 위키피디아)  


한국어로 번역하자면,  
어떤 궁금한 사건에 대해서 새로운 데이터가 추가되었을 때 베이즈 이론을 이용해 그 사건의 확률을 갱신할 수 있다.  
이것을 베이지안 추론이라 부르고 확률적인 추론의 한 가지 방법으로 보고 있다.   
   
베이즈 이론의 기초적인 이론인 조건부 확률부터 확인해보자.  
아래 공식은 고등학교 때 배웠던 조건부 확률 공식이다.

$$ P(A|B) = \frac{P(A \cap B)}{P(B)} \qquad (1) $$  

간단하게 설명하자면 B라는 사건이 발생하였다는 조건에, A라는 사건이 발생할 확률을 의미한다.  
이것의 형태를 조금만 변경하면 베이즈 이론이 된다.  

$$ P(A \cap B) = P(A|B)*P(B) = P(B|A)*P(A) \qquad (2) $$  
$$ P(A|B) = \frac{P(A \cap B)}{P(B)} = \frac{P(B|A)P(A)}{P(B)} \qquad (3) $$

식 (3)이 베이즈 이론이다.  
처음에 봤을 때 수식은 이해가 된다만, 대체 무슨 의미가 있는지 의문이었다.  
베이즈 이론 수식의 각 성분 별로 무슨 의미가 있는지 정리해보자.  

1. 사건 A : Hypothesis라고 하며, 기존 데이터에 의해 영향을 받는 확률이다.
2. 사건 B : Evidence라고 하며, 기존 확률을 계산할 때 사용되지 않았던 데이터에 해당된다.
3. $P(A)$ : Prior Probability라고 하며, 사전 확률이라는 의미이다. 데이터에 해당되는 사건 B가 관측되지 않은 상태에서의 사건 A(Hypothesis)의 확률이다.
4. $P(B|A)$ : Likelihood라고 하며, 사건 A라는 데이터가 새로 주어졌을 때 갱신되는 B의 조건부 확률이다.
5. $P(B)$ : Marginal likelihood라고 하며, 모든 가능한 사건 A가 고려되었을 때의 사건 B의 확률이다.
6. $P(A|B)$ : Posterior Probability라고 하며, 데이터에 의해 갱신되는 가장 중요한 확률이다.

간단하게 해당 성분들의 역할? 흐름?을 생각해보자.  
우리는 사건 B가 발생했을 때의 사건 A의 확률(Posterior Probability)이 궁금하다.  
Posterior Probability은 기존에 존재했던 사건 A로 계산된 Prior Probability에 의해 갱신되는 새로운 사건 A에 대한 확률이다.  
(처음에 이게 너무 헷갈렸다.)  

기존의 사건 A로 계산된 $P(A)$ (Prior Probability)가 있고, 새로운 데이터인 사건 B가 관찰된다.  
그러면 $P(B|A)$ (Likelihood)와 $P(B)$ (Marginal likelihood)를 계산할 수 있다.  
결과적으로 식 (3)에 의해 새로운 데이터 사건 B에 대하여 새로운 사건 A에 대한 확률(Posterior Probability)를 얻을 수 있다.  

내용이 다소 복잡해보일 수 있는데 간단히 요약하자면,   

>새로운 데이터가 입력되었을 때 기존 데이터 기반으로 만들어졌던 확률을 갱신할 수 있다

는 의미다.  

수식으로 위 분포에 대해서 이해해보자.

***
   
### 1. Prior Probability

위에서 정리한 베이즈 이론을 이용해서 우리가 알고 있는 것은 무엇이며, 알고 싶은 정보는 무엇인지 정리해보자.  
알고 있는 정보는 관찰된 데이터 즉, $\mathbf{x, y}$이다.  
추론해야 하는 정보는 weight인 $\mathbf{w}$이다.  
이것에 따라서 Likehood, Prior, Posterior 순으로 정리해보면 다음과 같다.  
($X$는 모든 $\mathbf{x}$의 집합이다.)

* Likehood : $p(\mathbf{y}|X, \mathbf{w})$
* Prior : $p(\mathbf{w})$
* Posterior : $p(\mathbf{w}|X, \mathbf{y})$

여기서 Prior Probability는 기존에 주어진 $\mathbf{w}$에 대한 확률이다.  
$\mathbf{w}$는 다변수 정규분포를 따르는 벡터라고 가정하자.
그러면 Prior Probability는 다음과 같다.

$$p(\mathbf{w}) = \frac{exp(-\frac{1}{2}\mathbf{w}^T \Sigma^{-1}_p\mathbf{w})}{\sqrt{(2\pi)^k |\Sigma_p|}} \qquad (4)$$

***

### 2. Likelihood
바로 위에서 정의한 Likelihood는 아래와 같이 나타낼 수 있다.   
$$p(\mathbf{y}|X, \mathbf{w})=\prod_{i=1}^n p(y_i|\mathbf{x}_i, \mathbf{w})= \prod _{i=1}^n \frac{1}{\sqrt{2\pi}\sigma_n}exp(-\frac{(y_i - \mathbf{x}_i^T\mathbf{w})^2}{2\sigma_n^2})$$

$$=\frac{1}{(2\pi\sigma_n^2)^{n/2}}exp(-\frac{1}{2\sigma_n^2}|\mathbf{y}-X^T \mathbf{w}|^2)=\mathcal{N}(X^T\mathbf{w}, \sigma_n^2I) \qquad (5)$$
위 수식을 간단히 설명해보자면 다음과 같다.  
>한 쌍의 parameter인 $\mathbf{w}$에 대한 Likelihood의 값은 모든 $\mathbf{x, y}$ 쌍에 대한 Likelihood의 합과 같다.

Likelihood는 확률이라기보다 데이터와 분포 사이의 적합도로 볼 수 있다.  
   
하나의 예시를 간단히 들어보겠다.  
평균이 각각 0(분포 1)과 100(분포 2)인 분포 2개가 있다고 가정하자. (분산은 같다.)   
둘 중에 임의로 선택한 분포에서 추출된 데이터가 [-1, 0, 1, 0, 0.5]라고 하자.   
그럼 해당 데이터는 분포 1, 2 중에 어느 분포에서 추출되었을 확률이 높은가?  
단순히 생각해보면 분포 1에서 추출되었을 확률이 높다.  
즉, 해당 데이터는 분포 2보다 분포 1과의 Likelihood가 높은 것이다.  
(위 예시에서의 데이터가 $\mathbf{x, y}$이고, 평균과 분산이 $\mathbf{w}$라고 생각하면 된다.)  

#### - Marginal Likelihood
Marginal Likelihood는 일반 Likelihood에서 특정 변수를 모두 고려한 결과이다.  
예를 들어, 식 (5)와 같이 정의되었던 Likelihood인 $p(\mathbf{y}|X, \mathbf{w})$에서 $\mathbf{w}$를 모두 고려한다는 의미다.  
식으로 나타내면 아래와 같다.  

$$p(\mathbf{y}|X)=\int{p(\mathbf{y}|X, \mathbf{w})p(\mathbf{w})d\mathbf{w}} \qquad (6)$$

이 식은 $\mathbf{w}$와 independent하며, Posterior Probability 관점에서 봤을 때는 단순히 상수에 불과하다.  
Posterior Probability 기준에서는 $\mathbf{w}$에 dependent한 요소만 변수로 보기 때문이다.

***

### 3. Posterior Probability
Bayesian 추론에서 가장 중요하다고 볼 수 있는 Posterior Probability(사후 확률)이다.  
Bayes' rule에 의하면 식 (3)과 같이 사후 확률은 아래와 같이 정의된다.

$$Posterior=\frac{likelihood \times prior}{marginal \ likelihood} \qquad (7)$$

수식으로 나타내면 아래와 같다.  

$$p(\mathbf{w}|\mathbf{y}, X)=\frac{p(\mathbf{y}|X, \mathbf{w})p(\mathbf{w})}{p(\mathbf{y}|X)} \qquad (8)$$

식 (8)를 상수를 배제한 수식으로 간단히 하면 아래와 같다.  

$$p(\mathbf{w}|X, \mathbf{y}) \propto exp(-\frac{1}{2\sigma_n^2}(\mathbf{y}-X^T\mathbf{w})^T(\mathbf{y}-X^T\mathbf{w}))exp(-\frac{1}{2}\mathbf{w}^T\Sigma_p^{-1}\mathbf{w})$$
$$\propto exp(-\frac{1}{2}(\mathbf{w}-\bar{\mathbf{w}})^T(\frac{1}{\sigma_n^2}XX^T+\Sigma_p^{-1})(\mathbf{w}-\bar{\mathbf{w}})) \qquad (9)$$

식 (9)에서 $\bar{\mathbf{w}}=\sigma_n^{-2}(\sigma_n^{-2}XX^T+\Sigma_p^{-1})^{-1}X\mathbf{y}$이다.  
그리고 $A=\sigma_n^{-2}XX^T+\Sigma_p^{-1}$이라 할 때, 사후 확률은 아래와 같은 Gaussian 분포를 따른다.  

$$p(\mathbf{w}|X, \mathbf{y}) \sim \mathcal{N}(\bar{\mathbf{w}}, A^{-1}) \qquad (10)$$

Posterior Probability, 즉 사후 분포를 이해하기 위해서 위와 같은 긴 과정을 거쳤지만,  
결국 분포를 계산하기 위해서는 식 (10)만 이용하면 된다.  
다음 장에서는 지금까지의 베이즈 이론을 이용해 분포를 그려 Gaussian Process를 이해해볼 것이다.
