
Gaussian Process(2) - The Standard Linear Model
=============

해당 글의 전체적인 내용은 책 Gaussian Process for Machine Learning (Carl Edward Rasmussen)에서 가져왔다.  
이번 장에서는 이전에 설명했던 베이즈 이론을 기반으로 Gaussian Process를 이해해보자.   

***

### 1. Definition
이 책에 따르면 GP의 정의는 다음과 같다.   

> Gaussian Process is a collection of random variables, any infinite number of which have a joint Gaussian Distribution.

해석하면 GP는 Random Variable의 집합이며, 그것의 부분 집합은 어떤 조합이든지 Joint Gaussian Distribution을 따른다.   
내가 이해한 바로는 하나의 $\mathbf{x}$값에 대한 GP의 결과가 특정한 함수 값이 아니라, 평균과 분산으로 표현되는 확률 분포라는 말이다.   

Real Process $f(\mathbf{x})$가 있다고 가정할 때, 그것의 평균( $m(\mathbf{x})$ )과 공분산( $k(\mathbf{x, x'})$ )은 아래와 같다.   

$$m(\mathbf{x}) = E[f(\mathbf{x})] \qquad (1)$$
$$k(\mathbf{x, x'}) = E[(f(\mathbf{x})-m(\mathbf{x}))(f(\mathbf{x'})-m(\mathbf{x'}))] \qquad (2)$$

그리고 Gaussian Process는 다음과 같은 분포를 따른다.   

$$f(\mathbf{x}) \sim GP(m(\mathbf{x}), k(\mathbf{x, x'})) \qquad (3)$$

$\mathbf{x, x'}$는 서로 다른 데이터라는 의미다.   
공분산 함수인 $k(\mathbf{x, x'})$은 Kernel이라고도 하며, 여러 종류의 함수가 존재한다.   
그 중에서 우리는 간단한 식 (4)와 같은 함수를 이용할 것이다.   

$$k(\mathbf{x, x'})=exp(-\frac{1}{2} |\mathbf{x}-\mathbf{x'}|^2) \qquad (4)$$

***

### 2. GP Prior

Gaussian Process에서 학습된 데이터가 어떤 효과를 가지는 지 확인할 수 있도록 Prior 분포를 먼저 그려보자.   
Prior 분포는 이전에 설명했던 것처럼 학습된 데이터가 없을 때의 분포이다.   
즉, 관찰하려는 데이터가 모두 테스트 데이터($\mathbf{x_*}$)이다.   

테스트 데이터에 대한 결과는 식 (5)와 같은 분포를 따른다.   
추정하려는 함수에 대한 정보가 없으니 평균은 0으로 가정한다.   

$$\mathbf{f} _* \sim \mathcal{N} (\mathbf{0}, K(X _ *, X _ *)) \qquad (5)$$

위 식에서 *가 붙은 $X _ *, \mathbf{f} _ *$는 각각 테스트 입력 데이터, 데스트 결과 데이터이다.   
학습된 데이터가 없을 때 테스트 결과 데이터는 테스트 데이터의 공분산에 의해서 정해진다.   
사실 GP Prior는 크게 의미 있진 않지만, 학습 데이터가 어떤 효과를 가져오는 지 확인하기 위함이다.    
(그냥 N변수 분포에서 임의의 난수를 추출한 데이터라고 보면 된다.)    

참고로 $m \times m$ 크기의 $K(X _ *, X _ *)$ 행렬은 식 (6)과 같다.   
($m$은 테스트 데이터의 개수)

$$K(X _ *, X _ *) = \begin{bmatrix} k(x _ {*1}, x _ {*1}) & k(x _ {*1}, x _ {*2}) &  \cdots  & k(x _ {*1}, x _ {*m}) \\\ k(x _ {*2}, x _ {*1}) & k(x _ {*2}, x _ {*2}) &  \cdots  & k(x _ {*2}, x _ {*m}) \\\  \vdots  &  \vdots  &  \ddots  &  \vdots  \\\ k(x _ {*m}, x _ {*1}) & k(x _ {*m}, x _ {*2}) &  \cdots  & k(x _ {*m}, x _ {*m}) \\\  \end{bmatrix} \qquad (6)$$

모든 테스트 데이터끼리의 공분산을 원소로 가지는 행렬이다.   
작성 코드는 아래와 같다.    

```matlab
clc; clear all; close all; 

X = linspace(-5,5,200);                            % The Region of GP's input
N_sample = 5;                                      % The number of GP's samples
y_result = zeros([N_sample,length(X)]);
K_prior = zeros(length(X),length(X)); 

for i=1:length(X) 
	for j=1:length(X) 
		K_prior(i,j) = exp(-0.5*(X(i) - X(j))^2);  % Kernel Function
	end 
end 

mu_prior = zeros(1, length(X)); 
y_result = mvnrnd(mu_prior,K_prior,N_sample); 
lw_bd = mu_prior - 2*sqrt(diag(K_prior)); 
up_bd = mu_prior + 2*sqrt(diag(K_prior)); 

% Plot Prior Distribution
figure(1) 
plot(X, y_result); 
hold on; 
grid on; 
plot(X, lw_bd, 'b', 'LineWidth', 3); 
plot(X, up_bd, 'b', 'LineWidth', 3); 
plot(X, mu_prior, 'r', 'LineWidth', 2); 
legend('Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Variance -(95%)', 'Variance +(95%)', 'Mean'); 
ylim([-3, 3]); 
title('GP Prior'); 
xlabel('input, x'); 
ylabel('output, f(x)');
```

<p align="center">
  <img src="/picture/gp_prior.png"/>   
  그림 (1) GP Prior 분포
</p> 

(평균은 빨간 선임. Legend가 잘못 표시됨.)   
Sample 1~5번은 공분산을 따르는 분포에서 임의로 추출한 데이터 집합을 표현한 것이다.   
헷갈리지 말아야 할 것이 각 Sample로 그려진 하나의 그래프가 N차원(input의 크기) 데이터 하나인 것이다.   
Sample에 그려진 그래프 점들이 각각 개별적인 데이터가 아니라는 말이다.   

이해를 돕기 위해서 1변수 정규분포부터 시작해보자.   

$$x \sim  \mathcal{N}(0, 1) \qquad (7)$$

식 (7)과 같은 1변수 정규분포를 따르는 분포에서 임의로 난수를 하나 추출했다고 생각해보자.   

```matlab
for i=1:5 
	normrnd(0, 1)
end
```

결과는 다음과 같다.   

$$0.5377, 1.8339, -2.2588, 0.8622, 0.3188$$

1변수 정규분포에서 난수를 5번 뽑았으니 하나의 스칼라 값이 5번 나왔다.(당연한 것)   

그럼 2변수 정규분포를 따르는 분포에서 임의의 난수를 추출해보자. 이번엔 3개만   

```matlab
for i=1:3 
	mvnrnd([0 0], [1 0; 0 1]) 
end
```

결과는 다음과 같다.   

$$(-1.3499, 3.0349), (0.7254, -0.0631), (0.7147, -0.2050)$$

2개가 1쌍인 벡터가 3쌍 나왔다.

이번엔 공분산을 바꿔서 난수를 추출해보자.   

```matlab
for i=1:3 
	mvnrnd([0 0], [1 0.99; 0.99 1]) 
end
```

$$(0.8884, 0.7177), (-1.0689, -1.1724), (-2.9443, -2.7119)$$

두 변수의 유사성을 나타내는 성분(대각이 아닌 성분)을 0에서 0.99로 증가시켰다.   
그러니 임의의 난수를 추출했을 때 두 변수가 수치가 비슷해진 것을 확인할 수 있다.   
공분산이 단위행렬인 이전의 경우를 보면 두 변수의 차이가 비교적 큰 것을 알 수 있다.   

그럼 아래와 같은 공분산 분포를 가지는 N차원 행렬은 무엇을 의미할까?   
(노란색에 가까울 수록 값이 1에, 어두울 수록 값이 0에 가까움)

(공분산 분포 그림(2) 추가)
<p align="center">
  <img src="/picture/se_kernel.png"/>   
  그림 (2) GP Prior 분포
</p> 

**나만의 나름의 해석**
- 임의의 난수를 한 번 추출했을 때 N개가 1쌍인 데이터가 추출되며, 가까이 있는 성분일 수록 추출된 값이 비슷할 것이다.
- 예를 들면, N이 10이라고 가정하면 난수를 한 번 추출했을 때 10개의 스칼라가 1쌍인 벡터가 추출될 것이다. 이때 1번째 성분(스칼라)은 10번째 성분보다 2번째 성분과의 공분산이 높다. → 1번째 성분은 10번째 성분보다 2번째 성분과 유사할 확률이 높다.

그림 (2)와 같은 공분산 분포는 GP Prior를 그릴 때 사용된 공분산으로서, GP Prior의 각 Sample이 부드럽게 그려지는 이유가 바로 위에서 언급한 공분산의 영향 때문이지 않을까 싶다.
- 가까운 input끼리는 비슷한 output을 가질 확률이 높다. (추측)

이와 관련해서 책에서는 GP Prior가 부드럽게 그려지는 이유를 아래와 같이 설명한다.   

> Notice that "informally" the functions look smooth. In fact the squared exponential covariance function is infinitely differentiable, leading to the process being infinitely mean-square differentiable

해석하자면, 공분산 함수가 무한번 미분 가능하기 때문에 그에 의한 GP도 무한번 미분 가능한 부드러운 형태가 나오는 것이다.   
느낌상으로는 그럴 것 같은데 수학적?으로는 이해가 되지 않는 부분이다.   

***

### 3. GP Posterior
이번엔 GP Posterior 분포를 그려보자.   
Prior 분포와 다르게 아래와 같은 학습 데이터가 임의로 주어진다.   

$$X = [-4, -3, -1, 0, 2]^T, \quad  \mathbf{f} = [-2, 0, 1, 2, -1]^T$$

GP는 조건부 분포의 일종으로 training output인 $\mathbf{f}$가 주어졌을 때 test output인 $\mathbf{f_*}$은 식 (8)과 같은 분포를 가진다.

$$\begin{bmatrix}  \mathbf{f}  \\\  \mathbf{f _ *}  \end{bmatrix}  \sim  \mathcal{N}(\mathbf{0}, \begin{bmatrix} K(X, X) & K(X, X _ *) \\\ K(X _ *, X) & K(X _ *, X _ *) \end{bmatrix}) \qquad (8)$$

참고로 여기서 사용하는 Kernel은 다음과 같이 이전에 쓰던 것에서 조금 수정하였다.   
사실상 $l$은 1로 둘 것이기 때문에 GP Prior과 차이가 없음.   

$$k(\mathbf{x}, \mathbf{x'}) = exp(-\frac{1}{2l^2}|\mathbf{x}-\mathbf{x'}|^2) \qquad (9)$$

$l$은 length scale이라는 hyperparameter로 input의 차이에 따라 공분산의 비중을 변화시키는 역할을 한다.   
$l$이 감소하면 먼 거리에 있는 input의 비중이 커지고, 증가하면 가까운 거리에 있는 input의 비중이 커진다.   
여기서 $\mathbf{f_*}$에 대한 조건부 분포는 다음과 같다.   

$$\mathbf{f _ *}|X _ *, X, \mathbf{f}  \sim  \mathcal{N}(K^T _ *K^{-1}\mathbf{f}, K _ {**}-K^T _ *K^{-1}K _ *) \qquad (10)$$
$$(K = K(X, X), \quad K _ *=K(X, X _ *)=K(X _ *, X)^T, \quad K _ {**}=K(X _ *, X _ *))$$

여기서 $X$는 학습 데이터이며, $X_*$는 테스트 데이터다.   
-5부터 5까지 입력해서 결과를 볼 예정이기 때문에 이를 테스트 데이터로 본다.

코드는 아래와 같다.   

```matlab
clc; clear all; close all; 

X_star = linspace(-5,5,200);  % Test Input Data
X_train = [-4 -3 -1 0 2];     % Train Input Data
y_train = [-2 0 1 2 -1];      % Train Output Data
X_mat = [X_train, X_star];

N_sample = 5;                 % The number of GP's samples

y_result = zeros([N_sample,length(X_star)]); 
K_matrix = zeros(length(X_mat),length(X_mat)); 

len_val = 1;                  % Length Scale

for i=1:length(X_mat) 
	for j=1:length(X_mat) 
		% Kernel Function
		K_matrix(i,j) = exp(-0.5/(len_val^2)*(X_mat(i) - X_mat(j))^2); 
	end 
end 

l = length(X_train); 
m = length(X_star); 

K_x_x = K_matrix(1:l, 1:l); 
K_x_star = K_matrix(l+1:l+m, 1:l); 
K_star_star = K_matrix(l+1:l+m, l+1:l+m); 

mu_poster = K_x_star*inv(K_x_x)*y_train';
K_poster = K_star_star - K_x_star*inv(K_x_x)*K_x_star';

y_result = mvnrnd(mu_poster,K_poster,N_sample); 

lw_bd = mu_poster - 2*sqrt(diag(K_poster)); 
up_bd = mu_poster + 2*sqrt(diag(K_poster)); 

%Plot GP Posterior
figure(1); 
plot(X_star, y_result); 
hold on; 
grid on; 
plot(X_star, lw_bd, 'b', 'LineWidth', 3); 
plot(X_star, up_bd, 'b', 'LineWidth', 3); 
plot(X_star, mu_poster, 'g', 'LineWidth', 2); 
scatter(X_train, y_train, 50, 'red', 'filled'); 
legend('Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Variance -(95%)', 'Variance +(95%)', 'Mean', 'Observed Points'); 
ylim([-3, 3]); 
title('GP Posterior'); 
xlabel('input, x'); 
ylabel('output, f(x)');
```

(GP Posterior 그림 (3) 추가)
<p align="center">
  <img src="/picture/gp_posterior.png"/>   
  그림 (3) GP Posterior 분포
</p> 

학습 데이터인 빨간 점에서는 분산의 범위가 줄어들고 학습 데이터가 없는 곳에서는 분산의 범위가 커지는 모습을 볼 수 있다.   
초록선은 각 입력에 대한 평균이며 GP에 의한 최적의 모델 추정이라고 볼 수 있다.   

공분산에 대해서 3차원으로 확인해보자. (위 코드 다음으로 실행시킬 것)   

```matlab
w1 = linspace(-5,5,200); % Range of w1 
w2 = linspace(-5,5,200); % Range of w2 

% Plot Covariance Distribution
figure(2); 
surf(w1,w2,K_poster); 
hold on; 
scatter3(X_train, X_train, 0, 'red', 'filled'); 
title('Covariance of GP Posterior'); 
xlabel('input, x'); 
ylabel('input, x'); 
zlabel('Covariance');
```

<p align="center">
  <img src="/picture/3d_gp_posterior_cov(1).png"/>   
  그림 (4) GP Posterior Covariance 3D 분포(1)
</p> 

빨간 점은 학습 데이터이다.   
학습 데이터의 공분산은 0이고 그 사이에서는 공분산이 커지는 모습을 볼 수 있다.

<p align="center">
  <img src="/picture/3d_gp_posterior_cov(2).png"/>   
  그림 (5) GP Posterior Covariance 3D 분포(2)
</p> 

측면에서 보면 더 확실히 구분할 수 있다.

(학습 데이터 : $X = [-4, -3, -1, 0, 2]^T$)

```matlab
figure(3); 
plot(w1, K_poster(60, :)); 
title('Covariance of GP Posterior about x = -2'); 
xlabel('input, x'); 
ylabel('Covariance');
```

<p align="center">
  <img src="/picture/2d_gp_posterior_cov(2).png"/>   
  그림 (6) GP Posterior Covariance 2D 분포
</p> 

입력이 -2일때를 기준으로 2차원의 분포를 확인해보자.
$x=-2$일 때, $x$가 -2에 가까운 input들과 공분산이 높은 것을 알 수 있다.
다른 먼 입력 값보다 가까운 입력 값과 유사한 결과를 가질 확률이 높다는 것을 알 수 있다.
