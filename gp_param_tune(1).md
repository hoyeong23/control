GP Parameter Tuning(1) - Manual Method
=============

해당 글의 전체적인 내용은 책 Gaussian Process for Machine Learning (Carl Edward Rasmussen)에서 가져왔다.   
GP를 조금 더 완성도 높게 마무리 짓기 위해서는 Parameter의 Tuning이 필요하다.   
Parameter Tuning이라 하면, GP에 쓰이는 Kernel에 들어가는 Hyperparameter를 적절히 찾는 것이다.   

***

### 1. Maximum Likelihood(MLE)

이때까지 사용하였던 Exponential Kernel을 다시 살펴보자.   

$$k(\mathbf{x, x'}) = exp(-\frac{1}{2}|x-x'|^2) \qquad (1)$$

식 (1)에서 Length Scale인 $l$과 분산의 크기를 조절하는 Signal Variance인 $\sigma_f$을 추가하면 아래와 같다.   

$$k(\mathbf{x, x'}, \sigma_f, l) = \sigma_f^2exp(-\frac{1}{2l^2}|x-x'|^2) \qquad (2)$$

Gaussian Process를 이해하면서 식 (2)와 같은 Kernel의 역할이 중요하다는 것을 깨달았다.   
하지만, 식 (2)와 같은 Kernel에서는 $\sigma_f, l$이라는 변수 두 개가 있는데 이것을 어떻게 정해야 할까?   

여러가지 방법이 있겠지만 책에서 소개하기도 하고 널리 알려진 방법으로는 Maximum Likelihood라는 개념이 있다.   
GP 1장에서 Likelihood는 데이터와 분포 사이의 적합도라고 설명하였다.   
즉, 주어진 데이터에 대해서 Likelihood를 가장 크게 만들 수 있는 분포의 Parameter를 찾는 것이 목표가 되겠다.   

이전에 정의한 Likelihood는 식 (3)과 같다.   

$$p(\mathbf{y}|X, \mathbf{w})=\frac{1}{(2\pi\sigma_n^2)^{n/2}}exp(-\frac{1}{2\sigma_n^2}|\mathbf{y}-X^T \mathbf{w}|^2) \qquad (3)$$

보통 식을 최대화하기 위해서는 미분의 개념을 이용하는데, 식 (3)은 두 변수가 곱셈으로 연결되어 있어서 미분이 쉽지 않다.   
그래서 Log를 취해서 식 (4)와 같은 Log Likelihood(L)의 최대값을 찾도록 한다.   

$$L = logp(\mathbf{y}|X, \mathbf{w})=-\frac{1}{2}\mathbf{y}^TK^{-1}\mathbf{y}-\frac{1}{2}log|K|-\frac{n}{2}log2\pi  \qquad (4)$$

***

### 2. Plot Log Likelihood

가장 직관적인 첫 번째 방법을 소개하겠다.   
두 변수 $\sigma_f, l$에 대해서 Log Likelihood를 3차원 그래프로 그리고 직접 최대값을 찾는 방법이다.   
Matlab에 max 함수를 이용해서 바로 찾을 수도 있다.   

3차원 그래프를 그리는 코드는 다음과 같다.   
```matlab
clc; clear all; close all;

l = 0.1:0.1:3;
sigma_f = 0.1:0.1:3;

rng(1);

noise_mag = 0.05;

X_train = [-4, -3, -2.5, -1.2, -1, 0.2, 1.3, 2.3, 3.1, 4];
X_train = linspace(-4, 4, 10);
y_train = sin(X_train) + cos(X_train);
y_train = y_train + noise_mag*randn(1, length(X_train));
y_train = y_train';

log_likelihood = zeros(length(l), length(sigma_f));

for i = 1:length(l)
	for j = 1:length(sigma_f)
		K_y_val = zeros(length(X_train), length(X_train));
		
		% Calculate K_y Matrix
		for m = 1:length(X_train)
			for n = 1:length(X_train)
				K_y_val(m, n) = sigma_f(i)^2*exp(-0.5/(l(j)^2)*(X_train(m)-X_train(n))^2);
				if m == n
					K_y_val(m, n) = K_y_val(m, n) + noise_mag^2;
				end
			end
		end
		log_likelihood(i, j) = -0.5*y_train'*inv(K_y_val)*y_train -0.5*log(det(K_y_val));
	end
end

% Plot Distribution
surf(l,sigma_f,log_likelihood);
xlabel('length scale')
ylabel('sigma f')
zlabel('Likelihood Distribution')
zlim([-100 10]);

max_val = max(max(log_likelihood));

% Find the maximum value of log likelihood
for i = 1:length(l)
	for j = 1:length(sigma_f)
		if log_likelihood(j, i) == max_val;
			max_i = i;
			max_j = j;
		end
	end
end

l(max_i)
sigma_f(max_j)
```

<p align="center">
  <img src="/picture/3d_log_likelihood.png"/>
  
  그림 (1) 3D Log Likelihood 분포
</p>

Log Likelihood가 최대가 되는 $\sigma_f, l$을 수동으로 찾음.
- $\sigma_f = 1.4, l = 1.9$

이 방법에는 다음과 같은 몇 가지 문제점이 있다.   
- 관찰할 수 있는 범위가 한정적 -> 변수의 범위를 사용자가 설정하기 때문에 Global Maximum의 위치를 유추해야 함.
- 변수의 개수가 증가하면 육안으로 최대값을 찾기 어려움.

***

Likelihood의 최댓값을 찾기 위해서 나름의 수동적인 방법을 이용해보았다.   
이것보단 좀 더 편리?할 수 있는 방법을 다음 장에서 소개하고자 한다.   
Gradient Descent를 이용한 방법인데, 간단한 학습을 이용해 반자동적으로 Parameter 값을 찾을 수 있다.   
