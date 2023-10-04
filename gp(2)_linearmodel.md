Gaussian Process(2) - The Standard Linear Model
=============

해당 글의 전체적인 내용은 책 Gaussian Process for Machine Learning (Carl Edward Rasmussen)에서 가져왔다.  
이전 장에서 설명했던 베이즈 이론을 기반으로 분포를 시각적으로 표현해볼 것이다.
코드는 Matlab으로 이루어져 있다.

## The Standard Linear Model Regression

우리가 추측하려는 모델은 다음과 같다.

$$ f(\mathbf{x}) = \mathbf{x}^T \mathbf{w}=w_1 + w_2x \qquad (1)$$

$\mathbf{x}$는 입력값으로, 벡터이다. $\mathbf{w}$또한 벡터이며 입력에 대한 가중치인 weight이다.   
몇 개의 관찰된 데이터를 이용해서 식 (1)의 $w_1, w_2$를 추론해보는 것이다.  
추론하는 방법은 여러가지가 있지만 그 중에서 우리는 베이즈 이론을 이용해서 추론할 것이다.  

***

### 1. 데이터 및 변수

관찰된 데이터는 가우시안 분포를 따르는 노이즈가 추가되었다고 **가정**하자.  
노이즈를 가정하는 이유는 실제 데이터라고 생각하기 위해서다.  
(실제 세상에서 측정되는 데이터는 대부분 노이즈가 포함되어 있다.)  
노이즈가 포함된 관찰 데이터는 다음과 같이 나타낼 수 있다.  

$$y=f(\mathbf{x}) + \epsilon \qquad (2)$$

노이즈는 다음과 같이 정규분포를 따른다.

$$\epsilon \sim \mathcal{N}(0, \sigma^2_n) \qquad (3)$$

예시로 드는 데이터는 다음과 같다.

$$x = [-5, 2, 5], \quad y = [-6, -1, 4] \qquad (4)$$

(4)의 $y$ 데이터는 겉으로 보기에는 깔끔해보이지만, 노이즈가 포함되어 있다고 생각하면 된다.  

그래프를 그리기 전에 parameter인 weight와 데이터를 정리해보자.

$$ \mathbf{w}=[w_1, w_2]^T, \quad X= \begin{bmatrix} 1 & -5 \\ 1 & 2 \\ 1 & 5\end{bmatrix}, \quad \mathbf{y}=y=[-6, -1, 4]^T \qquad (5)$$

여기서 $\mathbf{x}$에 1열이 모두 1이어야 하는 이유는 선형 모델인 $y$의 $w_1$에는 데이터 $\mathbf{x}$와 상관없이 1이 곱해져 있기 때문이다.  

우선 Prior Probability인 $p(\mathbf{w})$를 그려보자.

Prior Probability는 변수가 $w_1, w_2$로 두 개이다.   
두 변수는 다음과 같은 간단한 정규분포를 따른다고 가정하자.    

$$p(\mathbf{w}) \sim (\mathbf{0}, I) \qquad (6)$$

이변수 정규분포이므로 평균과 분산 모두 행렬로 나타내어진다.

***

### 2. Multivariate Gaussian Distribution of Weights
Matlab에서 이변수 정규분포를 그리려면 내장함수를 이용하면 된다.    
코드는 아래와 같다.    

```matlab
clc; 
clear all; 
close all; 

% Multivariate Normal Distribution 
mu = [0 0];              % Mean of Noise 
Sigma = [1 0; 0 1];      % Covariance of Noise 
w1 = -2:0.1:2;           % Range of w1 
w2 = -2:0.1:2;           % Range of w2 
[W1,W2] = meshgrid(w1,w2); 
W = [W1(:) W2(:)]; 

y = mvnpdf(W,mu,Sigma); 
y = reshape(y,length(w1),length(w2)); 

% Plot Distribution 
surf(w1,w2,y); 
caxis([min(y(:))-0.5*range(y(:)),max(y(:))]); 
xlabel('w1');
ylabel('w2'); 
zlabel('Probability Density');
```

<p align="center">
  <img src="/picture/multivariate_dist.png">
  그림 (1) 서로 독립적인 정규분포를 따르는 weight 분포
</p>

$w_1, w_2$가 각각 독립적으로 정규분포를 따르게 설정한 것을 그림 (1)로 확인할 수 있다.

***

### 3. Likelihood
이전 장에서 설명했던 Likelihood 수식을 다시 가져오자.    
$$p(\mathbf{y}|X, \mathbf{w})=\frac{1}{(2\pi\sigma_n^2)^{n/2}}exp(-\frac{1}{2\sigma_n^2}|\mathbf{y}-X^T \mathbf{w}|^2)=\mathcal{N}(X^T\mathbf{w}, \sigma_n^2I) \qquad (7)$$

해당 수식을 이용해서 코드를 작성하면 아래와 같다.    
```matlab
clear all; % Clear prior code's parameters data 

% data list 
x_data_1 = [1, 1, 1]; 
x_data_2 = [-5, 2, 5]; 
y_data = [-6, -1, 4]; 

X = [x_data_1; x_data_2]; % Combine x data 1 and 2 

sigma_noise = 1; % Variance of Noise 
n = length(x_data_1); % The number of data set 

% Likelihood 
w1 = -3:0.05:2;   % Range of w1 
w2 = -3:0.05:2;   % Range of w2 

[W1, W2] = meshgrid(w1, w2); 
W = [W1(:), W2(:)]; 

% Likelihood Probability Density function 
y = (1/(2*pi*sigma_noise^2)).^(-n/2)*exp((-1/(2*sigma_noise^2))*vecnorm(y_data'-X'*W').^2); 
y = reshape(y, length(w1), length(w2)); 

% Plot Distribution 
figure(2); 
surf(W1,W2,y); 
caxis([min(y(:))-0.5*range(y(:)),max(y(:))]); 
axis([-3 2 -3 2]); 
xlabel('w1');
ylabel('w2'); 
zlabel('Probability Density');
```

분포를 그릴 때 정규분포 내장함수(mvnpdf)를 사용해도 되긴 하지만 위 코드는 수식으로 직접 작성되었다.   
코드에 의한 Likelihood 분포는 아래 그림과 같다.   
우선 평면 상에서 색깔의 분포를 확인해보자.

(평면 Likelihood 그림 (2) 추가)

노란색에 가까울 수록 값이 크고, 파란색에 가까울 수록 값이 작다.   
위 분포를 기울여서 3차원으로 보면 아래와 같다.

(입체 Likelihood 그림 (3) 추가)

이전 장에서 언급했던 것처럼 Likelihood 값이 높을 수록 데이터가 추출되었을 확률이 높은 것으로 본다.   
높은 Likelihood 값을 갖도록 하는 $w_1, w_2$를 weight로 가지는 함수가 데이터에 적합하다는 것이다.   

그림 (2, 3)에 따르면 Likelihood가 가장 높을 때의 weight는 $w_1 = -1.65, w_2=0.95$이다.   
즉, 우리가 추정하는 Linear Model은 $f(x)=-1.65+0.95x$의 형태가 주어진 데이터에 적합한 모델일 확률이 가장 높다는 것이다.   
여기서 Linear Model의 parameter가 확률 분포로 나타나는 것은 $y$ 데이터가 실제 값이 아닌 Noise가 포함되어 있다고 가정되었기 때문이다.   
이 때문에 Noise의 분산에 따라서 parameter 분포의 형태도 충분히 달라질 수 있다.   

***

### 4. Posterior Probability Distribution
이번엔 Posterior Probability를 분포로 그려보자.
Posterior Probability는 Likelihood와 Prior Probability의 곱이다.   
즉, 함수와 데이터의 적합성(Likelihood)과 데이터가 관찰되기 전 확률(Prior)의 곱이다.    
상수 성분을 모두 제거하고 변수에 대해서만 함수를 가져오면 간단하게 분포를 나타낼 수 있다.    
아래 식과 같이 특정 함수에 비례하는 것으로 Posterior 분포를 표현할 수 있다.   

$$p(\mathbf{w}|X, \mathbf{y})\propto exp(-\frac{1}{2}(\mathbf{w}- \mathbf{\bar{w}})^T(\frac{1}{\sigma^2_n}XX^T+\Sigma^{-1}_p)(\mathbf{w}- \mathbf{\bar{w}})) \qquad (8)$$

여기서 $\bar{\mathbf{w}}$는 아래와 같다.

$$\mathbf{\bar{w}}=\sigma^{-2}_n(\sigma^{-2}_nXX^T+\Sigma^{-1}_p)^{-1}X\mathbf{y} \qquad (9)$$

분포를 그리는 코드는 아래와 같다.   
```matlab
clear all; % Clear prior code's parameters data 

% Data list 
x_data_1 = [1, 1, 1]; 
x_data_2 = [-5, 2, 5]; 
y_data = [-6, -1, 4]; 
X = [x_data_1; x_data_2]; % Combine x data 1 and 2 

% Parameters 
sigma_noise = 1;      % Variance of Noise 
n = length(x_data_1); % The number of data set 
cov_mat = eye(2); % Covariance matrix of prior distribution 

w1 = -3:0.05:2; % Range of w1 
w2 = -3:0.05:2; % Range of w2 

% w bar 
w_bar = sigma_noise^(-2)*inv((sigma_noise^(-2)*X*X' + inv(cov_mat)))*X*y_data'; 
Y = zeros(1, length(w1)*length(w2)); 
for i = 1:length(w1) 
	for j = 1:length(w2)
		W_ = [w1(i), w2(j)]; 
		% Posterior Distribution function
		y = exp(-0.5*(W_'-w_bar)'*(sigma_noise^(-2)*X*X' + inv(cov_mat))*(W_'-w_bar));  
		Y((i-1)*length(w1)+j) = y; 
	end 
end 

% Plot Distribution 
figure(3); 
Y = reshape(Y, length(w1), length(w2)); 
surf(w1,w2,Y); 
axis([-3 2 -3 2]); 
xlabel('w1'); 
ylabel('w2'); 
zlabel('Posterior Distribution');
```

평면 상의 분포를 그리면 아래 그림과 같다.   
(평면 Posterior 그림(4) 추가)
이것 또한 노란색일 수록 확률 값이 높고, 파란색일 수록 확률 값이 낮다.

위 분포를 기울여서 3차원으로 보면 아래 그림과 같다.   

(입체 Posterior 그림(5) 추가)

위 그림을 통해서 적절한 weight를 추정하면 $w_1 = -1.2, w_2 = 0.9$ 이다.   
즉, Linear Model은 $f(x)=-1.2+0.9x$의 형태가 데이터에 적합한 모델일 확률이 가장 높다는 것이다.   
$w_1$는 이전에 -1.65였던 것이 -1.2로 바뀌었고, $w_2$는 0.95였던 것이 0.9로 바뀌었다.   
(이것에 대한 분석은 추후에.. 개념이 덜 잡힘.)

***

### 5. Linear Model Regression
마지막으로 Posterior 분포를 이용해서 새로운 데이터에 대한 결과를 추정하는 Linear Model의 형태를 그려보자.   
코드는 아래와 같다.
```matlab
clear all; % Clear prior code's parameters data 

% Variables 
x_data_1 = [1, 1, 1];     % Observed x1 
x_data_2 = [-5, 2, 5];    % Observed x2 
y_data = [-6, -1, 4];     % Observed y 
X = [x_data_1; x_data_2]; % Combine x data 1 and 2 

x = -5:0.1:5;    % New observed x 
new_x = ones(2, length(x)); 
new_x(2, :) = x; 

% Parameters 
w1 = -1.2; 
w2 = 0.9; 
sigma_noise = 1; 
Sigma = [1 0; 0 1]; % Covariance of Noise 
A = sigma_noise^(-2)*X*X' + inv(Sigma); y = w1 + w2*x; % Linear Model 

mean_of_post = zeros(1, length(x)); 
var_of_post = zeros(2, length(x)); 

for i=1:length(x) 
	mean_of_posterior = sigma_noise^(-2)*new_x(:, i)'*inv(A)*X*y_data'; 
	var_of_posterior = new_x(:, i)'*inv(A)*new_x(:, i); 
	var_of_95_1 = mean_of_posterior - 1.96*var_of_posterior/sqrt(3); 
	var_of_95_2 = mean_of_posterior + 1.96*var_of_posterior/sqrt(3); 
	mean_of_post(i) = mean_of_posterior; 
	var_of_post(1, i) = var_of_95_1; 
	var_of_post(2, i) = var_of_95_2; 
end 

% Plot Linear Regression
figure(4); 
plot(x, y, 'LineWidth', 2); h
old on; 
plot(x, mean_of_post, 'LineWidth', 3); 
plot(x, var_of_post(1, :), 'LineWidth', 3); 
plot(x, var_of_post(2, :), 'LineWidth', 3); 
legend('Linear Model', 'Mean of predictive distribution', 'Variance -(95%)', 'Variance +(95%)', 'Location', 'northwest');
```

위 코드를 이용해서 분포를 그리면 아래와 같다.

(Linear Regression 그림 (6) 추가)

파란 선은 Posterior 분포를 이용해서 구했던 $w_1, w_2$를 적용한 Linear Model을 그린 것이다.   
빨간 선은 기존의 데이터를 기반으로 새로운 데이터(-5~5)가 입력되었을 때 그에 대한 결과의 평균 값이다.    
노란 선과 보라 선은 추정하는 분포의 95% 신뢰구간의 경계를 나타내었다.

이 글의 예시는 데이터가 3쌍으로 매우 적고, parameter가 2개로 매우 간단한 선형 모델을 기반으로 추정하였다.   
다음 장에 진행할 Gaussian Process는 위 추정과는 다르게 특정되는 parameter가 없다.   
학습한 데이터를 기준으로 새로운 데이터와의 연관성을 체크해서 분포로 나타내는 추정이다.
