# 📉  Performance Optimization of MLP for Regression Problem
<br/>
  
### 1. &nbsp; Research Objective <br/><br/>

- _This study aims to train and evaluate a multilayer perceptron model on the Diabetes 130-US hospitals for years 1999-2008 dataset. To design the optimal model, we referred to the method proposed in the paper "Diabetes Mellitus Diagnosis Using Artificial Neural Networks with Fewer Features". In this paper, the authors tested the number of hidden layers from 2 to 8 to optimize the number of hidden layers for the Pima Indians Diabetes Database dataset and found that 3 or 4 hidden layers were the optimal model structure._  <br/>

- _Since the dataset used in this study is similar to the dataset used in the paper, we limited the network structure to three hidden layers based on the experiments in the paper. However, it is not completely identical to the network structure presented in the paper.In this study, we increased the number of nodes in each layer by about 10 times than the number of nodes presented in the paper to increase the complexity of the model. In addition, we added batch regularization and dropout layers to solve internal covariate shift and overfitting problems. Furthermore, we conducted experiments based on various trial and error methods, such as variable selection techniques, scaling techniques, and initialization techniques, to solve multicollinearity problems and improve the model's performance. Through these efforts, we hope to improve performance of the model and prevented problems such as gradient vanishing._ <br/><br/><br/> 

### 2. &nbsp; Key Components of the Neural Network Model and Experimental Settings  <br/><br/>

- _Input Nodes & Output Nodes_<br/>

  - _As the model has 6 input variables and 1 target variable, the input/output nodes were configured accordingly._ <br/>
  
  - _For reference, the diabetes dataset (Diabetes 130-US hospitals for years 1999-2008 Data Set) has 10 characteristics, but we removed the characteristics of age and sex from the input variables because we thought they were variables to be excluded from the analysis._ <br/>
  
  - _In addition, to avoid multicollinearity issues due to highly correlated independent variables, we used a variable selection technique to remove two highly correlated variables._ <br/><br/>

- _Hidden Layers & Batch Normalization & Dropout_<br/>

  - _There are a total of 3 hidden layers, with 60 nodes in the 1st and 3rd hidden layers, and 120 nodes in the 2nd hidden layer._<br/>
  
  - _To prevent the internal covariate shift problem where the distribution of input data changes with each hidden layer, we used batch normalization layers._ <br/>
  
  - _In addition, we used dropout layers, which randomly drop out some neurons during the learning process, to prevent overfitting and improve the generalization performance of the model._ <br/><br/>

- _Activation Function: ReLU (Rectified Linear Unit)_ <br/>

  - _I set the activation function to the ReLU function, which is a non-linear function that outputs 0 when the input value is less than 0, and outputs the input value itself when it is greater than 0._

  - _Also, depending on how you initialize the hidden layer weights when using the ReLU activation function, the gradient may disappear during the model training process. To avoid this problem, we used the He initialization technique, which is a neural network initialization method that takes nonlinearity into account like the ReLU activation function._<br/><br/>
  
- _Optimization Algorithm : Adam (Adaptive Moment Estimation)_ <br/>

  - _I used the Adam (Adaptive Moment Estimation) optimization algorithm, which combines the advantages of Momentum, which adjusts the learning rate considering the direction of the gradient, and RMSProp, which adjusts the learning rate considering the magnitude of the gradient._<br/><br/>

- _Cost Function : MSE (Mean Squared Error)_ <br/>

  - _I used the mean square error (MSE), which calculates the average of the squared differences between the predicted and actual values._<br/>

  - _MSE has the advantage of being easy to converge to the optimal value because the shift changes differently as you get closer to the optimal value, but it also has the disadvantage of being highly influenced by outliers._ <br/><br/>

- _Evaluation Metric: MAPE (Mean Absolute Percentage Error)_ <br/>

  - _I used MAPE (Mean Absolute Percentage Error), which calculates the mean absolute percentage difference between predicted values and actual values._<br/>

  - _MAPE is one of the most commonly used metrics in regression analysis and is characterized by the fact that the larger the difference between predicted and actual values, the larger the value of MAPE. In other words, the better the model predicts, the smaller the value of MAPE will be._ <br/><br/>


- _Maximum Number of Learning Iterations : 100_ <br/>

  - _In this experiment, the model is trained by iterating up to a maximum of 100 times._<br/>
  
  - _The number of iterations during training affects the speed and accuracy of the model, and I, as the researcher conducting the experiment, have set the number of iterations based on my experience of tuning deep learning models._<br/>
  
  - _Setting the number of iterations too low can result in underfitting, where the model is not trained enough, while setting it too high can result in overfitting._<br/> 
  
  - _Therefore, I have chosen what I believe to be the most appropriate number of iterations._ <br/> <br/> <br/> 

### 3. &nbsp; Data Preprocessing and Analysis <br/><br/>

- _**Package Settings**_ <br/> 
  
  ```
  # MLP의 회귀분석을 위해서 sklearn의 MLPRegressor 모듈을 사용
  from sklearn.neural_network import MLPRegressor
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_absolute_percentage_error
  import numpy as np
  import matplotlib.pyplot as plt
  ```

- _**Package Settings**_ <br/> 
  
  ```
  # MLP의 회귀분석을 위해서 sklearn의 MLPRegressor 모듈을 사용
  from sklearn.neural_network import MLPRegressor
  from sklearn.preprocessing import MinMaxScaler
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_absolute_percentage_error
  import numpy as np
  import matplotlib.pyplot as plt
  ```  
- _**Data Preparation**_ <br/> 
  
  ```
  # 입력값 : X 변수값의 범위는 1~100 사이의 정수로 지정
  # 출력값 : X의 값에 대응하는 Y값 계산
  x = np.array(range(1, 101))
  y = np.sqrt(x)

  print(f"x 값의 범위 : {np.min(x)} ~ {np.max(x)}")
  print(f"y 값의 범위 : {int(np.min(y))} ~ {int(np.max(y))}")
  ```
  
- _**Exploratory Data Analysis (EDA)**_ <br/> 
  
  ```
  # X와 Y간의 관계를 그래프로 그려보면, X 값이 증가할 때 Y 값도 점차 증가하지만, 
  # 그 증가 속도는 X 값이 증가함에 따라 감소하는 비선형 곡선
  # 이러한 형태를 지니는 완만하게 증가하는 곡선을 로그 곡선(log curve)이라고 함
  plt.figure(figsize=(12, 3))
  plt.scatter(x, y, color='g')
  plt.title('y=sqrt(x)')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend(['Actual Data']) 
  plt.show()
  ```
  <img src="https://github.com/qortmdgh4141/Comparing-Performance-of-Shallow-and-Deep-MLP-for-Regression-Analysis/blob/main/image/actual_data_graph.png?raw=true"  width="970" > <br/>
   
- _**Splitting Data**_ <br/>  
  
  ```
  # 학습용과 테스트용 데이터를 7:3으로 분리
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
    
  print(f"1) 학습용 입력 데이터(X) 형상 : {x_train.shape}")
  print(f"2) 학습용 정답 데이터(Y) 형상 : {y_train.shape}")
  print(f"3) 평가용 입력 데이터(X) 형상 : {x_test.shape}")
  print(f"4) 평가용 정답 데이터(Y) 형상 : {x_test.shape}")
  ```
  
- _**Feature Scaling**_ <br/> 
  
  ```
  # 데이터 세트의 차원을 1열 구조로 변형
  x_train = x_train.reshape(-1, 1)
  x_test = x_test.reshape(-1, 1)
  y_train = y_train.reshape(-1, 1)
  y_test = y_test.reshape(-1, 1)

  print(f"1) 학습용 입력 데이터(X) 형상 : {x_train.shape}")
  print(f"2) 학습용 정답 데이터(Y) 형상 : {y_train.shape}")
  print(f"3) 평가용 입력 데이터(X) 형상 : {x_test.shape}")
  print(f"4) 평가용 정답 데이터(Y) 형상 : {x_test.shape}")  
     
  ```
  
  ```
  # 최솟값은 0, 최댓값은 1이 되도록 학습 데이터에 대해 정규화
  # 피처 스케일링 : 학습 데이터의 입력 값
  scalerX = MinMaxScaler()
  scalerX.fit(x_train)
  x_train_norm = scalerX.transform(x_train)

  # 피처 스케일링 : 학습 데이터의 출력 값
  scalerY = MinMaxScaler()
  scalerY.fit(y_train)
  y_train_norm = scalerY.transform(y_train)

  # 피처 스케일링 : 테스트 데이터의 입출력 값
  x_test_norm = scalerX.transform(x_test)
  y_test_norm = scalerY.transform(y_test)
  ```
  <br/> 

### 4. &nbsp; Training and Testing MLP Models <br/><br/>

- _**Shallow Model**_ <br/> 
  
  ```
  """
  1. 입출력 노드 : 1개
     - 학습 시에 입출력 값이 각각 1개이기 때문에, 그에 대응하는 입출력 노드도 각각 1개씩 존재

  2. 은닉층 개수 / 노드 : 1개 / 2개
      - 총 1개의 은닉층이 존재하며 각 은닉층에는 2개의 노드가 존재

  3. 활성화 함수 :  로지스틱 함수(시그모이드 함수)
     - 0과 1 사이의 값을 출력하는 S자 형태의 비선형 함수인 로지스틱 함수(시그모이드 함수)로 설정

  4. 최적화 알고리즘 
     - 작은 데이터셋에서는 일반적으로 수렴 속도가 빠르고, 
     대부분의 경우 다른 최적화 알고리즘보다 더 나은 성능을 보이는 L-BFGS (Limited-memory BFGS)를 사용

  5. 비용함수 : 
     - 예측값과 실제값의 차이를 제곱한 값의 평균을 계산함으로써, 
       예측값과 실제값 사이의 오차를 잘 나타내는 MSE를 사용

  6. 최대 학습 반복 횟수 : 100회
  """

  # 모형화
  shallow_model = MLPRegressor(hidden_layer_sizes=(2,), activation='logistic'
                                , solver='lbfgs', max_iter=100)

  # 학습
  shallow_model.fit(x_train_norm, y_train_norm)
  ```
  
  ```
  # 예측
  y_pred = shallow_model.predict(x_test_norm)
  # 예측 값의 데이터를 1열 구조로 변형
  y_pred = y_pred.reshape(-1,1)
  # 예측 값을 실제 스케일로 역변환
  y_pred_inverse = scalerY.inverse_transform(y_pred)
  print(f"Shallow Model - 예측 값의 범위 : {np.min(y_pred_inverse)} ~ {np.max(y_pred_inverse)}")
  ```
  
  ```
  # 실제 값 대비 절대 오차의 평균 백분율(MAPE)로 정확도를 계산
  # MAPE(오차 측정) 값이 0에 가까울수록 모델의 예측 성능은 일반적으로 좋다고 판단
  shallow_model_mape = mean_absolute_percentage_error(y_test, y_pred_inverse)
  print(f"Shallow Model - MAPE : {shallow_model_mape:.2f}")
  ```
  
  ```
  # 1~100 사이의 X값에 대응하는 Y의 실제 값과 테스트 데이터로 예측한 값을 산포도로 출력하면, 
  # 예측 값이 실제 값과 비슷한 결과를 도출 
  # 실제 데이터의 분포
  plt.figure(figsize=(12, 3))
  plt.scatter(x, y, color='g')
  # 테스트 데이터의 분포
  plt.scatter(x_test, y_pred_inverse, color='r')

  plt.title('y=sqrt(x)')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend(['Actual Data', 'Predicted Data by Shallow Model']) 
  plt.show()
  ```
   <img src="https://github.com/qortmdgh4141/Comparing-Performance-of-Shallow-and-Deep-MLP-for-Regression-Analysis/blob/main/image/shallow_data_graph.png?raw=true"  width="960" > 
- _**Deep Model**_ <br/> 
  
  ```
  """
  1. 입출력 노드 : 1개
     - 학습 시에 입출력 값이 각각 1개이기 때문에, 그에 대응하는 입출력 노드도 각각 1개씩 존재

  2. 은닉층 개수 / 노드 : 4개 / 8개
      - 총 4개의 은닉층이 존재하며 각 은닉층에는 8개의 노드가 존재

  3. 활성화 함수 :  로지스틱 함수(시그모이드 함수)
     - 0과 1 사이의 값을 출력하는 S자 형태의 비선형 함수인 로지스틱 함수(시그모이드 함수)로 설정

  4. 최적화 알고리즘 
     - 작은 데이터셋에서는 일반적으로 수렴 속도가 빠르고, 
     대부분의 경우 다른 최적화 알고리즘보다 더 나은 성능을 보이는 L-BFGS (Limited-memory BFGS)를 사용

  5. 비용함수 : 
     - 예측값과 실제값의 차이를 제곱한 값의 평균을 계산함으로써, 
       예측값과 실제값 사이의 오차를 잘 나타내는 MSE를 사용

  6. 최대 학습 반복 횟수 : 100회
  """

  # 모형화
  deep_model = MLPRegressor(hidden_layer_sizes=(8, 8, 8, 8), activation='logistic'
                                , solver='lbfgs', max_iter=100)

  # 학습
  deep_model.fit(x_train_norm, y_train_norm)
  ```
  
  ```
  # 예측
  y_pred_extended = deep_model..predict(x_test_norm)
  # 예측 값의 데이터를 1열 구조로 변형
  y_pred_extended = y_pred_extended.reshape(-1,1)
  # 예측 값을 실제 스케일로 역변환
  y_pred_inversey_extended = scalerY.inverse_transform(y_pred_extended)
  print(f"Deep Model - 예측 값의 범위 : {np.min(y_pred_inversey_extended)} ~ {np.max(y_pred_inversey_extended)}")
  ```
  
  ```
  # 실제 값 대비 절대 오차의 평균 백분율(MAPE)로 정확도를 계산
  # MAPE(오차 측정) 값이 0에 가까울수록 모델의 예측 성능은 일반적으로 좋다고 판단
  Deep_model_mape = mean_absolute_percentage_error(y_test, y_pred_inversey_extended)
  print(f"Deep Model - MAPE : {shallow_model_mape:.2f}")
  ```
  
  ```
  # 1~100 사이의 X값에 대응하는 Y의 실제 값과 테스트 데이터로 예측한 값을 산포도로 출력하면, 
  # 예측 값이 실제 값과 비슷한 결과를 도출 
  # 실제 데이터의 분포
  plt.figure(figsize=(12, 3))
  plt.scatter(x, y, color='g')
  # 테스트 데이터의 분포
  plt.scatter(x_test, y_pred_inversey_extended, color='b')

  plt.title('y=sqrt(x)')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend(['Actual Data', 'Predicted Data by Deep Model']) 
  plt.show()
  ```
  <img src="https://github.com/qortmdgh4141/Comparing-Performance-of-Shallow-and-Deep-MLP-for-Regression-Analysis/blob/main/image/deep_data_graph.png?raw=true"  width="950" > 
  <br/>
  
### 5. &nbsp; Research Results  <br/><br/>
    
- _The purpose of this study was to train and evaluate a multi-layer perceptron regression model for the function y=sqrt(x) using integers from 1 to 100 based on the hypothesis that a simpler structure may perform better on a small and simple problem. At this time, the performance of the models was evaluated using a metric called MAPE (Mean Absolute Percentage Error), which measures the average percentage error between the predicted and actual values. Just to let you know, the closer the model's predicted values are to the actual values, the lower the MAPE value and the closer the MAPE value is to 0, the better the model's performance is considered to be._ <br/><br/><br/>

  ```
  def gradientbars(bars, cmap_list):
    # cmap 가중치 설정
    grad = np.atleast_2d(np.linspace(0,1,256)).T
    # 플롯 영역 재설정
    ax = bars[0].axes
    lim = ax.get_xlim()+ax.get_ylim()
    ax.axis(lim)
    # 각 막대에 색 입히기
    max = 0
    for i, bar in enumerate(bars):
        bar.set_facecolor("none")
        x,y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        ax.imshow(grad, extent=[x,x+w,y,y+h], aspect="auto", cmap=cmap_list[i])
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f' ==> {df.Mape[i]:.2f}', ha='left', va='center', fontsize=10, color='black')

  fig, ax = plt.subplots(figsize=(8,4))
  df = pd.DataFrame({'Model':['Shallow Model', 'Deep Model'], 'Mape':[shallow_model_mape, deep_model_mape]})
  cmap_color = ['viridis_r', 'YlOrRd']
  gradientbars(ax.barh(df.Model, df.Mape), cmap_color)

  plt.title(f"Comparison of MAPE values between shallow and deep models", fontsize=12)
  plt.xlabel('Mape', fontsize=10)
  plt.xlim([0, 0.4])
  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)
  plt.tight_layout()
  plt.show()
  ```
  <br/>
  

<img src="https://github.com/qortmdgh4141/Comparing-Performance-of-Shallow-and-Deep-MLP-for-Regression-Analysis/blob/main/image/shallow_deep_mape_graph.png?raw=true" width="930">

- _In this study, the MAPE value for the Deep Model was 0.35, while the MAPE value for the Shallow Model was 0.01, indicating that the performance of the Shallow Model was superior._ <br/><br/><br/>


  ```
  # 1~100 사이의 X값에 대응하는 Y의 실제 값과 테스트 데이터로 예측한 값을 산포도로 출력하면, 예측 값이 실제 값과 비슷한 결과를 도출 
  # 실제 데이터의 분포
  plt.figure(figsize=(8, 4))
  plt.scatter(x, y, color='g')
  # 테스트 데이터의 분포
  plt.scatter(x_test, y_pred_inverse, color='r')
  plt.scatter(x_test, y_pred_inverse_extended, color='b')

  plt.title('y=sqrt(x)')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend(['Actual Data', 'Predicted Data by Shallow Model', 'Predicted Data by Deep Model'],fontsize=8)
  plt.show()
  ```
  <br/>
  
<img src="https://github.com/qortmdgh4141/Comparing-Performance-of-Shallow-and-Deep-MLP-for-Regression-Analysis/blob/main/image/shallow_deep_data_graph.png?raw=true" width="940">

- _Furthermore, when we examine the scatter plot, most of the data predicted by the Shallow Model matches the actual data, while the Deep Model only predicts values between 6.3509680903363845 and 6.351427249440577._ <br/><br/>
  
- _The reason for these results is believed to be overfitting in the Deep Model due to the simple problem and small amount of data. When predicting simple problems or with small amounts of data, a Shallow Model that does not experience overfitting may perform better, and this implies that increasing the number of parameters or deepening the network structure does not always lead to good results._ <br/><br/>

- _Therefore, it is important to design the model appropriately based on the complexity of the problem, as well as the diversity and quantity of data, as Deep Model may not always guarantee better results in all situations._ <br/> <br/> <br/>


--------------------------
### 💻 S/W Development Environment
<p>
  <img src="https://img.shields.io/badge/Windows 10-0078D6?style=flat-square&logo=Windows&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google Colab-black?style=flat-square&logo=Google Colab&logoColor=yellow"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit learn-blue?style=flat-square&logo=scikitlearn&logoColor=F7931E"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=blue"/>
</p>

### 🚀 Machine Learning Model
<p>
  <img src="https://img.shields.io/badge/MLP-5C5543?style=flat-square?"/>
</p> 
