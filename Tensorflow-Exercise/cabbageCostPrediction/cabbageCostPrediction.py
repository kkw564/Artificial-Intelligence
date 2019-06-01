import tensorflow as tf
import numpy as np

from pandas.io.parsers import read_csv

import os # Docker의 directory를 파악
print (os.getcwd())

'''
  /root
'''

model = tf.global_variables_initializer()
data = read_csv('pricedata.csv',sep=',')

xy = np.array(data, dtype=np.float32) # csv로 받은 data 파일을 np.array로 변환
'''
  [[ 2.0100100e+07 -4.9000001e+00 -1.1000000e+01  8.9999998e-01
    0.0000000e+00  2.1230000e+03]
  [ 2.0100102e+07 -3.0999999e+00 -5.5000000e+00  5.5000000e+00
    8.0000001e-01  2.1230000e+03]
  [ 2.0100104e+07 -2.9000001e+00 -6.9000001e+00  1.4000000e+00
    0.0000000e+00  2.1230000e+03]
  ...
  [ 2.0171228e+07  2.9000001e+00 -2.0999999e+00  8.0000000e+00
    0.0000000e+00  2.9010000e+03]
  [ 2.0171230e+07  2.9000001e+00 -1.6000000e+00  7.0999999e+00
    6.0000002e-01  2.9010000e+03]
  [ 2.0171232e+07  2.0999999e+00 -2.0000000e+00  5.8000002e+00
    4.0000001e-01  2.9010000e+03]]
'''

x_data = xy[:,1:-1] # 처음부터 가장 오른쪽 바로 직전까지 데이터
y_data = xy[:,[-1]] # 가장 오른쪽 데이터

X = tf.placeholder(tf.float32, shape=[None,4]) # 4개의 데이터를 담을 수 있도록
Y = tf.placeholder(tf.float32, shape=[None,1])
W = tf.Variable(tf.random_normal([4,1]), name="weight") # 가중치
b = tf.Variable(tf.random_normal([1]), name="bias") # 편향

hypothesis = tf.matmul(X,W) + b # 가설 식 세우기, 행렬의 곱 연산 이용
cost = tf.reduce_mean(tf.square(hypothesis - Y)) # 비용 함수 연산
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005) # 최적화 함수 설정, 학습률 0.000005
train = optimizer.minimize(cost)

sess = tf.Session() # 세션 생성 및 초기화
sess.run(tf.global_variables_initializer())


for step in range(100001):
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:
        print("#", step, " 손실 비용 : ", cost_)
        print("- 배추 가격 : ", hypo_[0]) # 첫번째 데이터를 통해 배추 가격 확인

'''
  # 0  손실 비용 :  2229339.8
  - 배추 가격 :  [2584.3118]
  # 500  손실 비용 :  2229087.0
  - 배추 가격 :  [2584.3462]
  # 1000  손실 비용 :  2228834.8
  - 배추 가격 :  [2584.381]
  # 1500  손실 비용 :  2228582.0
  - 배추 가격 :  [2584.4155]
  # 2000  손실 비용 :  2228329.8
  - 배추 가격 :  [2584.4504]
            ...
  # 99000  손실 비용 :  2182676.5
  - 배추 가격 :  [2590.584]
  # 99500  손실 비용 :  2182457.0
  - 배추 가격 :  [2590.6155]
  # 100000  손실 비용 :  2182237.2
  - 배추 가격 :  [2590.647]
'''


# 학습된 데이터 저장 시키기
saver = tf.train.Saver()
save_path = saver.save(sess, "./saved.cpkt")


# 학습된 트레이닝 데이터를 가져와서 직접 입력한 값에 대한 예측 값을 출력
saver = tf.train.Saver()
model = tf.global_variables_initializer()

avg_tmp = float(input('평균 온도 : '))
min_tmp = float(input('최저 온도 : '))
max_tmp = float(input('최고 온도 : '))
rain_fall = float(input('강수량 : '))

with tf.Session() as sess:
    sess.run(model)
    
    save_path = "./saved.cpkt" # 저장된 트레이닝 데이터
    saver.restore(sess, save_path) 
    
    data = ((avg_tmp, min_tmp, max_tmp, rain_fall),) # 기존 학습된 데이터가 2차원 배열이기에 2차원 배열 데이터를 만든다.
    arr = np.array(data, dtype=np.float32) # 사용자가 입력한 데이터를 토대로 초기화
    
    x_data = arr[0:4]
    dict = sess.run(hypothesis, feed_dict={X: x_data}) 
    
    print(dict[0]) # 데이터 하나만 들어갔으니 첫번째 데이터만