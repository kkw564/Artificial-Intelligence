import tensorflow as tf
xData = [1,2,3,4,5,6,7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]

W = tf.Variable(tf.random_uniform([1],-100,100)) # 가중치(weight)
b = tf.Variable(tf.random_uniform([1],-100,100)) # 편향(y 절편, bias)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32) # 실제값

H = W * X + b # 예측값
cost = tf.reduce_mean(tf.square(H -Y)) # (H - Y)^2의 평균

a = tf.Variable(0.01) # 경사하강법에서 스텝의 크기
optimizer = tf.train.GradientDescentOptimizer(a) # 경사하강법

train = optimizer.minimize(cost) # 비용함수 가장 적게 만드는 방향으로
init = tf.global_variables_initializer() # 변수 초기화

sess = tf.Session() # 세션 정의
sess.run(init) # 세션 초기화

# 가중치와 편향이 점점 수렴해가는 것을 볼 수 있다.
for i in range(5001):
  sess.run(train, feed_dict ={X: xData, Y: yData})
  if i % 500 == 0:
    print(i,sess.run(cost,feed_dict ={X: xData, Y: yData}), sess.run(W), sess.run(b))

print(sess.run(H, feed_dict ={X: [8]}))
