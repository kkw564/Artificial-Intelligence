'''
  placeholder은 다른 텐서를 할당하기 위해 사용
  실제로 값을 할당할 때는 피딩(Feeding)을 수행한다.

  텐서 자체는 다차원 배열과 같은 배열형태라 placeholder도 배열이 들어갈 수 있다.

  즉, placeholder은 학습데이터를 포함하는 변수라고 할 수 있다.

  placeholder의 형태
  tf.placeholder(dtype, shape, name)
  dtype : 플레이스 홀더에 저장되는 자료형을 의미
  shape : 배열의 차원을 의미
  name : 플레이스 홀더의 이름을 의미
'''

import tensorflow as tf

input = [1,2,3,4,5]
x = tf.placeholder(dtype=tf.float32) # x를 placeholder로 설정
y = x + 5 # 수식 설정

sess = tf.Session()
sess.run(y, feed_dict={x: input}) # y라는 수식에 x부분을 input으로 초기화해서 실행시키겠다는 의미
'''
array([ 6.,  7.,  8.,  9., 10.], dtype=float32)
'''

aScore = [88,99,11,22,33,44]
bScore = [66,77,88,99,12,55]

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
y = (a + b) / 2

sess = tf.Session()
sess.run(y, feed_dict={a: aScore, b: bScore})
'''
array([77. , 88. , 49.5, 60.5, 22.5, 49.5], dtype=float32)
'''