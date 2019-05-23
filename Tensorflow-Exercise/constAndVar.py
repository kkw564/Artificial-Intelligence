import tensorflow as tf

# 상수는 constant() 함수를 이용해 정의 가능하며 변하지 않는 숫자를 의미한다.
a = tf.constant(1) # 하나의 텐서 자료형을 a에 삽입
b = tf.constant(2)
c = tf.add(a,b)

sess = tf.Session() # 텐서 자료형을 계산 할 수 있는 session을 생성
print(sess.run(c)) # 세션 객체에서 c를 계산
print(c) # 텐서 자료형 형태가 나타남
'''
3
Tensor("Add_1:0", shape=(), dtype=int32)
'''

a = tf.Variable(5)
b = tf.Variable(3)
c = tf.multiply(a,b)

init = tf.global_variables_initializer() # 텐서 플로우에서 변수를 만들면 항상 초기화를 해야한다
sess.run(init)# 세션 객체를 통해 init을 돌리면 된다
sess.run(c)
print(sess.run(c))
'''
15
'''

a = tf.Variable(15)
sess.run(c) # 초기화를 안했기에 값이 그대로 15가 나옴
'''
15
'''

a = tf.Variable(15)
c = a*b
init = tf.global_variables_initializer() # 변수 초기화 후 다시 진행 
sess.run(init)
sess.run(c)
'''
45
'''