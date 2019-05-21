import tensorflow as tf

# 특정 값을 tf에서 상수로 생성할 수 있다.

a = tf.constant(1)
b = tf.constant(2)

# tf끼리 서로 더하거나 사칙연산을 수행 할 수 있다.

c = a + b
d = a * b

print(a)
print(b)
print(c)
print(d)

'''
  Tensor("Const:0", shape=(), dtype=int32)
  Tensor("Const_1:0", shape=(), dtype=int32)
  Tensor("add:0", shape=(), dtype=int32)
  Tensor("mul:0", shape=(), dtype=int32)
'''

# tensors(텐서)는 다차원 배열을 의미한다. 
# 이때 파이썬에서 사용하는 리스트를 텐서플로에 보내면 적절한 차원의 텐서로 변환한다.
 
m1 = tf.constant([[1.,2.],[3.,4.]])
m2 = tf.constant([[3.,4.],[5.,6.]])

m3 = m1 + m2
m4 = m1 * m2 # == m4 = tf.matmul(m1,m2)

# 텐서플로에서 실제 계산을 하기 위해서는 계산이 수행되는 세션이라는 공간을 만들어야한다.
sess = tf.Session()

ret = sess.run(m4)
print("ret :: ", ret)

'''
  ret ::  [[ 3.  8.]
  [15. 24.]]
'''
sess.close()

# 대화형 세션을 만들면 eval 메서드를 이용하여 어느공간에서도 계산할 수 있다.
sess = tf.InteractiveSession()

w = tf.Variable(0, name="weight")
print(w)

# 텐서플로에서 변수를 자동으로 초기화 하기 위해 아래와 같은 메서드를 이용한다.
# 이는 sess.run으로 호출할 수 있다.
init = tf.global_variables_initializer()
sess.run(init)
print(w.eval())

# 변수에 1을 더하면 1이 돼야 한다.
w = w.assign_add(1) # w += 1(Not recommended)

print(w.eval())

'''
  <tf.Variable 'weight_21:0' shape=() dtype=int32_ref>
  0
  1
'''