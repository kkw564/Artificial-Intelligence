#기본적인 numpy에서 볼 수 있는 형태

import numpy as np

tmp = [i for i in range(0,10)]

x = np.array(tmp)

print(x)

np.ndim(x) # 배열의 차원 수 확인 가능 현재 1차원이니 출력이 1로 나온다.

x.shape # 배열의 형상 반환 (10,)

y = np.array([[1,2],[2,3],[5,6]])

np.ndim(y) # 2차원이니 2 출력

y.shape # 3행 2열이니 (3,2) 튜플 출력



# 행렬의 곱

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])

c = np.dot(a,b) # 행렬 곱은 dot로 가능하다.
d = a * b # d는 행렬 곱이 아닌 각 행,열의 위치에 맞는 인덱스끼리 곱하는 과정이다.
print(c)
print(d)

np.ndim(c) # 2

c.shape # (2,2)


# 신경망에 행렬의 곱 적용

x = np.array([5,10])

w = np.array([[1,3,5],[2,4,6]])

print(w)

w.shape

y = np.dot(x,w)
print(y)