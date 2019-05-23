from sklearn.cluster import KMeans # KMeans 라이브러리
import numpy as np # 연산 처리를 위해 필요
import pandas as pd # 데이터 포인트를 만들기 위해 사용
import matplotlib.pyplot as plt # 데이터 그래프화(시각화)
from random import *
%matplotlib inline

df = pd.DataFrame(columns=['x','y']) # x,y를 가지는 데이터 프레임 생성

X,Y = [],[]
for i in range(50): # 총 50개의 데이터 무작위 생성
    df.loc[i] = [randint(1,50), randint(1,50)]
    X.append(df.loc[i][0])
    Y.append(df.loc[i][1])
    
df.head(50) # 데이터를 표 형태로 나타내준다.

plt.title("K-means Example(using scatter plot)")
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(X, Y)
plt.show()

points = df.values # 데이터 프레임 값들을 numpy 객체로 초기화
kmeans = KMeans(n_clusters = 4).fit(points) # 총 4개의 클러스터 생성

centers = kmeans.cluster_centers_ # 클러스터의 중심값을 계산
labels = kmeans.labels_ # 각 데이터들이 속하는 클러스터 넘버링

print(centers)
print(labels)

df['cluster'] = labels
df.head(50) # df를 테이블화 시켜 보여준다.
# pht에 각 데이터를 직어준다. 이때 각 라벨에 맞게 색상을 부여받는다.
plt.scatter(df['x'], df['y'], c= labels, s = 50, cmap='viridis')
# plt에 중심점을 찍어준다. 이때 색은 검은색, 크기는 200, 투명도는 0.5이다.
plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5)