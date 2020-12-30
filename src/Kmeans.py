#!/usr/bin/python

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv


data_file = open("../dataset/normalized.csv", "r")
data_reader = csv.reader(data_file, delimiter=',')
features = []
tags = []
for row in data_reader:
    features.append([float(x) for x in row[1:]])
    tags.append(int(row[0])-1)

model = KMeans(n_clusters=3, random_state=0)
kmeans = model.fit(features)
pre = model.fit_predict(features)
sse = kmeans.inertia_
centers = kmeans.cluster_centers_
tran = [0, 2, 1]
result = sum([tags[i] == tran[pre[i]] for i in range(len(tags))])
accuracy = result / len(tags)

X0 = []
X1 = []
X2 = []
Y0 = []
Y1 = []
Y2 = []
d1 = 0
d2 = 11

for i in range(len(pre)):
    if pre[i] == 0:
        X0.append(features[i][d1])
        Y0.append(features[i][d2])
    elif pre[i] == 1:
        X1.append(features[i][d1])
        Y1.append(features[i][d2])
    elif pre[i] == 2:
        X2.append(features[i][d1])
        Y2.append(features[i][d2])

plt.figure()
plt.plot(X0, Y0, 'bo', X1, Y1, 'yo', X2, Y2, 'go')
plt.plot(kmeans.cluster_centers_[:, d1], kmeans.cluster_centers_[:, d2], 'ro')
plt.title('SSE: ' + str(sse) + '   ACC: ' + str(accuracy))
plt.xlabel('dimension  ' + str(d1))
plt.ylabel('dimension  ' + str(d2))
plt.show()
