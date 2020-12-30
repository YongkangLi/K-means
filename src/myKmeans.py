#!/usr/bin/python

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import csv


data_file = open("../dataset/normalized.csv", "r")
data_reader = csv.reader(data_file, delimiter=',')
features = []
tags = []
for row in data_reader:
    features.append([float(x) for x in row[1:]])
    tags.append(int(row[0])-1)

centers = rd.sample(features, 3)

centers = np.array(centers)
features = np.array(features)

features = np.array(features)
centers = np.array(centers)

pre = np.array([-1 for i in range(len(features))])
times = 0
while True:
    for i in range(len(features)):
        min_dis = float(len(features))
        for j in range(len(centers)):
            dis = np.sum(np.square(features[i] - centers[j]))
            if dis < min_dis:
                min_dis = dis
                pre[i] = j
    s = np.zeros((len(centers), len(features[0])))
    count = np.zeros(len(centers))
    for i in range(len(pre)):
        if pre[i] == 0:
            s[0] += features[i]
            count[0] += 1
        elif pre[i] == 1:
            s[1] += features[i]
            count[1] += 1
        elif pre[i] == 2:
            s[2] += features[i]
            count[2] += 1

    if count[0]:
        s[0] /= count[0]
    else:
        s[0] = features[rd.randint(0, len(features))]
    if count[1]:
        s[1] /= count[1]
    else:
        s[1] = features[rd.randint(0, len(features))]
    if count[2]:
        s[2] /= count[2]
    else:
        s[2] = features[rd.randint(0, len(features))]

    if not (s - centers).any():
        break
    elif times >= 100000:
        break
    else:
        centers = s
        times += 1

c = np.zeros((len(centers), len(features[0])))
count = np.zeros(len(centers))
for i in range(len(pre)):
    if tags[i] == 0:
        s[0] += features[i]
        count[0] += 1
    elif tags[i] == 1:
        s[1] += features[i]
        count[1] += 1
    elif tags[i] == 2:
        s[2] += features[i]
        count[2] += 1
s[0] /= count[0]
s[1] /= count[1]
s[2] /= count[2]
mapping = [-1, -1, -1]
for i in range(3):
    min_dis = len(features)
    for j in range(3):
        dis = np.sum(np.square(s[i] - centers[j]))
        if dis < min_dis:
            min_dis = dis
            mapping[i] = j

sse = 0
for i in range(len(features)):
    sse += np.sum(np.square(features[i]-centers[pre[i]]))
result = sum([mapping[tags[i]] == pre[i] for i in range(len(tags))])
accuracy = result / len(tags)
pre = [2 - pre[i] for i in range(len(pre))]
print(accuracy)

X0 = []
X1 = []
X2 = []
Y0 = []
Y1 = []
Y2 = []
d1 = 5
d2 = 6

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
plt.plot(centers[:, d1], centers[:, d2], 'ro')
plt.title('SSE: ' + str(sse) + '   ACC: ' + str(accuracy))
plt.xlabel('dimension  ' + str(d1))
plt.ylabel('dimension  ' + str(d2))
plt.show()
