import random
import numpy as np


def hello(kwargs):
    # Every node sends "Hello" to its peers
    # Peers respond with "World!"
    print(kwargs)
    x = kwargs.get('x', 55)
    m = kwargs.get('m', "ttttt")
    print(f"{x}: hello {m}")


def say(model, **kwargs):
    model(kwargs)
    # kwargs.get('model', "default value")


# say(hello, m="world", x=33)

a = [{'A': 1, 'B': 2}, {'A': 3, 'B': 4}]


def f(**kwargs):
    for x in a:
        if x.get('A') == 1:
            x.update(kwargs)


f(B=333)
# print(a)

peer = random.choice(a)
# print(peer)
# a[0].update({'B': 88})
# print(a)
# print(peer)

# r = {'p1': {12: 12, 13: 13}, 'p2': {12: 12, 13: 13}, 'p3': {12: 12, 13: 13}}
#
# if 'p4' not in r:
#     print('hhhh')

# import random
#
# dictionary = [{'A': 1, 'B': 2}, {'A': 1, 'B': 2}, {'A': 1, 'B': 2}, {'A': 1, 'B': 2}]
# # dictionary = [1,2,3,4,5]
# print(dictionary)
# N = 3
# x = random.sample(dictionary, 3)
# print(x)

# import numpy as np
# a = "2.4"
# b = "4"
# x = np.sqrt(81)
# print(x)


# peers = [{'A': 1, 'B': 3, 'C': 11}, {'A': 2, 'B': 6, 'C': 22}, {'A': 1, 'B': 7, 'C': 33}, {'A': 4, 'B': 8, 'C': 44}]
#
# x = next((p for p in peers if p["A"] == 1 and p["B"] == 7), False)
# if x:
#     x.update({'C': 123, 'D': "Hello"})
# else:
#     peers.append({})
# print(x)

# from sklearn.datasets import load_breast_cancer
#
# data = load_breast_cancer()
#
# # Organize our data
# label_names = data['target_names']
# labels = data['target']
# feature_names = data['feature_names']
# features = data['data']
# print(len(features))


nodeCount = 4
maxVertices = nodeCount * (nodeCount - 1) / 2 + 1
vertexCount = np.random.randint(1, maxVertices)

vertices = {}
while len(vertices) < vertexCount:
    x = random.randint(1, nodeCount)
    y = random.randint(1, nodeCount)
    if x == y:
        continue
    # comment the following line if the graph is directed
    if y < x: x, y = y, x
    w = random.random()
    vertices[x, y] = w


# just for debug
# for (x, y), w in vertices.items():
#     print('Vertex from {} to {} with weight {}.'.format(x, y, w))

# N = 4
# a = np.random.randint(0, 2, (N, N))
# m = np.tril(a) + np.tril(a, -1).T
# np.fill_diagonal(m, 0)
#
# print(m)


class C(object):
    def __init__(self):
        self._x = 0

    @property
    def parameters(self):
        print('getting')
        return self._x

    @parameters.setter
    def parameters(self, value):
        print("setter")
        self._x = value


if __name__ == '__main__':
    c = C()
    # print(c.parameters)
    c.parameters = 10
    # print(c.parameters)

n = 7
for j in range(1, n):
    for i in range(1, n):
        if i < j:
            print(f"W({i}, {j})")