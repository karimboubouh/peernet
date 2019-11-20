import random


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

import numpy as np
a = "2.4"
b = "4"
x = np.sqrt(81)
print(x)
