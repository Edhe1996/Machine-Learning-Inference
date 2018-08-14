import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread

def add_gaussian_noise(im, prop, varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N], im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im)
    im2[index] += e[index]
    return im2


def add_saltnpeppar_noise(im, prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N], im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2


prop1 = 0.7
prop2 = 0.1
varSigma = 0.1
im = imread('image/pug.jpg')
im = im/255
fig = plt.figure()
#ax = fig.add_subplot(141)
#ax.imshow(im, cmap='gray')
im2 = add_gaussian_noise(im, prop1, varSigma)
#ax2 = fig.add_subplot(142)
#ax2 = fig.add_subplot(121)
#ax2.imshow(im2, cmap='gray')
im3 = add_saltnpeppar_noise(im, prop2)
#ax3 = fig.add_subplot(143)
#ax3.imshow(im3, cmap='gray')

y = np.copy(im2)

M, N = np.shape(y)
wei = 5
W = 0.1

maxiteration = 5
x = np.ones((M, N))
mu = np.ones((M, N))
new_mu = np.ones((M, N))
q = np.zeros((M, N))

for i in range(M):
    for j in range(N):
        y[i, j] = y[i, j] * 2 - 1
print(y)


def neighbours(i, j, M, N, size=4):
    if size == 4:
        if i == 0 and j == 0:
            n = [(0, 1), (1, 0)]
        elif i == 0 and j == N - 1:
            n = [(0, N - 2), (1, N - 1)]
        elif i == M - 1 and j == 0:
            n = [(M - 1, 1), (M - 2, 0)]
        elif i == M - 1 and j == N - 1:
            n = [(M - 1, N - 2), (M - 2, N - 1)]
        elif i == 0:
            n = [(0, j - 1), (0, j + 1), (1, j)]
        elif i == M - 1:
            n = [(M - 1, j - 1), (M - 1, j + 1), (M - 2, j)]
        elif j == 0:
            n = [(i - 1, 0), (i + 1, 0), (i, 1)]
        elif j == N - 1:
            n = [(i - 1, N - 1), (i + 1, N - 1), (i, N - 2)]
        else:
            n = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        return n
    if size == 8:
        if i == 0 and j == 0:
            n = [(0, 1), (1, 0), (1, 1)]
        elif i == 0 and j == N-1:
            n = [(0, N-2), (1, N-1), (1, N-2)]
        elif i == M-1 and j == 0:
            n = [(M-1, 1), (M-2, 0), (M-2, 1)]
        elif i == M-1 and j == N-1:
            n = [(M-1, N-2), (M-2, N-1), (M-2, N-2)]
        elif i == 0:
            n = [(0, j-1), (0, j+1), (1, j-1), (1, j), (1, j+1)]
        elif i == M-1:
            n = [(M-1, j-1), (M-1, j+1), (M-2, j-1), (M-2, j), (M-2, j+1)]
        elif j == 0:
            n = [(i-1, 0), (i+1, 0), (i-1, 1), (i, 1), (i+1, 1)]
        elif j == N-1:
            n = [(i-1, N-1), (i+1, N-1), (i-1, N-2), (i, N-2), (i+1, N-2)]
        else:
            n = [(i-1, j-1), (i-1, j), (i-1, j+1), (i, j-1), (i, j+1), (i+1, j-1), (i+1, j), (i+1, j+1)]
        return n


def approximation_distribution(i, j, mmu):
    m = 0
    n = neighbours(i, j, M, N, size=8)
    for itt in range(len(n)):
        m = m + W * mmu[n[itt]]
    if x[i, j] == 1:
        return 1 / (1 + np.exp(-2 * (m + L(i, j, 1) / 2 - L(i, j, -1) / 2)))
    else:
        return 1 / (1 + np.exp(2 * (m + L(i, j, 1) / 2 - L(i, j, -1) / 2)))


def L(i, j, x):
    return x * y[i, j] * wei


def updatemu(i, j, mmu):
    m = 0
    n = neighbours(i, j, M, N, size=8)
    for itt in range(len(n)):
        m = m + W * mmu[n[itt]]
    a = m + L(i, j, 1) / 2 - L(i, j, -1) / 2
    return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))


for itr in range(maxiteration):
    old_mu = mu
    for i in range(M):
        for j in range(N):
            q[i, j] = approximation_distribution(i, j, old_mu)

    for i in range(M):
        for j in range(N):
            new_mu[i, j] = updatemu(i, j, old_mu)

    mu = np.copy(new_mu)

for i in range(M):
    for j in range(N):
        uni = np.random.uniform(0.0, 1.0, None)
        if q[i, j] > uni:
            x[i, j] = 1
        else:
            x[i, j] = 0


ax4 = fig.add_subplot(111)
ax4.imshow(q, cmap='gray')
plt.show()

