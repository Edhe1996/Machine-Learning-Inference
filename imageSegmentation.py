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


# y = np.copy(im3)
# row_y = imread('blackpuggray.jpg', mode='F')
color_y = imread('image/color.jpg')
y = cv2.imread('image/color.jpg', 0)
y = cv2.GaussianBlur(y, (25, 25), 0)

M, N = np.shape(y)
print(M, N)
wei = 1
W = 0.2

maxiteration = 5
x = np.ones((M, N))
mu = np.ones((M, N))
new_mu = np.ones((M, N))
q = np.zeros((M, N))


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


histogram = cv2.calcHist([y], [0], None, [256], [0, 256])

# Plot the normalized histogram
# plt.hist(histogram, np.arange(256))
plt.hist(y.ravel(), 256, [0, 256], normed=True)
plt.show()
img = plt.gcf()

img.savefig('hist.png', dpi=100)

hist_normalize = histogram.ravel()/histogram.max()

Q = hist_normalize.cumsum()
# print Q.shape

x_axis = np.arange(256)
mini = np.inf
thresh = -1
for i in range(1, 256):
    p1, p2 = np.hsplit(hist_normalize, [i])

    q1, q2 = Q[i], Q[255]-Q[i]

    b1, b2 = np.hsplit(x_axis, [i])

    m1, m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1, v2 = np.sum(((b1-m1)**2)*p1)/q1, np.sum(((b2-m2)**2)*p2)/q2

    fn = v1*q1 + v2*q2
    if fn < mini:
        mini = fn
        thresh = i

print(thresh)


def likelihood(i, j, x):
    if y[i, j] > thresh:
        return wei * x
    else:
        return - wei * x


def approximation_distribution(i, j, mmu):
    m = 0
    n = neighbours(i, j, M, N, size=8)
    for itt in range(len(n)):
        m = m + W * mmu[n[itt]]
    if x[i, j] == 1:
        return 1 / (1 + np.exp(-2 * (m + likelihood(i, j, 1) / 2 - likelihood(i, j, -1) / 2)))
    else:
        return 1 / (1 + np.exp(2 * (m + likelihood(i, j, 1) / 2 - likelihood(i, j, -1) / 2)))


def L(i, j, x):
    return x * y[i, j] * wei


def updatemu(i, j, mmu):
    m = 0
    n = neighbours(i, j, M, N, size=8)
    for itt in range(len(n)):
        m = m + W * mmu[n[itt]]
    a = m + likelihood(i, j, 1) / 2 - likelihood(i, j, -1) / 2
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
        if q[i, j] > 0.5:
            color_y[i, j] = 255


plt.imshow(color_y)
plt.show()


