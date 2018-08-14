def gibbs(y, iteration, likelihood=lambda yt, xt: np.exp((5 * (yt * xt))), shuffled=False):
    y = y * 2 - 1
    print(y)
    x = [list(map(lambda yt: 1.0 if yt >= 0 else -1.0, y_t)) for y_t in y]
    x = np.array(x)
    x_temp = np.zeros(x.shape)
    R, C = y.shape
    index = []

    def prior(x_i, r, c):
        neighbers = neighbours(r, c, R, C)
        result = 0
        for neighb in neighbers:
            result += x_i * x[neighb]
        result *= 0.8
        return np.exp(result)

    for r in range(R):
        for c in range(C):
            index.append((r, c))
    if shuffled is True:
        shuffle(index)
    for i in range(iteration):
        for r, c in index:
            part1 = likelihood(y[r, c], 1) * prior(1, r, c)

            part2 = likelihood(y[r, c], 1) * prior(1, r, c) + likelihood(y[r, c], -1) * prior(-1, r, c)
            # print(part1, part2, likelihood(y[r, c], -1) * prior(-1, r, c))
            posterior = part1 / part2
            t = np.random.rand()
            x_temp[r, c] = 1 if posterior > t else -1
        x = x_temp.copy()
    return x


# def vairente_byes


def icm(y, iteration, n, b, h=0):
    y = y * 2 - 1
    print(y)

    def cal_energy(x, y, r, c):
        R, C = x.shape
        part1 = 0
        part2 = 0
        part3 = 0
        neighbs = neighbours(r, c, R, C)
        for neigh in neighbs:
            part1 += x[r, c] * x[neigh]
        part2 += x[r, c] * y[r, c]
        part3 += x[r, c]
        result = b * part1 + n * part2 - h * part3
        return result

    # init x
    x = y.copy()
    R, C = y.shape
    for i in range(iteration):
        for r in range(R):
            for c in range(C):
                x[r, c] = 1
                r1 = cal_energy(x, y, r, c)
                x[r, c] = -1
                r2 = cal_energy(x, y, r, c)
                if r1 > r2:
                    x[r, c] = 1
                else:
                    x[r, c] = -1
    return x
