import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys


def fun(x, y):
    '''
    :param x:
    :param y:
    :return　関数の返り値:
     これが解くべき方程式の解である。
    '''

    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def ff(x, y):
    return - 8 * (np.pi ** 2) * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


def make_initial(n):
    xs = np.arange(1 / n, 1, 1 / n)
    ys = np.arange(1 / n, 1, 1 / n)
    xss, yss = np.mgrid[1 / n:1:1 / n, 1 / n:1:1 / n]
    fx0 = fun(0, ys)
    fx1 = fun(1, ys)
    fy0 = fun(xs, 0)
    fy1 = fun(xs, 1)
    return ff(xss, yss), fx0, fx1, fy0, fy1


def make_equation(n, f, fx0, fx1, fy0, fy1):
    d = (n - 1) ** 2
    subA = np.zeros((d, d))
    subA2 = np.zeros((d, d))
    for k in range(n - 1):
        if k * (n - 1) - 1 >= 0:
            subA[k * (n - 1) - 1][k * (n - 1)] = 1
            subA[k * (n - 1)][k * (n - 1) - 1] = 1
    for i in range(d - 1):
        subA2[i][i + 1] = 1
        subA2[i + 1][i] = 1
    for i in range(d - (n - 1)):
        subA2[i][i + n - 1] = 1
        subA2[i + n - 1][i] = 1
    A = np.identity(d) * (-4) - subA + subA2
    b = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        for j in range(n - 1):
            b[i][j] += f[i][j] / (n ** 2)
            if i == 0:
                b[i][j] -= fx0[j]
            if i == n - 2:
                b[i][j] -= fx1[j]
            if j == 0:
                b[i][j] -= fy0[i]
            if j == n - 2:
                b[i][j] -= fy1[i]
    return A, np.ravel(b)


def CG(A, b):
    b = np.matrix(b).T
    d = len(b)
    x = np.matrix(np.zeros(d)).T
    p = b - np.array(A.dot(x))[0]
    r = p.copy()
    epsilon = 1e-12
    n = 0
    while np.linalg.norm(A.dot(x) - b) / np.linalg.norm(b) > epsilon:
        alpha = np.array(r.T.dot(r))[0][0] / np.array(p.T.dot(A).dot(p))[0][0]
        r_before = r.copy()
        r -= alpha * A.dot(p)
        x += alpha * p
        beta = np.array(r.T.dot(r))[0][0] / np.array(r_before.T.dot(r_before))[0][0]
        p = r + beta * p
        n += 1
    return x, n


def test1(N):
    '''
    :param N:
    :return:
    [0,1]をN分割して解を真の解と比較して出力する。
    '''
    s_all = time.time()
    f, fx0, fx1, fy0, fy1 = make_initial(N)
    start = time.time()
    A, b = make_equation(N, f, fx0, fx1, fy0, fy1)
    elapsed = time.time() - start
    print("elapsed make equation", elapsed)

    # A = np.matrix([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    # b = np.array([1, 2, 3])
    start2 = time.time()
    x, _ = CG(A, b)
    elapsed2 = time.time() - start2
    print("elapsed solve equation", elapsed2)
    # print("----------------------------")
    # print(x)
    # print("calc = ", np.array(A.dot(x)).reshape(1, len(x))[0])
    # print("true = ", b)

    xs, ys = np.mgrid[0:N + 1, 0:N + 1] / N
    zs = [[0 for _ in range(N + 1)] for _ in range(N + 1)]
    zs[0][0] = fun(0, 0)
    zs[0][N] = fun(0, 1)
    zs[N][0] = fun(1, 0)
    zs[N][0] = fun(1, 1)
    for j in range(1, N):
        zs[0][j] = fun(0, j / N)
        zs[N][j] = fun(0, j / N)
    for i in range(1, N):
        zs[i][0] = fun(i / N, 0)
        zs[i][N] = fun(i / N, 1)
    x = np.asarray(x.reshape((1, len(x))))[0]
    for i in range(1, N):
        for j in range(1, N):
            zs[i][j] = x[(i - 1) * (N - 1) + j - 1]
    fig = plt.figure(figsize=plt.figaspect(0.3))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_wireframe(xs, ys, zs, color='blue')
    ax.set_title("Solution of the discretization equation (CG method)")

    e_all = time.time() - s_all
    print("elapsed all = ", e_all)

    #N_ = int(sys.argv[2])
    N_ = 50
    xs_, ys_ = np.mgrid[0:N_ + 1, 0:N_ + 1] / N_
    zs_true = fun(xs_, ys_)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_wireframe(xs_, ys_, zs_true, color='red')
    ax.set_title("Exact solution")

    print("norm(empirical - true) = , ", np.linalg.norm(x - np.ravel(zs_true[1:N_, 1:N_])))

    plt.show()


def test2():
    '''
    :return:
    分割数と、CG法の反復回数との関係を出力する。
    '''
    n = 2
    iter_nums = []
    while True:
        start = time.time()
        f, fx0, fx1, fy0, fy1 = make_initial(n)
        A, b = make_equation(n, f, fx0, fx1, fy0, fy1)
        x, iter_num = CG(A, b)
        elapsed = time.time() - start
        iter_nums.append(iter_num)
        print(elapsed)
        if elapsed > 1:
            break
        n += 1
    plt.plot([i + 2 for i in range(len(iter_nums))], iter_nums)
    plt.title("Relation between number of partitions and number of iterations")
    plt.xlabel("number of partitions")
    plt.ylabel("number of iterations")
    plt.show()


def test3():
    '''
    :return:
    分割数と、残差（離散化方程式に真の解を代入した時の差）との関係を出力する。
    '''
    n = 3
    norms = []
    while True:
        start = time.time()
        f, fx0, fx1, fy0, fy1 = make_initial(n)
        A, b = make_equation(n, f, fx0, fx1, fy0, fy1)
        elapsed = time.time() - start
        xs, ys = np.mgrid[1:n, 1:n] / n
        u_star = fun(xs, ys)
        norms.append(np.linalg.norm(A.dot(np.ravel(u_star)) - b))
        print(elapsed)
        if elapsed > 1:
            break
        n += 1
    plt.plot([i + 2 for i in range(len(norms))][20:], norms[20:])
    plt.title("Relation between number of partitions and residual (Zoom in)")
    plt.xlabel("number of partitions")
    plt.ylabel("residual ||Au - b||")
    plt.show()

def test4():
    '''
    :return:
    分割数と、誤差（離散化方程式の解と真の解の差）との関係を出力する。
    '''
    n = 3
    norms = []
    while True:
        start = time.time()
        f, fx0, fx1, fy0, fy1 = make_initial(n)
        A, b = make_equation(n, f, fx0, fx1, fy0, fy1)
        x, _ = CG(A, b)
        x = np.asarray(x.reshape((1, len(x))))[0]
        elapsed = time.time() - start
        xss, yss = np.mgrid[0:n+1, 0:n+1] / n
        u_star = np.ravel(fun(xss, yss)[1:n, 1:n])
        norms.append(np.linalg.norm(x - u_star))
        print(elapsed)
        if elapsed > 1:
            break
        n += 1
    plt.plot([i + 2 for i in range(len(norms))], norms)
    plt.title("Relation between number of partitions and error")
    plt.xlabel("number of partitions")
    plt.ylabel("error ||u - u*||")
    plt.show()

def test5():
    '''
    :return:
    分割数と、誤差（離散化方程式の解と真の解の差）との関係を両対数グラフで出力する。
    '''
    n = 3
    norms = []
    while True:
        start = time.time()
        f, fx0, fx1, fy0, fy1 = make_initial(n)
        A, b = make_equation(n, f, fx0, fx1, fy0, fy1)
        x, _ = CG(A, b)
        x = np.asarray(x.reshape((1, len(x))))[0]
        elapsed = time.time() - start
        xss, yss = np.mgrid[0:n+1, 0:n+1] / n
        u_star = np.ravel(fun(xss, yss)[1:n, 1:n])
        norms.append(np.linalg.norm(x - u_star) / n)
        print(elapsed)
        if elapsed > 5:
            break
        n += 1
    range_iter = np.asarray([i + 2 for i in range(len(norms))])
    plt.plot(range_iter, norms, label="log(norms / partition number) = log(partition number)")
    plt.plot(range_iter, 1 / range_iter, label="log y = - log x")
    plt.plot(range_iter, 1 / (range_iter ** 2), label="log y = - 2 log x")
    plt.plot(range_iter, 1 / (range_iter ** 3), label="log y = -3 log x")
    plt.legend(loc="lower left")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(min(norms), max(norms)+0.1)
    plt.title("Relation between number of partitions and error (log)")
    plt.xlabel("log(number of partitions)")
    plt.ylabel("log(error)")
    plt.show()

#N = int(sys.argv[1])
N = 50
#test1(N)

#test2()

#test3()

#test4()

test5()