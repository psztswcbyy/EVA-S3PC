import numpy as np
from tabulate import tabulate
from S2PM import S2PM


E = [0, 0]

def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(E[0], E[1] + 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def S2PI(A, B):
    PE = np.random.randint(1, 11, A.shape[0])
    P = np.diag(PE)
    QE = np.random.randint(1, 11, B.shape[0])
    Q = np.diag(QE)

    PA = np.dot(P, A)
    BQ = np.dot(B, Q)

    Va1, Vb1 = S2PM(PA, Q)
    Va2, Vb2 = S2PM(P, BQ)
 
    V = (Va1 + Va2) + Vb1 + Vb2
    inv_V = np.linalg.inv(V)
    T = np.dot(Q, inv_V)
    Vb3, Va3 = S2PM(T, P)
    return Va3, Vb3


def get_max_error(S_res, res):
    return min(np.max(np.abs((res - S_res) / S_res)), 1)


def get_average_error(S_res, res):
    return min(np.average(np.abs((res - S_res) / S_res)), 1)


def get_f2norm(S_res, res):
    up = np.linalg.norm(S_res - res, 'fro')
    down = np.linalg.norm(S_res, 'fro')
    return min(up / down, 1)


def relative_test():
    headers = [['Max Relative Error'], ['Average Relative Error']]
    for item in headers:
        item += ['E[0, 0]', 'E[-2, 2]', 'E[-4, 4]', 'E[-6, 6]', 'E[-8, 8]', 'E[-10, 10]']
    context = [[] for item in headers]
    for shape in [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]:
        tmpcontext0 = [shape]
        tmpcontext1 = [shape]
        for tmp_E in [0, 2, 4, 6, 8, 10]:
            global E 
            E = [-tmp_E, tmp_E]
            A = get_random_array(shape)
            while np.linalg.cond(A) > 1000 :
                A = get_random_array(shape)
            B = get_random_array(shape)
            while np.linalg.cond(B) > 1000 :
                B = get_random_array(shape)
            Va, Vb = S2PI(A, B)
            res = Va + Vb
            S_res = np.linalg.inv(A + B)
            maxe = get_max_error(S_res, res)
            averagee = get_average_error(S_res, res)
            tmpcontext0.append(maxe)
            tmpcontext1.append(averagee)
        context[0].append(tmpcontext0)
        context[1].append(tmpcontext1)
    
    for i in range(len(headers)):
        print('\n')
        print(tabulate(context[i], headers=headers[i]))

def norm_test():
    headers = [['E[0, 0]'], ['E[-2, 2]'], ['E[-4, 4]'], ['E[-6, 6]'], ['E[-8, 8]'], ['E[-10, 10]']]
    shape_list = [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
    for item in headers:
        item += ['Fnorm(Q1)', 'Fnorm(Q2)', 'Fnorm(Q3)', 'Fnorm(Q4)', 'Fnorm(Q5)', 'Accuracy']
    context = [[[0 for i in range(len(headers[0]))] for j in range(len(shape_list))] for item in headers]
    cnt_all = 1000

    tmp_E_list = [0, 2, 4, 6, 8, 10]
    for i in range(len(tmp_E_list)):
        tmp_E = tmp_E_list[i]
        global E 
        E = [-tmp_E, tmp_E]
        for j in range(len(shape_list)):
            shape = shape_list[j]
            context[i][j][0] = shape
            norm_list = []
            for _ in range(cnt_all):
                A = get_random_array(shape)
                while np.linalg.cond(A) > 1000 :
                    A = get_random_array(shape)
                B = get_random_array(shape)
                while np.linalg.cond(B) > 1000 :
                    B = get_random_array(shape)
                Va, Vb = S2PI(A, B)
                res = Va + Vb
                S_res = np.linalg.inv(A + B)
                f2norm = get_f2norm(S_res, res)
                norm_list.append(f2norm)
            norm_list = np.array(norm_list)
            percentiles = np.percentile(norm_list, [0, 25, 50, 75, 100])
            context[i][j][1], context[i][j][2], context[i][j][3], context[i][j][4], context[i][j][5] = percentiles[0], percentiles[1], percentiles[2], percentiles[3], percentiles[4]
            context[i][j][6] = np.sum(norm_list < 1) / cnt_all

    for i in range(len(headers)):
        print('\n')
        print(tabulate(context[i], headers=headers[i]))

def main():
    relative_test()
    norm_test()

if __name__ == "__main__":
    res = main()
