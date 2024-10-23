import numpy as np
from tabulate import tabulate
from S2PM import S2PM
from S3PM import S3PM


E = [0, 0]

def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(E[0], E[1] + 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def S3PHM(A_list, B_list, Y):
    A1, A2 = A_list[0], A_list[1]
    B1, B2 = B_list[0], B_list[1]

    A1A2 = np.dot(A1, A2)
    B1B2 = np.dot(B1, B2)

    Va1, Vc1 = S2PM(A1A2, Y)
    Vb1, Vc2 = S2PM(B1B2, Y)
    Va2, Vb2, Vc3 = S3PM(A1, B2, Y)
    Vb3, Va3, Vc4 = S3PM(B1, A2, Y)

    Va = Va1 + Va2 + Va3
    Vb = Vb1 + Vb2 + Vb3
    Vc = Vc1 + Vc2 + Vc3 + Vc4
    return Va, Vb, Vc


def get_max_error(S_res, res):
    return min(np.max(np.abs((res - S_res) / S_res)), 1)


def get_average_error(S_res, res):
    return min(np.average(np.abs((res - S_res) / S_res)), 1)


def get_f2norm(S_res, res):
    up = np.linalg.norm(S_res - res, 'fro')
    down = np.linalg.norm(S_res, 'fro')
    return min(up / down, 1)


def relative_test():
    headers = [['Max Relative Error/shape'], ['Average Relative Error/shape']]
    for item in headers:
        item += ['E[0, 0]', 'E[-2, 2]', 'E[-4, 4]', 'E[-6, 6]', 'E[-8, 8]', 'E[-10, 10]']
    context = [[] for item in headers]
    for shape in [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]:
        tmpcontext0 = [shape]
        tmpcontext1 = [shape]
        for tmp_E in [0, 2, 4, 6, 8, 10]:
            global E 
            E = [-tmp_E, tmp_E]
            A1 = get_random_array(shape)
            B1 = get_random_array(shape)
            A2 = get_random_array(shape)
            B2 = get_random_array(shape)
            Y = get_random_array(shape)
            Va, Vb, Vc = S3PHM([A1, A2], [B1, B2], Y)
            res = Va + Vb + Vc
            S_res = (A1 + B1).dot(A2 + B2).dot(Y)
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
    headers = [['E[0, 0]/shape'], ['E[-2, 2]/shape'], ['E[-4, 4]/shape'], ['E[-6, 6]/shape'], ['E[-8, 8]/shape'], ['E[-10, 10]/shape']]
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
                A1 = get_random_array(shape)
                B1 = get_random_array(shape)
                A2 = get_random_array(shape)
                B2 = get_random_array(shape)
                Y = get_random_array(shape)
                Va, Vb, Vc = S3PHM([A1, A2], [B1, B2], Y)
                res = Va + Vb + Vc
                S_res = (A1 + B1).dot(A2 + B2).dot(Y)
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
