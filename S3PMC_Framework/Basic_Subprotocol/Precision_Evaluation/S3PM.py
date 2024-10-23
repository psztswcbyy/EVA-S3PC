import numpy as np
from tabulate import tabulate


E = [0, 0]

def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(E[0], E[1] + 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def get_similar_array(A):
    res = np.abs(A)
    zero_index = (res == 0)
    res[zero_index] = 1
    res = (np.random.random(A.shape) + np.random.randint(1, 10, A.shape)) * np.power(10, np.floor(np.log10(res)))
    return res.astype(A.dtype)


def full_rank_decomposition(data):
    return np.linalg.qr(data)


def dot3p(A, B, C):
    return A.dot(B).dot(C)


def make_list2(data1, data2):
    return [data1, data2]


def S3PM_CS_Preprocessing_phase(A_pre, B_pre, C_pre):
    Ra = get_similar_array(A_pre)
    Rb = get_similar_array(B_pre)
    Rc = get_similar_array(C_pre)
    St = dot3p(Ra, Rb, Rc)
    ra = get_similar_array(St)
    rb = get_similar_array(St)
    rc = St - ra - rb
    Ra_ra_2A = make_list2(Ra, ra)
    Rb_rb_2B = make_list2(Rb, rb)
    Rc_rc_2C = make_list2(Rc, rc)
    return Ra_ra_2A[0], Ra_ra_2A[1], Rb_rb_2B[0], Rb_rb_2B[1], Rc_rc_2C[0], Rc_rc_2C[1]


def S3PM_Online_Computing_Phase(A, Ra, ra, B, Rb, rb, C, Rc, rc):
    A_hat = A + Ra
    A_hat_2B = A_hat
    B_hat = B + Rb
    C_hat = C + Rc
    C_hat_2B = C_hat

    phi1 = np.dot(A_hat_2B, B_hat)
    y1 = np.dot(A_hat_2B, Rb)
    phi2 = np.dot(B_hat, C_hat_2B)
    y2 = np.dot(Rb, C_hat_2B)
    Mb = np.dot(y1, C_hat_2B)

    y1_phi1_2C = make_list2(y1, phi1)
    y2_phi2_2A = make_list2(y2, phi2)

    Sa = np.dot(Ra, y2_phi2_2A[0])
    Ma = np.dot(A, y2_phi2_2A[1])

    Sc = np.dot(y1_phi1_2C[0], Rc)
    Mc = np.dot(y1_phi1_2C[1], Rc)

    B1_B2 = full_rank_decomposition(B_hat)
    B1_2A = B1_B2[0]
    B2_2C = B1_B2[1]

    Va = get_similar_array(ra)
    Ta = Ma + Sa - (Va + ra)
    t1 = np.dot(Ra, B1_2A)

    Vb = get_similar_array(rb)
    Tb = Ta - Mb - (Vb + rb)

    Sb = dot3p(t1, B2_2C, Rc)
    Vc = Tb - Mc + Sb + Sc - rc
    return Va, Vb, Vc


def S3PM(A, B, C):
    A_pre = get_similar_array(A)
    B_pre = get_similar_array(B)
    C_pre = get_similar_array(C)
    Ra, ra, Rb, rb, Rc, rc = S3PM_CS_Preprocessing_phase(A_pre, B_pre, C_pre)
    Va, Vb, Vc = S3PM_Online_Computing_Phase(A, Ra, ra, B, Rb, rb, C, Rc, rc)
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
            A = get_random_array(shape)
            B = get_random_array(shape)
            C = get_random_array(shape)
            Va, Vb, Vc = S3PM(A, B, C)
            res = Va + Vb + Vc
            S_res = A.dot(B).dot(C)
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
                A = get_random_array(shape)
                B = get_random_array(shape)
                C = get_random_array(shape)
                Va, Vb, Vc = S3PM(A, B, C)
                res = Va + Vb + Vc
                S_res = A.dot(B).dot(C)
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
