import numpy as np
import time
from tabulate import tabulate


def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(-0, 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def dot3p(A, B, C):
    return A.dot(B).dot(C)


def full_rank_decomposition(data):
    return np.linalg.qr(data)


def make_list2(data1, data2):
    return [data1, data2]


def make_list3(data1, data2, data3):
    return [data1, data2, data3]


def S3PM_CS_Preprocessing_phase(A_shape, B_shape, C_shape):
    Ra = get_random_array(A_shape)
    Rb = get_random_array(B_shape)
    Rc = get_random_array(C_shape)
    St = dot3p(Ra, Rb, Rc)
    ra = get_random_array(St.shape)
    rb = get_random_array(St.shape)
    rc = St - ra - rb
    Ra_ra_St_2A = make_list3(Ra, ra, St)
    Rb_rb_St_2B = make_list3(Rb, rb, St)
    Rc_rc_St_2C = make_list3(Rc, rc, St)
    return Ra_ra_St_2A[0], Ra_ra_St_2A[1], Ra_ra_St_2A[2], Rb_rb_St_2B[0], Rb_rb_St_2B[1], Rb_rb_St_2B[2], Rc_rc_St_2C[0], Rc_rc_St_2C[1], Rc_rc_St_2C[2]


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

    Va = get_random_array(ra.shape)
    VFa = Ma + Sa - Va
    Ta = VFa - ra
    t1 = np.dot(Ra, B1_2A)

    Ta_VFa_2B = make_list2(Ta, VFa)
    t1_VFa_2C = make_list2(t1, VFa)

    Vb = get_random_array(rb.shape)
    VFb = 0 - Mb - Vb
    Tb = Ta_VFa_2B[0] + VFb - rb

    VFb_2A = VFb
    VFb_Tb_2C = make_list2(VFb, Tb)

    Sb = dot3p(t1_VFa_2C[0], B2_2C, Rc)
    Vc = VFb_Tb_2C[1] - Mc + Sb + Sc - rc
    VFc = Sc + Sb - Vc - Mc

    VFc_2A = VFc
    VFc_2B = VFc
    return Va, VFa, VFb_2A, VFc_2A, Vb, Ta_VFa_2B[1], VFb, VFc_2B, Vc, t1_VFa_2C[1], VFb_Tb_2C[0], VFc


def S3PM_Result_Verification_Phase(VFa, VFb_2A, VFc_2A, St_2A, VFa_2B, VFb, VFc_2B, St_2B, VFa_2C, VFb_2C, VFc, St_2C, count):
    def verify(VFa, VFb, VFc, St, count):
        def check(VFa, VFb, VFc, St):
            delta = np.random.randint(0, 2, (St.shape[1], 1))
            Er = (VFa + VFb + VFc - St).dot(delta)
            St_prime = St.dot(delta)
            St_prime[St_prime == 0] = 1
            Er = np.max(np.abs(Er / St_prime))
            return Er > 1e-5
        for _ in range(count):
            if check(VFa, VFb, VFc, St):
                return False
        return True
    A_bool = verify(VFa, VFb_2A, VFc_2A, St_2A, count) 
    B_bool = verify(VFa_2B, VFb, VFc_2B, St_2B, count)
    C_bool = verify(VFa_2C, VFb_2C, VFc, St_2C, count)
    return A_bool and B_bool and C_bool


def S3PM(A, B, C, count=0):
    t1 = time.time()
    Ra, ra, St_2A, Rb, rb, St_2B, Rc, rc, St_2C = S3PM_CS_Preprocessing_phase(A.shape, B.shape, C.shape)
    t2 = time.time()
    Va, VFa, VFb_2A, VFc_2A, Vb, VFa_2B, VFb, VFc_2B, Vc, VFa_2C, VFb_2C, VFc = S3PM_Online_Computing_Phase(A, Ra, ra, B, Rb, rb, C, Rc, rc)
    t3 = time.time()
    res_bool = S3PM_Result_Verification_Phase(VFa, VFb_2A, VFc_2A, St_2A, VFa_2B, VFb, VFc_2B, St_2B, VFa_2C, VFb_2C, VFc, St_2C, count)
    t4 = time.time()
    if res_bool:
        return Va, Vb, Vc, t2 - t1, t3 - t2, t4 - t3
    else:
        raise ValueError('S3PM verification error.')


def main():
    header = ['S3PM/shape', 'Preparing(s)', 'Online(s)', 'Verification(s)', 'Computation(s)']
    res = []
    for x in [1, 2, 3, 4, 5]:
        shape = (x * 10, x * 10)
        A = get_random_array(shape)
        B = get_random_array(shape)
        C = get_random_array(shape)
        a, b, c, t1, t2, t3 = S3PM(A, B, C, 1)
        res.append([shape, t1, t2, t3, t1 + t2 + t3])
    
    print(tabulate(res, headers=header))   


if __name__ == "__main__":
    res = main()
