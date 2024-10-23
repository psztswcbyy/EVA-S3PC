import numpy as np


def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(-0, 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def RNMG(shape):
    r = min(shape[0], shape[1]) - 1
    M0 = np.zeros(shape)
    lamb = np.random.random(r)
    index = np.random.permutation(min(shape[0], shape[1]))
    for i in range(r):
        M0[index[i]][index[i]] = lamb[i]
    P = np.random.random((shape[0], shape[0]))
    Q = np.random.random((shape[1], shape[1]))
    return P.dot(M0).dot(Q)


def make_list2(data1, data2):
    return [data1, data2]


def make_list3(data1, data2, data3):
    return [data1, data2, data3]


def S2PM_CS_Preprocessing_phase(A_shape, B_shape):
    Ra = RNMG(A_shape)
    Rb = RNMG(B_shape)
    St = np.dot(Ra, Rb)
    ra = get_random_array(St.shape)
    rb = St - ra
    Ra_ra_St_2A = make_list3(Ra, ra, St)
    Rb_rb_St_2B = make_list3(Rb, rb, St)
    return Ra_ra_St_2A[0], Ra_ra_St_2A[1], Ra_ra_St_2A[2], Rb_rb_St_2B[0], Rb_rb_St_2B[1], Rb_rb_St_2B[2]


def S2PM_Online_Computing_Phase(A, Ra, ra, B, Rb, rb):
    A_hat = A + Ra
    A_hat_2B = A_hat
    B_hat = B + Rb
    B_hat_2A = B_hat

    Vb = get_random_array(rb.shape)
    VFb = Vb - np.dot(A_hat_2B, B)
    T = rb - VFb
    VFb_T_2A = make_list2(VFb, T)
    VFa = VFb_T_2A[1] + ra
    Va = VFa - np.dot(Ra, B_hat_2A)
    VFa_2B = VFa
    return Va, VFa, VFb_T_2A[0], Vb, VFa_2B, VFb


def S2PM_Result_Verification_Phase(VFa, VFb_2A, St_2A, VFa_2B, VFb, St_2B, count):
    def verify(VFa, VFb, St, count):
        def check(VFa, VFb, St):
            delta = np.random.randint(0, 2, (St.shape[1], 1))
            Er = (VFa + VFb - St).dot(delta)
            St_prime = St.dot(delta)
            St_prime[St_prime == 0] = 1
            Er = np.max(np.abs(Er / St_prime))
            return Er > 1e-5
        for _ in range(count):
            if check(VFa, VFb, St):
                return False
        return True
    A_bool = verify(VFa, VFb_2A, St_2A ,count) 
    B_bool = verify(VFa_2B, VFb, St_2B, count)
    return A_bool and B_bool


def S2PM(A, B, count=0):
    Ra, ra, St_2A, Rb, rb, St_2B = S2PM_CS_Preprocessing_phase(A.shape, B.shape)
    Va, VFa, VFb_2A, Vb, VFa_2B, VFb = S2PM_Online_Computing_Phase(A, Ra, ra, B, Rb, rb)
    res_bool = S2PM_Result_Verification_Phase(VFa, VFb_2A, St_2A, VFa_2B, VFb, St_2B, count)
    if res_bool:
        return Va, Vb
    else:
        raise ValueError('S2PM verification error.')
