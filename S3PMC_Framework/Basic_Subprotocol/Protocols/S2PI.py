import numpy as np
from .S2PM import S2PM_Online_Computing_Phase


def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(-0, 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def make_list9(data1, data2, data3, data4, data5, data6, data7, data8, data9):
    return [data1, data2, data3, data4, data5, data6, data7, data8, data9]


def S2PI_CS_Preprocessing_phase(A_shape, B_shape):
    shape_2C = A_shape
    Ra1 = get_random_array(shape_2C)
    Rb1 = get_random_array(shape_2C)
    St1 = np.dot(Ra1, Rb1)
    ra1 = get_random_array(shape_2C)
    rb1 = St1 - ra1

    Ra2 = get_random_array(shape_2C)
    Rb2 = get_random_array(shape_2C)
    St2 = np.dot(Ra2, Rb2)
    ra2 = get_random_array(shape_2C)
    rb2 = St2 - ra2

    Ra3 = get_random_array(shape_2C)
    Rb3 = get_random_array(shape_2C)
    St3 = np.dot(Rb3, Ra3)
    ra3 = get_random_array(shape_2C)
    rb3 = St3 - ra3

    PE = np.random.randint(1, 11, A_shape[0])
    P = np.diag(PE)
    QE = np.random.randint(1, 11, B_shape[0])
    Q = np.diag(QE)

    Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A = make_list9(Ra1, ra1, St1, Ra2, ra2, St2, Ra3, ra3, St3)
    Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B = make_list9(Rb1, rb1, St1, Rb2, rb2, St2, Rb3, rb3, St3)
    Ra1, ra1, St1_2A = Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[0], Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[1], Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[2]
    Ra2, ra2, St2_2A = Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[3], Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[4], Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[5]
    Ra3, ra3, St3_2A = Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[6], Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[7], Ra1_ra1_St1_Ra2_ra2_St2_Ra3_ra3_St3_2A[8]
    Rb1, rb1, St1_2B = Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[0], Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[1], Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[2]
    Rb2, rb2, St2_2B = Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[3], Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[4], Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[5]
    Rb3, rb3, St3_2B = Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[6], Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[7], Rb1_rb1_St1_Rb2_rb2_St2_Rb3_rb3_St3_2B[8]    
    return P, Ra1, ra1, St1_2A, Ra2, ra2, St2_2A, Ra3, ra3, St3_2A, Q, Rb1, rb1, St1_2B, Rb2, rb2, St2_2B, Rb3, rb3, St3_2B


def S2PI_Online_Computing_Phase(A, P, Ra1, ra1, Ra2, ra2, Ra3, ra3, B, Q, Rb1, rb1, Rb2, rb2, Rb3, rb3):
    PA = np.dot(P, A)
    BQ = np.dot(B, Q)
    Va1, VFa1, VFb1_2A, Vb1, VFa1_2B, VFb1 = S2PM_Online_Computing_Phase(PA, Ra1, ra1, Q, Rb1, rb1)
    Va2, VFa2, VFb2_2A, Vb2, VFa2_2B, VFb2 = S2PM_Online_Computing_Phase(P, Ra2, ra2, BQ, Rb2, rb2)
 
    V = (Va1 + Va2) + Vb1 + Vb2
    inv_V = np.linalg.inv(V)
    T = np.dot(Q, inv_V)
    Vb3, VFb3, VFa3_2B, Va3, VFb3_2A, VFa3 = S2PM_Online_Computing_Phase(T, Rb3, rb3, P, Ra3, ra3)
    return Va3, VFa1, VFb1_2A, VFa2, VFb2_2A, VFa3, VFb3_2A, Vb3, VFa1_2B, VFb1, VFa2_2B, VFb2, VFa3_2B, VFb3


def S2PI_Result_Verification_Phase(VFa1, VFb1_2A, St1_2A, VFa2, VFb2_2A, St2_2A, VFa3, VFb3_2A, St3_2A, VFa1_2B, VFb1, St1_2B, VFa2_2B, VFb2, St2_2B, VFa3_2B, VFb3, St3_2B, count):
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
    A_bool1 = verify(VFa1, VFb1_2A, St1_2A, count) 
    A_bool2 = verify(VFa2, VFb2_2A, St2_2A, count) 
    A_bool3 = verify(VFa3, VFb3_2A, St3_2A, count) 
    B_bool1 = verify(VFa1_2B, VFb1, St1_2B, count)
    B_bool2 = verify(VFa2_2B, VFb2, St2_2B, count)
    B_bool3 = verify(VFa3_2B, VFb3, St3_2B, count)
    return A_bool1 and A_bool2 and A_bool3 and B_bool1 and B_bool2 and B_bool3


def S2PI(A, B, count=0):
    P, Ra1, ra1, St1_2A, Ra2, ra2, St2_2A, Ra3, ra3, St3_2A, Q, Rb1, rb1, St1_2B, Rb2, rb2, St2_2B, Rb3, rb3, St3_2B = S2PI_CS_Preprocessing_phase(A.shape, B.shape)
    Va, VFa1, VFb1_2A, VFa2, VFb2_2A, VFa3, VFb3_2A, Vb, VFa1_2B, VFb1, VFa2_2B, VFb2, VFa3_2B, VFb3 = S2PI_Online_Computing_Phase(A, P, Ra1, ra1, Ra2, ra2, Ra3, ra3, B, Q, Rb1, rb1, Rb2, rb2, Rb3, rb3)
    res_bool = S2PI_Result_Verification_Phase(VFa1, VFb1_2A, St1_2A, VFa2, VFb2_2A, St2_2A, VFa3, VFb3_2A, St3_2A, VFa1_2B, VFb1, St1_2B, VFa2_2B, VFb2, St2_2B, VFa3_2B, VFb3, St3_2B, count)
    if res_bool:
        return Va, Vb
    else:
        raise ValueError('S2PI verification error.')   