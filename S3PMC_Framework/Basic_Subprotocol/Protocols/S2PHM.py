import numpy as np
from .S2PM import S2PM_Online_Computing_Phase


def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(-0, 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def make_list6(data1, data2, data3, data4, data5, data6):
    return [data1, data2, data3, data4, data5, data6]


def S2PHM_CS_Preprocessing_phase(A_shape, B_shape):
    left_shape = A_shape
    right_shape = B_shape
    Ra1 = get_random_array(left_shape)
    Rb1 = get_random_array(right_shape)
    St1 = np.dot(Ra1, Rb1)
    ra1 = get_random_array(St1.shape)
    rb1 = St1 - ra1

    Ra2 = get_random_array(right_shape)
    Rb2 = get_random_array(left_shape)
    St2 = np.dot(Rb2, Ra2)
    ra2 = get_random_array(St2.shape)
    rb2 = St2 - ra2

    Ra1_ra1_St1_Ra2_ra2_St2_2A = make_list6(Ra1, ra1, St1, Ra2, ra2, St2)
    Rb1_rb1_St1_Rb2_rb2_St2_2B = make_list6(Rb1, rb1, St1, Rb2, rb2, St2)
    Ra1, ra1, St1_2A = Ra1_ra1_St1_Ra2_ra2_St2_2A[0], Ra1_ra1_St1_Ra2_ra2_St2_2A[1], Ra1_ra1_St1_Ra2_ra2_St2_2A[2]
    Ra2, ra2, St2_2A = Ra1_ra1_St1_Ra2_ra2_St2_2A[3], Ra1_ra1_St1_Ra2_ra2_St2_2A[4], Ra1_ra1_St1_Ra2_ra2_St2_2A[5]
    Rb1, rb1, St1_2B = Rb1_rb1_St1_Rb2_rb2_St2_2B[0], Rb1_rb1_St1_Rb2_rb2_St2_2B[1], Rb1_rb1_St1_Rb2_rb2_St2_2B[2]
    Rb2, rb2, St2_2B = Rb1_rb1_St1_Rb2_rb2_St2_2B[3], Rb1_rb1_St1_Rb2_rb2_St2_2B[4], Rb1_rb1_St1_Rb2_rb2_St2_2B[5]
    return Ra1, ra1, St1_2A, Ra2, ra2, St2_2A, Rb1, rb1, St1_2B, Rb2, rb2, St2_2B


def S2PHM_Online_Computing_Phase(A_list, Ra1, ra1, Ra2, ra2, B_list, Rb1, rb1, Rb2, rb2):
    A1, A2 = A_list[0], A_list[1]
    B1, B2 = B_list[0], B_list[1]

    A1A2 = np.dot(A1, A2)
    B1B2 = np.dot(B1, B2)

    Va1, VFa1, VFb1_2A, Vb1, VFa1_2B, VFb1 = S2PM_Online_Computing_Phase(A1, Ra1, ra1, B2, Rb1, rb1)
    Vb2, VFb2, VFa2_2B, Va2, VFb2_2A, VFa2 = S2PM_Online_Computing_Phase(B1, Rb2, rb2, A2, Ra2, ra2)

    Ua = A1A2 + Va1 + Va2
    Ub = B1B2 + Vb1 + Vb2
    return Ua, VFa1, VFb1_2A, VFa2, VFb2_2A, Ub, VFa1_2B, VFb1, VFa2_2B, VFb2


def S2PHM_Result_Verification_Phase(VFa1, VFb1_2A, St1_2A, VFa2, VFb2_2A, St2_2A, VFa1_2B, VFb1, St1_2B, VFa2_2B, VFb2, St2_2B, count):
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
    A_bool1 = verify(VFa1, VFb1_2A, St1_2A ,count) 
    A_bool2 = verify(VFa2, VFb2_2A, St2_2A ,count) 
    B_bool1 = verify(VFa1_2B, VFb1, St1_2B, count)
    B_bool2 = verify(VFa2_2B, VFb2, St2_2B, count)
    return A_bool1 and A_bool2 and B_bool1 and B_bool2


def S2PHM(A_list, B_list, count=0):
    Ra1, ra1, St1_2A, Ra2, ra2, St2_2A, Rb1, rb1, St1_2B, Rb2, rb2, St2_2B = S2PHM_CS_Preprocessing_phase(A_list[0].shape, B_list[1].shape)
    Ua, VFa1, VFb1_2A, VFa2, VFb2_2A, Ub, VFa1_2B, VFb1, VFa2_2B, VFb2 = S2PHM_Online_Computing_Phase(A_list, Ra1, ra1, Ra2, ra2, B_list, Rb1, rb1, Rb2, rb2)
    res_bool = S2PHM_Result_Verification_Phase(VFa1, VFb1_2A, St1_2A, VFa2, VFb2_2A, St2_2A, VFa1_2B, VFb1, St1_2B, VFa2_2B, VFb2, St2_2B, count)
    if res_bool:
        return Ua, Ub
    else:
        raise ValueError('S2PHM verification error.')