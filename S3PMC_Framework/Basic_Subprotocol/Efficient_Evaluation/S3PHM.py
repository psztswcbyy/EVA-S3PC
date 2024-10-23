import numpy as np
from S2PM import S2PM_Online_Computing_Phase
from S3PM import S3PM_Online_Computing_Phase
import time
from tabulate import tabulate


def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(-0, 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def dot3p(A, B, C):
    return A.dot(B).dot(C)


def make_list2(data1, data2):
    return [data1, data2]


def make_list9(data1, data2, data3, data4, data5, data6, data7, data8, data9):
    return [data1, data2, data3, data4, data5, data6, data7, data8, data9]


def make_list12(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12):
    return [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]


def S3PHM_CS_Preprocessing_phase(A_shape, B_shape, C_shape):
    left_shape = A_shape
    mid_shape = B_shape
    left_mid_shape = make_list2(left_shape[0], mid_shape[1])
    right_shape = C_shape

    Ra1 = get_random_array(left_mid_shape)
    Rc1 = get_random_array(right_shape)
    St1 = np.dot(Ra1, Rc1)
    ra1 = get_random_array(St1.shape)
    rc1 = St1 - ra1

    Rb1 = get_random_array(left_mid_shape)
    Rc2 = get_random_array(right_shape)
    St2 = np.dot(Rb1, Rc2)
    rb1 = get_random_array(St2.shape)
    rc2 = St2 - rb1

    Ra2 = get_random_array(left_shape)
    Rb2 = get_random_array(mid_shape)
    Rc3 = get_random_array(right_shape)
    St3 = dot3p(Ra2, Rb2, Rc3)
    ra2 = get_random_array(St3.shape)
    rb2 = get_random_array(St3.shape)
    rc3 = St3 - ra2 - rb2

    Rb3 = get_random_array(left_shape)
    Ra3 = get_random_array(mid_shape)
    Rc4 = get_random_array(right_shape)
    St4 = dot3p(Rb3, Ra3, Rc4)
    ra3 = get_random_array(St4.shape)
    rb3 = get_random_array(St4.shape)
    rc4 = St4 - ra3 - rb3

    Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A = make_list9(Ra1, ra1, St1, Ra2, ra2, St3, Ra3, ra3, St4)
    Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B = make_list9(Rb1, rb1, St2, Rb2, rb2, St3, Rb3, rb3, St4)
    Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C = make_list12(Rc1, rc1, St1, Rc2, rc2, St2, Rc3, rc3, St3, Rc4, rc4, St4)

    Ra1, ra1, St1_2A = Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[0], Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[1], Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[2]
    Ra2, ra2, St3_2A = Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[3], Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[4], Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[5]
    Ra3, ra3, St4_2A = Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[6], Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[7], Ra1_ra1_St1_Ra2_ra2_St3_Ra3_ra3_St4_2A[8]

    Rb1, rb1, St2_2B = Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[0], Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[1], Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[2]
    Rb2, rb2, St3_2B = Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[3], Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[4], Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[5]
    Rb3, rb3, St4_2B = Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[6], Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[7], Rb1_rb1_St2_Rb2_rb2_St3_Rb3_rb3_St4_2B[8]

    Rc1, rc1, St1_2C = Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[0], Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[1], Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[2]
    Rc2, rc2, St2_2C = Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[3], Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[4], Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[5]
    Rc3, rc3, St3_2C = Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[6], Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[7], Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[8]
    Rc4, rc4, St4_2C = Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[9], Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[10], Rc1_rc1_St1_Rc2_rc2_St2_Rc3_rc3_St3_Rc4_rc4_St4_2C[11]
    return Ra1, ra1, St1_2A, Ra2, ra2, St3_2A, Ra3, ra3, St4_2A, Rb1, rb1, St2_2B, Rb2, rb2, St3_2B, Rb3, rb3, St4_2B, Rc1, rc1, St1_2C, Rc2, rc2, St2_2C, Rc3, rc3, St3_2C, Rc4, rc4, St4_2C

def S3PHM_Online_Computing_Phase(A_list, Ra1, ra1, Ra2, ra2, Ra3, ra3, B_list, Rb1, rb1, Rb2, rb2, Rb3, rb3, Y, Rc1, rc1, Rc2, rc2, Rc3, rc3, Rc4, rc4):
    A1, A2 = A_list[0], A_list[1]
    B1, B2 = B_list[0], B_list[1]

    A1A2 = np.dot(A1, A2)
    B1B2 = np.dot(B1, B2)

    Va1, VFa1, VFc1_2A, Vc1, VFa1_2C, VFc1 = S2PM_Online_Computing_Phase(A1A2, Ra1, ra1, Y, Rc1, rc1)
    Vb1, VFb1, VFc2_2B, Vc2, VFb1_2C, VFc2 = S2PM_Online_Computing_Phase(B1B2, Rb1, rb1, Y, Rc2, rc2)
    Va2, VFa2, VFb2_2A, VFc3_2A, Vb2, VFa2_2B, VFb2, VFc3_2B, Vc3, VFa2_2C, VFb2_2C, VFc3 = S3PM_Online_Computing_Phase(A1, Ra2, ra2, B2, Rb2, rb2, Y, Rc3, rc3)
    Vb3, VFb3, VFa3_2B, VFc4_2B, Va3, VFb3_2A, VFa3, VFc4_2A, Vc4, VFb3_2C, VFa3_2C, VFc4 = S3PM_Online_Computing_Phase(B1, Rb3, rb3, A2, Ra3, ra3, Y, Rc4, rc4)

    Va = Va1 + Va2 + Va3
    Vb = Vb1 + Vb2 + Vb3
    Vc = Vc1 + Vc2 + Vc3 + Vc4
    return Va, VFa1, VFc1_2A, VFa2, VFb2_2A, VFc3_2A, VFa3, VFb3_2A, VFc4_2A, Vb, VFb1, VFc2_2B, VFa2_2B, VFb2, VFc3_2B, VFa3_2B, VFb3, VFc4_2B, Vc, VFa1_2C, VFc1, VFb1_2C, VFc2, VFa2_2C, VFb2_2C, VFc3, VFb3_2C, VFa3_2C, VFc4

def S3PHM_Result_Verification_Phase(VFa1, VFc1_2A, St1_2A, VFa2, VFb2_2A, VFc3_2A, St3_2A, VFa3, VFb3_2A, VFc4_2A, St4_2A, VFb1, VFc2_2B, St2_2B, VFa2_2B, VFb2, VFc3_2B, St3_2B, VFa3_2B, VFb3, VFc4_2B, St4_2B, VFa1_2C, VFc1, St1_2C, VFb1_2C, VFc2, St2_2C, VFa2_2C, VFb2_2C, VFc3, St3_2C, VFb3_2C, VFa3_2C, VFc4, St4_2C, count):
    def verify_2p(VFa, VFb, St, count):
        def check(VFa, VFb, St):
            delta = np.random.randint(0, 2, (St.shape[1], 1))
            Er = (VFa + VFb - St).dot(delta)
            St_prime = St.dot(delta)
            St_prime[St_prime == 0] = 1
            Er = np.max(np.abs(Er / St_prime))
            return Er > 1e-1
        for _ in range(count):
            if check(VFa, VFb, St):
                return False
        return True
    def verify_3p(VFa, VFb, VFc, St, count):
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
    A_bool1 = verify_2p(VFa1, VFc1_2A, St1_2A, count)
    A_bool2 = verify_3p(VFa2, VFb2_2A, VFc3_2A, St3_2A, count)
    A_bool3 = verify_3p(VFa3, VFb3_2A, VFc4_2A, St4_2A, count)
    B_bool1 = verify_2p(VFb1, VFc2_2B, St2_2B, count)
    B_bool2 = verify_3p(VFa2_2B, VFb2, VFc3_2B, St3_2B, count)
    B_bool3 = verify_3p(VFa3_2B, VFb3, VFc4_2B, St4_2B, count)
    C_bool1 = verify_2p(VFa1_2C, VFc1, St1_2C, count)
    C_bool2 = verify_2p(VFb1_2C, VFc2, St2_2C, count)
    C_bool3 = verify_3p(VFa2_2C, VFb2_2C, VFc3, St3_2C, count)
    C_bool4 = verify_3p(VFb3_2C, VFa3_2C, VFc4, St4_2C, count)
    return A_bool1 and A_bool2 and A_bool3 and B_bool1 and B_bool2 and B_bool3 and C_bool1 and C_bool2 and C_bool3 and C_bool4


def S3PHM(A_list, B_list, Y, count=0):
    t1 = time.time()
    Ra1, ra1, St1_2A, Ra2, ra2, St3_2A, Ra3, ra3, St4_2A, Rb1, rb1, St2_2B, Rb2, rb2, St3_2B, Rb3, rb3, St4_2B, Rc1, rc1, St1_2C, Rc2, rc2, St2_2C, Rc3, rc3, St3_2C, Rc4, rc4, St4_2C = S3PHM_CS_Preprocessing_phase(A_list[0].shape, B_list[1].shape, Y.shape)
    t2 = time.time()
    Va, VFa1, VFc1_2A, VFa2, VFb2_2A, VFc3_2A, VFa3, VFb3_2A, VFc4_2A, Vb, VFb1, VFc2_2B, VFa2_2B, VFb2, VFc3_2B, VFa3_2B, VFb3, VFc4_2B, Vc, VFa1_2C, VFc1, VFb1_2C, VFc2, VFa2_2C, VFb2_2C, VFc3, VFb3_2C, VFa3_2C, VFc4 = S3PHM_Online_Computing_Phase(A_list, Ra1, ra1, Ra2, ra2, Ra3, ra3, B_list, Rb1, rb1, Rb2, rb2, Rb3, rb3, Y, Rc1, rc1, Rc2, rc2, Rc3, rc3, Rc4, rc4)
    t3 = time.time()
    res_bool = S3PHM_Result_Verification_Phase(VFa1, VFc1_2A, St1_2A, VFa2, VFb2_2A, VFc3_2A, St3_2A, VFa3, VFb3_2A, VFc4_2A, St4_2A, VFb1, VFc2_2B, St2_2B, VFa2_2B, VFb2, VFc3_2B, St3_2B, VFa3_2B, VFb3, VFc4_2B, St4_2B, VFa1_2C, VFc1, St1_2C, VFb1_2C, VFc2, St2_2C, VFa2_2C, VFb2_2C, VFc3, St3_2C, VFb3_2C, VFa3_2C, VFc4, St4_2C, count)
    t4 = time.time()
    if res_bool:
        return Va, Vb, Vc, t2 - t1, t3 - t2, t4 - t3
    else:
        raise ValueError('S3PHM verification error.')
    

def main():
    header = ['S3PHM/shape', 'Preparing(s)', 'Online(s)', 'Verification(s)', 'Computation(s)']
    res = []
    for x in [1, 2, 3, 4, 5]:
        shape = (x * 10, x * 10)
        A1 = get_random_array(shape)
        B1 = get_random_array(shape)
        A2 = get_random_array(shape)
        B2 = get_random_array(shape)
        Y = get_random_array(shape)
        a, b, c, t1, t2, t3 = S3PHM([A1, A2], [B1, B2], Y, 1)
        res.append([shape, t1, t2, t3, t1 + t2 + t3])
    
    print(tabulate(res, headers=header))   


if __name__ == "__main__":
    res = main()
