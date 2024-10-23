import numpy as np


def verify_2p(VFa, VFb, St, count):
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
