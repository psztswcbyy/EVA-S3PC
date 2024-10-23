import numpy as np
from S2PM import S2PM
from S3PM import S3PM
from tabulate import tabulate


def get_random_array(size):
    A = np.random.random(size).astype(np.float64) + 1
    B = np.random.randint(-0, 1, size=size).astype(np.float64)
    C = A * (np.power(10, B))
    return C


def S2PM_test(shape1, shape2):
    A = get_random_array(shape1)
    B = get_random_array(shape2)
    return S2PM(A, B, 1)

def S3PM_test(shape1, shape2, shape3):
    A = get_random_array(shape1)
    B = get_random_array(shape2)
    C = get_random_array(shape3)
    return S3PM(A, B, C, 1)


def main():
    headers = [['S2PM/shape'], ['S3PM/shape']]
    for item in headers:
        item.append('Preparing(s)')
        item.append('Online(s)')
        item.append('Verification(s)')
        item.append('Computation(s)')
    
    res = [[] for item in headers]
    for x in [1, 2, 3, 4, 5, 6, 8, 10, 25, 50]:
        shape = (x * 10, x * 10)

        a, b, t1, t2, t3 = S2PM_test(shape, shape)
        res[0].append([shape, t1, t2, t3, t1 + t2 + t3])

        a, b, c, t1, t2, t3 = S3PM_test(shape, shape, shape)
        res[1].append([shape, t1, t2, t3, t1 + t2 + t3])
    
    for i in range(len(headers)):
        print('\n')
        print(tabulate(res[i], headers=headers[i]))   


if __name__ == "__main__":
    res = main()
