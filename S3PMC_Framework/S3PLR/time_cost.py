import os
import numpy as np
import csv
import time
from S3PLR import S3PLR


def get_f2norm(S_res, res):
    up = np.linalg.norm(S_res - res, 'fro')
    down = np.linalg.norm(S_res, 'fro')
    return up / down


def read_csv(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        file = []
        cnt = 0
        for i in csv.reader(f):
            file.append(i)
            cnt += 1
        f.close()
    return np.array(file, dtype='float64')


def diabetes_test():
    print('\ndiabetes_test:')
    current_path = os.path.dirname(__file__)
    x1_train = read_csv(os.path.join(current_path, 'data/diabetes_x1_train.csv'))
    x2_train = read_csv(os.path.join(current_path, 'data/diabetes_x2_train.csv'))
    y_train = read_csv(os.path.join(current_path, 'data/diabetes_y_train.csv'))
    x1_test = read_csv(os.path.join(current_path, 'data/diabetes_x1_test.csv'))
    x2_test = read_csv(os.path.join(current_path, 'data/diabetes_x2_test.csv'))
    y_test = read_csv(os.path.join(current_path, 'data/diabetes_y_test.csv'))

    model = S3PLR()
    t1 = time.time()
    model.fit(x1_train, x2_train, y_train)
    t2 = time.time()
    Ya, Yb, Yc = model.predict(x1_test, x2_test)
    res = Ya + Yb + Yc
    t3 = time.time()

    print(f'train time = {t2 - t1}')
    print(f'predict time = {t3 - t2}')

def boston_test():
    print('\nboston_test:')
    current_path = os.path.dirname(__file__)
    x1_train = read_csv(os.path.join(current_path, 'data/boston_x1_train.csv'))
    x2_train = read_csv(os.path.join(current_path, 'data/boston_x2_train.csv'))
    y_train = read_csv(os.path.join(current_path, 'data/boston_y_train.csv'))
    x1_test = read_csv(os.path.join(current_path, 'data/boston_x1_test.csv'))
    x2_test = read_csv(os.path.join(current_path, 'data/boston_x2_test.csv'))
    y_test = read_csv(os.path.join(current_path, 'data/boston_y_test.csv'))

    model = S3PLR()
    t1 = time.time()
    model.fit(x1_train, x2_train, y_train)
    t2 = time.time()
    Ya, Yb, Yc = model.predict(x1_test, x2_test)
    res = Ya + Yb + Yc
    t3 = time.time()

    print(f'train time = {t2 - t1}')
    print(f'predict time = {t3 - t2}')


def main():
    diabetes_test()
    boston_test()


if __name__ == "__main__":
    res = main()
