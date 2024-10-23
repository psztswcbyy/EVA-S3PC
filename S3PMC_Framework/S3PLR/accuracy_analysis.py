import os
import numpy as np
import csv
from S3PLR import S3PLR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_f2norm(S_res, res):
    up = np.linalg.norm(S_res - res, 'fro')
    down = np.linalg.norm(S_res, 'fro')
    return min(up / down, 1)


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
    model.fit(x1_train, x2_train, y_train)
    Ya, Yb, Yc = model.predict(x1_test, x2_test)
    res = Ya + Yb + Yc

    MAE = mean_absolute_error(y_test, res)
    MSE = mean_squared_error(y_test, res)
    R_Square = r2_score(y_test, res)
    Fnorm = get_f2norm(y_test, res)
    print(f'MAE = {MAE}')
    print(f'MSE = {MSE}')
    print(f'RMSE = {np.sqrt(MSE)}')
    print(f'Fnorm = {Fnorm}')
    print(f'R-Square = {R_Square}')


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
    model.fit(x1_train, x2_train, y_train)
    Ya, Yb, Yc = model.predict(x1_test, x2_test)
    res = Ya + Yb + Yc

    MAE = mean_absolute_error(y_test, res)
    MSE = mean_squared_error(y_test, res)
    R_Square = r2_score(y_test, res)
    Fnorm = get_f2norm(y_test, res)
    print(f'MAE = {MAE}')
    print(f'MSE = {MSE}')
    print(f'RMSE = {np.sqrt(MSE)}')
    print(f'Fnorm = {Fnorm}')
    print(f'R-Square = {R_Square}')


def main():
    diabetes_test()
    boston_test()


if __name__ == "__main__":
    res = main()
