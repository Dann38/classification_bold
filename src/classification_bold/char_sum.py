import numpy as np
import cv2

def char_sum(y):
    rez = [0]
    k = 0
    d = 1
    run = True
    for i in range(len(y)):
        if y[i] > 0.05:
            rez[k] += y[i]
            d += 1
            run = True
        else:
            if run:
                rez[k] = rez[k]/d
                d = 1
                k += 1
                rez.append(0)
                run = False
    return rez


def char_sum_row(row):
    img = (255-cv2.cvtColor(row, cv2.COLOR_BGR2GRAY))
    h, w = img.shape
    img = img[h//4:-h//4, :]
    y = img.mean(0)/255
    y = np.array(char_sum(y))[:-1]
    return y.mean(), y.std()


def styles_rows(img, box):
    array_coef = []
    array_types = []
    N = len(box)
    for j in range(N):
        f, x1, y1, h1, w1 = box[j]
        row = img[y1:y1 + h1, x1:x1 + w1]
        coef = char_sum_row(row)
        array_coef.append(coef[0])

    array_coef = np.array(array_coef)

    for j in range(N):
        if array_coef[j] >= 0.40:
            array_types.append(1)
        else:
            array_types.append(0)
    return array_types, array_coef


def classifier(img, box):
    N = len(box)
    style_rows_, coef_rows_ = styles_rows(img, box)
    style_rows = []
    info = []
    for i in range(N):
        if style_rows_[i] == 1:
            style_rows.append(1)
        else:
            style_rows.append(0)
        info.append(coef_rows_[i])
    return style_rows, info