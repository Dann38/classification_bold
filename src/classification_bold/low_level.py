import numpy as np


def row_level_0_10(row):
    h, w, _ = row.shape
    row = row[row < 10]
    return len(row)/(h*w)


def styles_rows(img, box):
    array_coef = []
    array_types = []
    N = len(box)
    for j in range(N):
        f, x1, y1, h1, w1 = box[j]
        row = img[y1:y1 + h1, x1:x1 + w1]
        coef = row_level_0_10(row)
        array_coef.append(coef)

    array_coef = np.array(array_coef)
    if array_coef.max() != 0:
        array_coef = array_coef / array_coef.max()

    for j in range(N):
        if array_coef[j] > 0.5:
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