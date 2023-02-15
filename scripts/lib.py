import os
import cv2
import numpy as np

PATH_IMG = r"C:\Users\danii\program\python\project\classification_bold\scripts\img-row"

WIDTH = 600
HEIGHT = 800
TEXT_IMG = 0.3
OFFSET_ROW = 2
BOLD_ROW = 1
REGULAR_ROW = 0

COLOR_OFFSET_ROW = (0, 0, 255)
COLOR_BOLD_ROW = (255, 0, 0)
COLOR_REGULAR_ROW = (0, 255, 0)


def get_row_image(path_img):
    img = read_img(path_img)
    data_rows = np.load(f"{path_img}.npy")
    image_rows = []

    for j in range(len(data_rows)):
        f, x1, y1, h1, w1 = data_rows[j]
        image_rows.append(img[y1:y1 + h1, x1:x1 + w1])

    rez = {
        "img": img,
        "npy": data_rows,
        "image_rows": image_rows}
    return rez


def read_img(name_file):
    """
    Открывает изображения cv2 в которых есть кириллические символы
    """
    with open(name_file, "rb") as f:
        chunk = f.read()
    chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
    img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
    return img


def evaluation_result(rez): 
    """
    Приведение результата в понятный вид
    """
    bold_count = rez[0]
    regular_count = rez[1]
    bold_right = rez[2]
    regular_right = rez[3]
    TP = bold_right                     # истино-положительное решение
    TN = regular_right
    FN = bold_count - bold_right
    FP = regular_count - regular_right  # ложно-отрицательное решение
    if TP+FP != 0:
        precision = TP/(TP+FP)
        print(f"precision: {precision*100:5.2f}%")
    if TP+FN != 0:
        recall = TP/(TP+FN)
        print(f"recall: {recall*100:5.2f}%")
    

def test_img(box, style_rows):
    bold_count = 0 
    regular_count = 0
    bold_right = 0
    regular_right = 0
    for i in range(len(box)):
        if box[i][0] == BOLD_ROW:
            bold_count += 1
            if style_rows[i] == 1:
                bold_right += 1
        elif box[i][0] == REGULAR_ROW:
            regular_count += 1
            if style_rows[i] == 0:
                regular_right += 1
                
    return [bold_count, regular_count, bold_right, regular_right]


def img_and_box(name_img):
    img = None
    box = None
    npy_path = os.path.join(PATH_IMG, name_img+".npy")
    if os.path.isfile(npy_path):
        print(f"\n{name_img}")
        path_img = os.path.join(PATH_IMG, name_img)
        img = read_img(path_img)
        with open(npy_path, 'rb') as f:
            box = np.load(f).tolist()
    return img, box


def draw_img(img, box, style_rows, info_rows=None):
    h = img.shape[0]
    w = img.shape[1]
    img_cope = img.copy()
    coef = h / w

    for i in range(len(box)):
        color = (155, 155, 155)
        border = 1
        font, x0, y0, h0, w0 = box[i]
        if style_rows[i] == BOLD_ROW:
            color = COLOR_BOLD_ROW
        elif style_rows[i] == OFFSET_ROW:
            color = COLOR_OFFSET_ROW
        elif style_rows[i] == REGULAR_ROW:
            color = COLOR_REGULAR_ROW
        cv2.rectangle(img_cope, (x0, y0), (x0+w0, y0+h0), color, border)
        if info_rows is not None:
            cv2.putText(img_cope, f"{info_rows[i]:.2f}", (x0, y0), cv2.FONT_HERSHEY_COMPLEX, TEXT_IMG, color, 1)
    img = cv2.resize(img_cope, (WIDTH, round(coef*WIDTH)))
    cv2.imshow("img", img)
    cv2.waitKey(0)
    
    
