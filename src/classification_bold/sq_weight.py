import numpy as np
import cv2
from sklearn.cluster import KMeans
from classification_bold.heterogeneity_row import is_heterogeneity_row


def evaluation_fun(img):
    black = (255 - img).mean()
    return black


def styles_words(img, box):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(gray, kernel, iterations=1)
    n_boxes = len(box)
    array = []
    for i in range(n_boxes):
        (font_type, x, y, h, w) = box[i]

        word = dilate[y:y + h, x:x + w]
        black = evaluation_fun(word)
        print((font_type, x, y, h, w))
        heterogeneity_row = is_heterogeneity_row(gray[y:y + h, x:x + w])

        print(heterogeneity_row)
        array.append(black)

    array2 = [array[0]]
    for i in range(1, len(array) - 1):
        array2.append((array[i] + array[i + 1] + array[i - 1]) / 3)
    array2.append(array[-1])
    N = len(array)
    X = np.zeros((N, 2))
    for i in range(N):
        X[i, 0] = array[i]
        X[i, 1] = array2[i]

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)

    bold_style = np.argmax(kmeans.cluster_centers_[:, 0])
    regular_style = np.argmin(kmeans.cluster_centers_[:, 0])

    delta_kmeans = np.max(kmeans.cluster_centers_[:, 0]) - np.min(np.argmin(kmeans.cluster_centers_[:, 0]))
    style_words = kmeans.labels_

    return style_words, bold_style, regular_style, delta_kmeans


def classifier(img, box):
    N = len(box)
    style_words_, bold_style, regular_style, delta_kmeans = styles_words(img, box)
    style_words = []
    info_ = []
    for i in range(N):
        if style_words_[i] == bold_style:
            style_words.append(1)
        else:
            style_words.append(0)
        info_.append(delta_kmeans)
    return style_words, info_
