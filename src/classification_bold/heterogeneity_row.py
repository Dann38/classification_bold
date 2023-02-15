import pytesseract
import classification_bold.eigengray as eigengray
TESSERACT_CONFIG_ROW = "-l rus --psm 7 "


def is_heterogeneity_row(img_row):
    words_row = pytesseract.image_to_data(img_row, config="-l rus+eng --psm 7", output_type=pytesseract.Output.DICT)
    array = []
    N = len(words_row["word_num"])
    for i in range(N):
        if words_row["level"][i] == 5:
            x0, y0, w0, h0 = words_row["left"][i], words_row["top"][i], words_row["width"][i], words_row["height"][i]
            word_img = img_row[y0:y0 + h0, x0:x0 + w0]
            array.append(eigengray(word_img))
