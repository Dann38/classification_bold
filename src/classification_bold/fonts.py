import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt

FAMILI_COEF = 0.10


class CharFont:
    def __init__(self, img, char):
        self.img = self.trim_char(img)
        self.char = char
        self.hist, _ = np.histogram(img.ravel(), bins=10)

        h, w = self.img.shape[:2]
        self.h = h
        self.count_px = h * w
        if h == 0:
            self.good = 0
        elif w / h < 0.5:
            self.good = 0
        else:
            self.good = 1

    def mean(self):
        return self.img.mean()

    def trim_char(self, img):
        start, end, position_h = self.trim_large_area(img, 0)
        horizon = img[:, start: end]

        start, end, position_v = self.trim_large_area(horizon, 1)
        img_rez = horizon[start: end, :]

        return img_rez

    def trim_large_area(self, img, axis):
        parts = [0]
        start = [0]
        lines = img.mean(axis)
        run = True
        for i in range(len(lines)):
            if lines[i] < 245:
                parts[-1] += 1
                if not run:
                    start.append(i)
                run = True

            else:
                if run:
                    run = False
                    parts.append(0)

        max_index = np.argmax(parts)
        start = start[max_index]
        end = start + parts[max_index]
        position = 0
        if max_index == len(parts) - 1:
            position = -1
        if max_index == 0:
            position = 1
        return start, end, position

    def get_delta_chars(self, char2):
        return sum(abs(self.hist - char2.hist)) / (self.count_px + char2.count_px)


class Font:

    def __init__(self):
        self.chars = {}
        self.size = None
        self.min_char = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                         'а', 'в', 'г', 'е', 'ж', 'з', 'и', 'к', 'л', 'м',
                         'н', 'о', 'п', 'с', 'т', 'х', 'ч', 'ш', 'э', 'ю',
                         'я'];
        self.max_char = ['А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
                         'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф',
                         'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я',
                         'б', 'р', 'у', 'ф'];

    def create_from_img_word(self, img_word, l="rus"):
        if len(img_word.shape) == 3:
            img_word = cv2.cvtColor(img_word, cv2.COLOR_BGR2GRAY)
        h = img_word.shape[0]
        chars_word = pytesseract.image_to_boxes(img_word, config=f"-l {l} --psm 6", output_type=pytesseract.Output.DICT)
        if len(chars_word) != 0:
            count_chars = len(chars_word["char"])
            for i in range(count_chars):
                x0, y0, y1, x1 = chars_word["left"][i], h - chars_word["top"][i], h - chars_word["bottom"][i], \
                                 chars_word["right"][i]
                char_img = img_word[y0:y1, x0:x1]
                char_ = chars_word["char"][i]
                obj_char = CharFont(char_img, char_)
                if obj_char.good > 0.8:
                    self.chars[char_] = obj_char

    def create_from_img_chars(self, char_imgs, chars_):  # ФОРМИРОВАНИЕ ПО СИМВОЛЬНО! ПРОТОТИП
        for i in range(len(chars_)):
            obj_char = CharFont(char_imgs[i], chars_[i])
            if obj_char.good > 0.8:
                self.chars[chars_[i]] = obj_char

    def get_mean(self):

        n = 0
        mean_rez = 0
        for char in self.chars.items():
            if char[0] in self.min_char:
                mean_rez += char[1].mean()
                n += 1
            if char[0] in self.max_char:
                mean_rez += char[1].mean()
                n += 1
        if n == 0:
            return None
        else:
            return mean_rez / n

    def is_bold(self):
        w = self.get_mean()
        if w is None:
            return 0
        else:
            if w < 145:
                return 1
            else:
                return 0

    def get_height(self):
        n = 0
        mean_rez = 0
        for char in self.chars.items():
            if char[0] in self.min_char:
                mean_rez += char[1].h
                n += 1
            if char[0] in self.max_char:
                mean_rez += char[1].h * 0.7
                n += 1
        if n == 0:
            return None
        else:
            return mean_rez / n

    def add_font(self, font):
        chars_new = list(set(font.chars) - set(self.chars))
        chars_exis = list(set(font.chars) & set(self.chars))
        for char_ in chars_new:
            self.chars[char_] = CharFont(font.chars[char_].img, char_)
        for char_ in chars_exis:
            if self.chars[char_].count_px < font.chars[char_].count_px:
                self.chars[char_] = font.chars[char_]

    def family(self, font2, debag=False):
        rez = []
        h1 = self.get_height()
        h2 = font2.get_height()

        if h1 is None or h2 is None:
            return None
        if min(h1, h2) / max(h1, h2) < 0.7:  # ПРОБЛЕМА ТОЧНОЙ ВЫСОТЫ
            return None

        for char1 in self.chars.keys():
            for char2 in font2.chars.keys():
                if char1 == char2:
                    obj_char1 = self.chars[char1]
                    obj_char2 = font2.chars[char2]
                    px_count1 = obj_char1.count_px
                    px_count2 = obj_char2.count_px
                    if min(px_count1, px_count2) / max(px_count1, px_count2) > 0.8:
                        delta = obj_char1.get_delta_chars(obj_char2)
                        if debag:
                            print(f"{char1},{char2}:{delta}")
                        rez.append(delta)
        if len(rez) == 0:
            return None
        else:
            return np.mean(rez)

    def plot_chars(self, m=10, n=5):
        N = min(m * n, len(self.chars))
        for i in range(N):
            ax = plt.subplot(m, n, i + 1)
            ax.imshow(list(self.chars.items())[i][1].img)


def get_words_cord(img_doc_gray):
    words = pytesseract.image_to_data(img_doc_gray, config=f"-l rus --psm 3", output_type=pytesseract.Output.DICT)
    N = len(words["text"])
    words_cord = []
    for i in range(N):
        if words["level"][i] == 5:
            word_x0, word_y0, word_w0, word_h0 = words["left"][i], words["top"][i], words["width"][i], words["height"][
                i]
            word_cord = [0, 0, 0, 0]
            word_cord[0] = word_x0
            word_cord[1] = word_y0
            word_cord[2] = word_x0 + word_w0
            word_cord[3] = word_y0 + word_h0
            words_cord.append(word_cord)
    words_cord = np.array(words_cord)
    return words_cord


def is_rect_in_rect(rect1, rect2):
    return (
            (rect1[0] >= rect2[0]) and
            (rect1[1] >= rect2[1]) and
            (rect1[2] <= rect2[2]) and
            (rect1[3] <= rect2[3])
    )


def font_from_chars_in_rect(img_doc_gray, chars, rect, h_img):
    chars_word = []
    chars_img_word = []

    for k in range(len(chars["char"])):
        x0, y0, y1, x1 = chars["left"][k], h_img - chars["top"][k], h_img - chars["bottom"][k], chars["right"][k]
        char_cord = [x0, y0, x1, y1]
        if is_rect_in_rect(char_cord, rect):
            char_img = img_doc_gray[y0:y1, x0:x1]
            chars_img_word.append(char_img)
            chars_word.append(chars["char"][k])

    font = Font()
    font.create_from_img_chars(chars_img_word, chars_word)
    return font


def union_fonts(fonts, exist_fonts=[]):
    fonts_word_mark = fonts

    for i in range(len(fonts)):
        add_ones = False  # Добавили ли новый шрифт к существующим ОДИН раз
        j = 0
        while j < len(exist_fonts):
            family_coff = exist_fonts[j].family(fonts[i])
            if family_coff is None:
                j += 1
                continue
            elif family_coff < FAMILI_COEF:
                # Если шрифт3 уже был добавлен в шрифт1, но при этом он похож и на шрифт2,
                # то этот шрифт2 нужно тоже добавить к шрифт1
                if add_ones:
                    exist_fonts[fonts_add_index].add_font(exist_fonts.pop(j))
                    j -= 1
                else:
                    exist_fonts[j].add_font(fonts[i])
                    add_ones = True
                    fonts_add_index = j  # Индект шрифта в который добавили
                    fonts_word_mark[i] = exist_fonts[j]  # соотношение слов и шрифтов
            j += 1

        # Если подходящего шрифта нет, то нужно добавить новый
        if not add_ones:
            if len(fonts[i].chars) > 0:
                exist_fonts.append(fonts[i])
                fonts_word_mark[i] = exist_fonts[-1]

    return (exist_fonts, fonts_word_mark)


def get_w_fonts(fonts):
    n = 0
    W = 0
    for font in fonts:
        w = font.is_bold()
        if w is not None:
            W += w
            n += 1

    if n != 0:
        return W / n
    else:
        return 0.0


def styles_rows(fonts_mark_doc, words_cord, box):
    array_coef = []
    array_types = []
    Nj = len(box)
    for j in range(Nj):
        f, x1, y1, h1, w1 = box[j]
        row_cord = [x1, y1, x1 + w1, y1 + h1]
        fonts_row = []
        for k in range(len(words_cord)):
            if is_rect_in_rect(words_cord[k], row_cord):
                fonts_row.append(fonts_mark_doc[k])
        coef = get_w_fonts(fonts_row)
        array_coef.append(coef)

    array_coef = np.array(array_coef)

    for j in range(Nj):
        if array_coef[j] == 1.0:
            array_types.append(1)
        elif array_coef[j] == 0.0:
            array_types.append(0)
        else:
            array_types.append(2)
    return array_types, array_coef


def classifier(img, box):
    hight_img_doc = img.shape[0]
    img_doc_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    chars = pytesseract.image_to_boxes(img_doc_gray, config=f"-l rus --psm 3", output_type=pytesseract.Output.DICT)
    words_cord = get_words_cord(img_doc_gray)
    fonts_rez = []
    fonts_mark_doc = []
    for i in range(len(words_cord)):
        fonts = [font_from_chars_in_rect(img_doc_gray, chars, words_cord[i], hight_img_doc)]
        fonts_rez, fonts_mark_word = union_fonts(fonts, fonts_rez)
        fonts_mark_doc.append(fonts_mark_word[0])
    #     word_index =0
    #     fonts_mark_doc[word_index].plot_chars()
    #     print(fonts_mark_doc[word_index].get_mean())
    N = len(box)
    style_rows_, coef_rows_ = styles_rows(fonts_mark_doc, words_cord, box)
    style_rows = []
    info = []
    for i in range(N):
        if style_rows_[i] == 1:
            style_rows.append(1)
        elif style_rows_[i] == 2:
            style_rows.append(2)
        else:
            style_rows.append(0)
        info.append(coef_rows_[i])
    return style_rows, info