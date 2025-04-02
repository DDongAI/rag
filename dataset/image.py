import os
import re

import cv2
import fitz  # pip install pymupdf
import pytesseract
from PIL import Image


# pdf切出来的图片
IMAGE_PDF = "image_pdf/"
# 生成的txt文件
IMAGE_TXT = "image_txt/"
# ocr语言
ocr_language = 'chi_sim+eng'

def orc_pdf(pdf_file_path: str):
    """ocr识别pdf上的图片，先将pdf按页码切成单个图片，再识别每个图片的内容，每页分别输出一个txt文件"""
    # 自定义tesseract目录
    pytesseract.pytesseract.tesseract_cmd = r'D:\AI_Model\ocr\tesseract.exe'
    # 自定义tessdata目录
    tessdata_dir_config = r'D:\AI_Model\ocr\tessdata"'

    pdf_document = fitz.open(pdf_file_path)

    fname = getfilename(pdf_file_path)

    # 定义缩放因子
    scale_factor = 2

    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)

        pix = page.get_pixmap(dpi=300)

        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        image_file_name = f"{IMAGE_PDF}{fname}_page_{page_number + 1}.png"

        img.save(image_file_name)

        img1_file_path = r"E:/code/GitWork/DDongAI/rag/dataset/" + image_file_name

        if not os.path.exists(img1_file_path):
            print("文件不存在！")
            return "文件不存在！"
        # img1 = cv2.imread(f"{IMAGE_PDF}{fname}_page_{page_number + 1}.png", 0)
        img1 = cv2.imread(img1_file_path, 0)  # 不支持中文路径，即不支持中文命名的文件夹

        if img1 is None:
            print("图片为空！")
            return "图片为空！"

        # text = pytesseract.image_to_string(img1, lang=ocr_language, config='--psm 4')
        text = pytesseract.image_to_string(img1, config=tessdata_dir_config, lang=ocr_language)

        text_file_name = f"{IMAGE_TXT}{fname}_page_{page_number + 1}.txt"

        with open(text_file_name, "w", encoding="utf-8") as text_file:
            text_file.write(text)

    pdf_document.close()


# 批量处理
def getfilename(file_path):
    """去除文件的后缀前缀"""
    text = re.sub('\.[^\.]+$', '', file_path)
    text = re.sub(r'[\s\S]+[\\/](?=[^\\/]+$)', "", text)
    text = re.sub(r"\\", "", text)
    return text


def ocr_dic(dicpath):
    file_name = os.listdir(dicpath)
    # filename = []
    #
    # for i in range(len(file_name)):
    #     filename.append(getfilename(file_name[i]))

    for fname in file_name:
        orc_pdf(dicpath + '\\' + fname)


def ocr_pdf_2(filepath: str):
    text1 = pytesseract.image_to_string(Image.open(filepath), lang='eng')
    print("英文模式识别结果：", text1)


def ocr_pdf_3(filepath: str):
    # 自定义tesseract目录
    pytesseract.pytesseract.tesseract_cmd = r'D:\AI_Model\ocr\tesseract.exe'
    # 自定义tessdata目录
    tessdata_dir_config = r'D:\AI_Model\ocr\tessdata"'

    print(pytesseract.image_to_string(Image.open(filepath), config=tessdata_dir_config, lang='chi_sim+eng'))


if __name__ == '__main__':
    # orc_pdf(r"E:\code\GitWork\DDongAI\rag\dataset\old\企业AI套餐宣传手册325的副本.pdf")
    # ocr_pdf_3(r"E:\code\GitWork\DDongAI\rag\dataset\page_1.png")
    ocr_dic(r"E:\code\GitWork\DDongAI\rag\dataset\old")
