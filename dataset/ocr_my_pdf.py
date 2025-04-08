# pip install ocrmypdf
from multiprocessing import Process

import ocrmypdf


def ocrmypdf_process():
    ocrmypdf.ocr('input.pdf', 'output.pdf')


def call_ocrmypdf_from_my_app():
    p = Process(target=ocrmypdf_process)
    p.start()
    p.join()


if __name__ == '__main__':  # To ensure correct behavior on Windows and macOS
    ocrmypdf.ocr('input.pdf', 'output.txt', deskew=True)  # 校正图像倾斜

# ocrmypdf.exceptions.MissingDependencyError: Could not find program 'gswin64c' on the PATH

