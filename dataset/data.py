import os
import re

from langchain_community.document_loaders import DirectoryLoader

def getfilename(file_name):
    """去除文件的后缀"""
    text = re.sub('\.[^\.]+$', '', file_name)
    return text


def clean_pdf(text):
    """数据清洗函数1：员工手册pdf"""
    text = re.sub(r'编号[：:][a-zA-Z\d]+(?:\n\n点动科技员工手册\n\n)|\n\n第\s*[\d]{1,3}\s*页\s*共\s*[\d]{1,3}\s*页\n\n|第\s*[\d]{1,3}\s*页\s*共\s*[\d]{1,3}\s*页(?=\n|$)|[\.\s]{3,}[\d]{1,3}', '', text)
    # 去除前后空格
    text = text.strip()
    text = re.sub(r'[\s]*', '', text)
    return text

def clear_02(text):
    text = re.sub(r'广州点动信息科技股份有限公司[\n\s]*[\d]{1,3}\n', '', text)
    text = re.sub(r'[\n\.]+|[\-—\n]{3,}', '', text)
    # 去除前后空格
    text = text.strip()
    text = re.sub(r'[\s]*', '', text)
    return text

def clear_03(text):
    text = re.sub(r'(?<=[。！；;])\n+', '', text)
    text = re.sub(r'(?<![。！；;])\n+', '；', text)
    # 去除前后空格
    text = text.strip()
    text = re.sub(r'[\s]*', '', text)
    return text


path = r'E:\code\GitWork\DDongAI\rag\dataset\old'

file_name = os.listdir(path)
# 列出目录中的所有文件和文件夹名称
for name in os.listdir(path):
    print(name)
    file_name.append(name)

print(file_name)
print(type(file_name))

filename = []

for i in range(len(file_name)):
    filename.append(getfilename(file_name[i]))

for fname in filename:
    if not os.path.exists(f"./new/{fname}.txt"):
        print(f"----------{fname}--准备数据清洗-----------")
        # 指定目录路径

        # 加载源文件
        loader = DirectoryLoader("./old", glob=f"**/{fname}.*", use_multithreading=True, show_progress=True)

        docs = loader.load()

        # print(type(docs))
        # print(docs[0].page_content)

        print(f"file:{len(docs)}")

        # 调用清洗函数
        text = clear_03(docs[0].page_content)

        with open(f"./new/{fname}.txt", "a", encoding="utf-8") as file:
            for line in text:
                file.write(line)
        print(f"----------{fname}--结束数据清洗-----------")
    else:
        print(f"----------{fname}--数据清洗文件已存在-----------")