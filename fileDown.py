import os
import re

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 数据库路径
db_dir = "./db"
# 文档路径
# source_directory = "./docs"
# 文件后缀
# file_ext = '*.docx'

ebd_model = HuggingFaceEmbeddings(model_name="D:\AI_Model\Ceceliachenen\paraphrase-multilingual-MiniLM-L12-v2")

loader = DirectoryLoader("data", glob="**/*", use_multithreading=True,show_progress=True)
docs = loader.load()
print(type(docs))
print(docs)
def clean_pdf(text):
    """数据清洗函数1：员工手册pdf"""
    text = re.sub(r'编号[：:][a-zA-Z\d]+(?:\n\n点动科技员工手册\n\n)|\n\n第\s*[\d]{1,3}\s*页\s*共\s*[\d]{1,3}\s*页\n\n|第\s*[\d]{1,3}\s*页\s*共\s*[\d]{1,3}\s*页(?=\n|$)|[\.\s]{3,}[\d]{1,3}', '', text)
    # 去除前后空格
    text = text.strip()
    text = re.sub(r'[\s]*', '', text)
    return text

print(docs[0].page_content)

text = clean_pdf(docs[0].page_content)
print("============================================================================================================")
print("============================================================================================================")
print("============================================================================================================")
print("============================================================================================================")
print("============================================================================================================")
print(text)

if not os.path.exists("./data2/file.txt"):
    with open("./data2/file.txt", "w", encoding="utf-8") as file:
        for line in text:
            file.write(line)
else:
    print("55")

loader_final = DirectoryLoader("data2", glob="**/*", use_multithreading=True,show_progress=True)
docs_final = loader_final.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#
# # 长文本切割
# doc = splitter.split_documents(docs)
doc = splitter.split_documents(docs_final)

print(doc)
#
db = Chroma.from_documents(doc, ebd_model, persist_directory=db_dir)
# db = Chroma.from_texts(doc, ebd_model, persist_directory=db_dir)
# # 文件保存
# db.persist()
#
# # 相似度检索
# search = db.similarity_search("请假", k=1)
# print(search)
