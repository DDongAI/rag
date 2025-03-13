from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 长文本切割
doc = splitter.split_documents(docs)

db = Chroma.from_documents(doc, ebd_model, persist_directory=db_dir)
# 文件保存
db.persist()

# 相似度检索
search = db.similarity_search("请假", k=1)
print(search)
