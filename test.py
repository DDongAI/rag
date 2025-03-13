import torch
from langchain.globals import set_verbose
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("device:", device)
# 准备模型

model_path = r"D:\AI_Model\modelscope\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device).half()
pipe = pipeline(
    "text-generation",
    # "document-question-answering",
    # "question-answering",
    model=model,
    tokenizer=tokenizer,
    max_length=10000,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=pipe)

set_debug(True)
set_verbose(True)

# print(llm.invoke("你好"))

# 加载文件
loader = DirectoryLoader("data", glob="**/*", use_multithreading=True, show_progress=True)
docs = loader.load()
print("file：", len(docs))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 长文本切割
doc = splitter.split_documents(docs)

print("file chunk：", len(doc))

# 数据库路径
db_dir = "./db"

# ebd_model = HuggingFaceEmbeddings(model_name="D:\AI_Model\Ceceliachenen\paraphrase-multilingual-MiniLM-L12-v2")
ebd_model = HuggingFaceEmbeddings(model_name=r"D:\AI_Model\maidalun\bce-embedding-base_v1")

vectorstore = Chroma.from_documents(doc, ebd_model, persist_directory=db_dir)
# 向量存储
vectorstore.persist()
print("保存完成！")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, )

message = """
用简单的文字回答下面的问题，不超过50字

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

# 检索器入链
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

rag_chain.invoke("怎么请假")

from pydantic import BaseModel
from typing import List, Tuple

class ChatMessage(BaseModel):
    role: str  # "human" 或 "ai"
    content: str

class ChatInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]]  # 使用元组表示聊天历史

# 示例数据
data = {
    'input': "今天天气怎么样",
    "chat_history": [
        ("human", "你好，我是鲁滨孙"),
        ("ai", "你好，有什么可以帮您的")
    ]
}

# 使用 Pydantic 模型封装数据
chat_input = ChatInput(**data)

# 输出验证后的数据
print(chat_input)


