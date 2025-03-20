import torch
import os
from fastapi import FastAPI,HTTPException, Header, Depends
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.globals import set_verbose
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from pydantic import BaseModel
from typing import List, Tuple
# from loguru import logger
from langchain_core.callbacks import FileCallbackHandler
import re
import logging

logfile = "./log/output.log"
API_KEY = "ddrag_83721"
# logger.add(logfile, colorize=True, enqueue=True)
# handler = FileCallbackHandler(logfile)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(logfile)  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

device_map = "cuda:0,1,2,3,4,5,6,7"

logger.info(f"device:{device_map}")

# 准备模型

# model_path = r"/home/ddtest/model/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
# model_path = r"/home/ddtest/model/tclf90/qwq-32b-gptq-int4"
model_path = r"/home/ddtest/model/Qwen/Qwen-72B-Chat-Int8"

# model_kwargs={"device_map": "auto"}
ebd_model = HuggingFaceEmbeddings(model_name=r"/home/ddtest/model/maidalun/bce-embedding-base_v1")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# balanced_low_0
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto",)

# 手动切换到推理模式
model.eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # max_length=250,
    max_new_tokens=512,
    # num_workers=10,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True,
    # device_map="balanced_low_0",
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    batch_size=10,
    model_kwargs={
        "temperature": 0,
        # "max_tokens":110,
        # "stop":["\n", "###"],
        },
)

set_debug(True)
set_verbose(True)

def clean_pdf(text):
    """数据清洗函数1：员工手册pdf"""
    text = re.sub(r'编号[：:][a-zA-Z\d]+(?:\n\n点动科技员工手册\n\n)|\n\n第\s*[\d]{1,3}\s*页\s*共\s*[\d]{1,3}\s*页\n\n|第\s*[\d]{1,3}\s*页\s*共\s*[\d]{1,3}\s*页(?=\n|$)|[\.\s]{3,}[\d]{1,3}', '', text)
    # 去除前后空格
    text = text.strip()
    text = re.sub(r'[\s]*', '', text)
    return text

if not os.path.exists("./final_data/file.txt"):
    logger.info("----------准备数据清洗-----------")
    # 加载源文件
    loader = DirectoryLoader("/home/ddtest/rag/testdata", glob="**/*.*", use_multithreading=True, show_progress=True)

    docs = loader.load()

    # print(type(docs))
    # print(docs[0].page_content)

    logger.info(f"file:{len(docs)}")

    # 目前就一个文件len(docs)
    text = clean_pdf(docs[0].page_content)

    with open("./final_data/file.txt", "w", encoding="utf-8") as file:
        for line in text:
            file.write(line)
    logger.info("----------结束数据清洗-----------")
else:
    logger.info("----------数据清洗文件已存在-----------")

loader_final = DirectoryLoader("final_data", glob="**/*", use_multithreading=True, show_progress=True)
docs_final = loader_final.load()
logger.info(f"file:{len(docs_final)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

# 长文本切割
doc = splitter.split_documents(docs_final)

logger.info(f"file chunk:{len(docs_final)}")
# print("file chunk:", len(doc))

# 数据库路径
db_dir = "/home/ddtest/rag/db"

vectorstore = Chroma.from_documents(doc, ebd_model, persist_directory=db_dir)
# 向量存储
vectorstore.persist()
logger.info("vectorstore保存完成！")
# print("保存完成！")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}, )

# 创建一个问题的模板
system_prompt = """你是点动科技最聪明的助理，请根据下面context的内容和历史对话记录回答Human提出的问题，请以最简要的方式概括回答。如果context中没有相关的内容，则回答"请您重新提问"
context:{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 得到chain
chain1 = create_stuff_documents_chain(llm, prompt)

chain = create_retrieval_chain(retriever, chain1)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def verify_api_key(api_key: str = Header(None)):
    if api_key is API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


class ChatInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]]

add_routes(
    app,
    chain,
    path="/chain",
    input_type=ChatInput,
    dependencies=[Depends(verify_api_key)]
)

# 服务入口函数
if __name__ == "__main__":
    import uvicorn

    # 10.108.112.34
    uvicorn.run(app, host="10.108.112.34", port=50002)

# uvicorn main:app --reload --port=50002 --host=10.108.112.34
# 环境不一样
