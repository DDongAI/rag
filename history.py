import os
import re

from fastapi import FastAPI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.globals import set_verbose
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device_map = "cuda:0"
print("device:", device_map)

# 准备模型


model_path = r"D:\AI_Model\modelscope\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B"

# model_kwargs={"device_map": "auto"}
ebd_model = HuggingFaceEmbeddings(model_name=r"D:\AI_Model\maidalun\bce-embedding-base_v1")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# balanced_low_0
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", )

# 手动切换到推理模式
model.eval()

pipe = pipeline(
    "text-generation",
    # "question-answering",
    model=model,
    tokenizer=tokenizer,
    # 数据加载时的多线程工作线程数
    num_workers=10,

    # 生成文本的最大长度
    # max_length=250,
    # 返回的生成文本数量
    # num_return_sequences=10
    max_new_tokens=50,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True,
    # device_map="balanced_low_0",
)

llm = HuggingFacePipeline(
    pipeline=pipe,
    # 批处理大小，用于加速推理。
    batch_size=10,
    model_kwargs={
        "temperature": 0,
        # "max_tokens":110,
        # "stop":["\n", "###"],
    },
)

set_debug(True)
set_verbose(True)

# 加载文件
loader = DirectoryLoader("data", glob="**/*.*", use_multithreading=True, show_progress=True)

docs = loader.load()

print("file:", len(docs))



splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# 长文本切割
doc = splitter.split_documents(docs)

print("file chunk:", len(doc))

# 数据库路径
db_dir = "./db"

vectorstore = Chroma.from_documents(doc, ebd_model, persist_directory=db_dir)
print("向量化完成完成！")
# 向量存储
vectorstore.persist()
print("保存完成！")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, )

# 创建一个问题的模板
system_prompt = """你是点动科技最聪明的助理，请根据下面context的内容和历史对话记录回答Human提出的问题，请以最简要的方式概括回答，回答尽可能在一百字以内。
context:{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 得到chain
chain1 = create_stuff_documents_chain(llm, prompt)

chain = create_retrieval_chain(retriever, chain1)

# 测试流式输出
# for chunk in chain.stream({"input": "请假流程是什么"}):
#     print(chunk, end="", flush=True)

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

add_routes(
    app,
    chain,
    path="/chain",
    input_type=dict
)

# 服务入口函数
if __name__ == "__main__":
    import uvicorn

    # 10.108.112.34
    uvicorn.run(app, host="localhost", port=8000)

# uvicorn main:app --reload --port=50002 --host=10.108.112.34
