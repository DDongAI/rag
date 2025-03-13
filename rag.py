import torch
from fastapi import FastAPI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.globals import set_verbose
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain_core.globals import set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("device:", device)
# 准备模型

model_path = r"D:\AI_Model\modelscope\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device).half()
pipe = pipeline(
    "text-generation",
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

ebd_model = HuggingFaceEmbeddings(model_name=r"D:\AI_Model\maidalun\bce-embedding-base_v1")

vectorstore = Chroma.from_documents(doc, ebd_model, persist_directory=db_dir)
# 向量存储
vectorstore.persist()
print("保存完成！")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1}, )

# 创建一个问题的模板
system_prompt = """你是一个很有用的助手，能根据文档的内容和历史对话记录回答用户提的问题。\n
{context}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 得到chain
chain1 = create_stuff_documents_chain(llm, prompt)

chain2 = create_retrieval_chain(retriever, chain1)

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
    chain2,
    path="/chain2",
    input_type=dict
)

# 服务入口函数
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
