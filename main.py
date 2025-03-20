import os

import torch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.globals import set_verbose
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.vllm import VLLM
from langchain_core.callbacks import FileCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.globals import set_debug
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import LLMChain
from typing import Generator
from loguru import logger

logfile = "output.log"

logger.add(logfile, colorize=True, enqueue=True)
handler = FileCallbackHandler(logfile)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("device:", device)
# 准备模型

ebd_model = HuggingFaceEmbeddings(model_name=r"D:\AI_Model\maidalun\bce-embedding-base_v1")
# print(ebd_model)
model_path = r"D:\AI_Model\modelscope\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device,)
# print(model)
# 手动切换到推理模式
model.eval()
# 将模型包装为 LangChain 的 LLM
pipe = pipeline(
    "text-generation",
    # "document-question-answering",
    # "question-answering",
    model=model,
    tokenizer=tokenizer,
    max_length=10000,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True,
    temperature=0.2,
)
llm = HuggingFacePipeline(pipeline=pipe)



# vllm_kwargs={
#     "max_model_len":"88000",
#     "gpu_memory_utilization":"0.95",
# }
#
# llm = VLLM(
#     model=model,
#     tensor_parallel_size=1,
#     max_model_len=125,
#     gpu_memory_utilization=0.9
# )

# llm = ChatOpenAI(
#     model="gpt-4o",
#     base_url='https://api.aihao123.cn/luomacode-api/open-api/v1',
#     api_key="sk-4bip3gmlp5dd3qiu9qc1uirjldv73n6o9j82vv309lvo8r3j"
# )


# llm = ChatTongyi(model="qwen-plus", api_key="sk-955a716c2390445388d3fc4e33d14e1e")



set_debug(True)
set_verbose(True)

# print(llm.invoke("你好"))

# 加载文件
loader = DirectoryLoader("data", glob="**/*", use_multithreading=True,show_progress=True)
print(loader)
docs = loader.load()
print("file：", len(docs))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 长文本切割
doc = splitter.split_documents(docs)

print("file chunk：", len(doc))

# 数据库路径
db_dir = "./db"

# ebd_model = HuggingFaceEmbeddings(model_name="D:\AI_Model\Ceceliachenen\paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma.from_documents(doc, ebd_model, persist_directory=db_dir)
# 向量存储
vectorstore.persist()
print("保存完成！")

# 是否可以切更小的块但是多匹配几个？？？？
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1},)

# 创建一个问题的模板
system_prompt = """你是一个很有用的助手，能根据文档的内容和历史对话记录回答用户提的问题。\n
回答问题时，不要补全用户问题，回答要精简概括，控制在50字以内\n
{context}
"""
prompt = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录  模板
    [
        ("system", system_prompt),
        # MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 得到chain
chain1 = create_stuff_documents_chain(llm, prompt)


chain2 = create_retrieval_chain(retriever, chain1)


# resp = chain2.invoke({
#     'input': "怎么请假",
#     "chat_history": [
#         ("human", "你好，我是鲁滨孙"),
#         ("ai", "你好，有什么可以帮您的")
#     ]
# })
# resp = chain2.invoke({
#     "input": "请假流程是怎样的",
#     "chat_history": [
#       ["human", "你好，我是鲁滨孙"],
#       ["ai", "你好，有什么可以帮您的"],
#       ["human", "请假流程是怎样的"],
#       ["ai", "请假先线上跟领导申请报备，事后再提交请假条"]
#     ]
# })
#
# print(resp['answer'])



from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Tuple, Generator

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
# 6、添加路由访问地址Adding chain route,
""" invoke stream
post请求，地址：localhost:8000/chain/invoke
请求参数：
    {
    'input': "今天天气怎么样",
    "chat_history": [
        ("human", "你好，我是鲁滨孙"),
        ("ai", "你好，有什么可以帮您的")
    ]
}
"""
"""历史消息参数示例
{
  "input": {
    "input": "我刚刚问了什么",
    "chat_history": [
      ["human", "你好，我是鲁滨孙"],
      ["ai", "你好，有什么可以帮您的"],
      ["human", "请假流程是怎样的"],
      ["ai", "请假先线上跟领导申请报备，事后再提交请假条"]
    ]
  },
  "config": {},
  "kwargs": {}
}
"""
# 配置
API_KEY = "ddrag_83721"
ALGORITHM = "HS256"

# JWT 数据模型
class TokenData(BaseModel):
    username: str
    role: str

# 解码 JWT Token
# def decode_token(token: str) -> TokenData:
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         return TokenData(**payload)
#     except JWTError:
#         raise HTTPException(status_code=401, detail="Invalid token")

# JWT Token 验证
# def verify_token(token: str = Header(None)):
#     if not token:
#         raise HTTPException(status_code=401, detail="Token missing")
#     return decode_token(token)

def verify_api_key(api_key: str = Header(None)):
    if api_key is API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

class ChatInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]]  # 使用元组表示聊天历史

add_routes(
    app,
    chain2,
    path="/chain2",
    input_type=dict,
    dependencies=[Depends(verify_api_key)]
)

from langchain.retrievers import EnsembleRetriever
# 上传文件

class fileInput(BaseModel):
    input: str
    file: str

# @app.post("/fileChat")
def filechat(fileinput: str) ->Runnable:
    file = fileinput
    file_chunk = splitter.split_text(file)
    file_vector = Chroma.from_texts(file_chunk, ebd_model)
    file_retriever = file_vector.as_retriever(search_type="similarity", search_kwargs={"k": 1}, )
    weighted_retriever = EnsembleRetriever(retrievers=[file_retriever, retriever], weights=[0.7, 0.3])
    sys_prompt = """你是一个很有用的助手，能根据文档的内容和历史对话记录回答用户提的问题。\n
    回答问题时，不要补全用户问题，回答要精简概括，控制在50字以内\n
    {context}
    """
    prompts = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录  模板
        [
            ("system", sys_prompt),
            ("human", "input"),
        ]
    )
    # 得到chain
    prompts_chain = create_stuff_documents_chain(llm, prompts)
    file_chain = create_retrieval_chain(weighted_retriever, prompts_chain)
    # resp = chain2.invoke(input)
    # print(resp)
    return file_chain

# 自定义 Runnable 类，用于包装 filechat
class FileChatRunnable(Runnable):
    def __init__(self, create_fn):
        self.create_fn = create_fn

    def invoke(self, input_data: fileInput, **kwargs):
        print(input_data)
        # 调用 create_filechat_chain 并传递输入参数
        chain = self.create_fn(input_data)
        # 调用 chain
        return chain.invoke(input_data)


# 定义处理函数
# def process_input(file_input: fileInput):
#     # 对输入内容进行处理
#     input = file_input["input"]
#     file = file_input["file"]
#     return {"input": input, "file": file}

# chain3 = RunnableLambda(process_input) | filechat()


chain3 = FileChatRunnable(filechat)


add_routes(
    app,
    chain3,
    path="/file",
    input_type=fileInput,
    dependencies=[Depends(verify_api_key)]
)


@app.post("/file_vector")
def file_vector(fileinputs: fileInput):
    return filechat(fileinputs.file).invoke(fileinputs.input)



# 服务入口函数
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)








#
# contextualize_q_system_prompt = """"给定聊天历史和最新的用户问题，"
#     "该问题可能引用聊天历史中的上下文，"
#     "重新构造一个可以在没有聊天历史的情况下理解的独立问题。"
#     "如果需要，不要回答问题，只需重新构造问题并返回。""""
#
# retriever_history_temp = ChatPromptTemplate.from_messages(
#     [
#         ('system', contextualize_q_system_prompt),
#         MessagesPlaceholder('chat_history'),
#         ("human", "{input}"),
#     ]
# )
#
# # 创建一个子链
# history_chain = create_history_aware_retriever(model, retriever, retriever_history_temp)
#
# # 保持问答的历史记录
# store = {}
#
#
# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]
#
#
# # 创建父链chain: 把前两个链整合
# chain = create_retrieval_chain(history_chain, chain1)
#
# result_chain = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key='input',
#     history_messages_key='chat_history',
#     output_messages_key='answer'
# )
#
# resp1 = result_chain.invoke(
#     # {'input': '请假流程是什么?'},
#     {'input': 'xtuner是什么?'},
#     config={'configurable': {'session_id': 'zs123456'}}
# )
#
# print(resp1['answer'])

