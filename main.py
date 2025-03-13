
import torch
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.globals import set_verbose
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.globals import set_debug
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import add_routes
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import LLMChain


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
loader = DirectoryLoader("data", glob="**/*", use_multithreading=True,show_progress=True)
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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1},)

# 创建一个问题的模板
system_prompt = """你是一个很有用的助手，能根据文档的内容和历史对话记录回答用户提的问题。\n
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
#     # "chat_history": [
#     #     ("human", "你好，我是鲁滨孙"),
#     #     ("ai", "你好，有什么可以帮您的")
#     # ]
# })
#
# print(resp['answer'])

from fastapi import FastAPI
from pydantic import BaseModel


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
    'input': "我是谁",
    "chat_history": [
        ("human", "你好，我是鲁滨孙"),
        ("ai", "你好，有什么可以帮您的")
    ]
}
"""
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

