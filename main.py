
import torch
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain


# if torch.cuda.is_available():
#     device = torch.device("cuda:0")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)
# 准备模型
# model_path = "D:\AI_Model\Qwen\Qwen1___5-0___5B-Chat"
model_path = r"D:\AI_Model\modelscope\deepseek-ai\DeepSeek-R1-Distill-Qwen-1___5B"
# model_path = r"D:\AI_Model\modelscope\deepseek-ai\deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device).half()
pipe = pipeline(
    "text-generation",
    # "document-question-answering",
    # "question-answering",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    top_p=1,
    repetition_penalty=1.15,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=pipe)

# print(llm.invoke("你好"))

# 准备提示词
prompt_temp = ChatPromptTemplate.from_messages(
    [
        ('system', "你是点动科技的在线客服，能根据文档的内容回答用户提的问题。如果文档里找不到对应的答案，就回复“您的问题超出了我的认知，请重新提问”"),
        ('human', "{question}")
    ]
)

# 加载文件
loader = DirectoryLoader("data", glob="**/*", use_multithreading=True,show_progress=True)
docs = loader.load()
print(len(docs))

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 长文本切割
splits = splitter.split_documents(docs)

print(len(splits))


chain = prompt_temp | llm
print(chain.invoke({"question": "你是谁"}))