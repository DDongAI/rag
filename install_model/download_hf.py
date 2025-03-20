# 模型下载
from modelscope import snapshot_download

# model_dir = snapshot_download('Ceceliachenen/paraphrase-multilingual-MiniLM-L12-v2',cache_dir='D:/AI_Model/')


# model_dir = snapshot_download('maidalun/bce-embedding-base_v1',cache_dir='D:/AI_Model/')
# model_dir = snapshot_download('maidalun/bce-embedding-base_v1',cache_dir='/home/ddtest/model')
# model_dir = snapshot_download('Qwen/Qwen-72B-Chat-Int8',cache_dir='/home/ddtest/model')
model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-70B', cache_dir='/home/ddtest/model')