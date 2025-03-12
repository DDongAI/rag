# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
                              cache_dir='D:/AI_Model/modelscope')

# model_dir = snapshot_download('Qwen/Qwen1___5-0___5B-Chat',
#                               cache_dir='D:/AI_Model')
