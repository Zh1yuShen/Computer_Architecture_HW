import os
import time
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
model_name = "/mnt/rao/home/szy/HBV/model/Llama-3.1-8B-Instruct"



# 加载 tokenizer 和模型

def vllm_client_inference(prompts, max_tokens=1024, temperature=0.7):
    """
    使用 vLLM 提供的 HTTP 服务进行推理。
    Args:
        prompts (list): 输入的文本列表。
        max_tokens (int): 生成的最大Token数。
        temperature (float): 生成的温度参数。
    """
    url = "http://127.0.0.1:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    
    # 构造请求数据
    data = {
        "model": "/mnt/rao/home/szy/HBV/model/Llama-3.1-8B-Instruct",  # 模型名称
        "prompt": prompts,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "n": 1,
        "stop": None
    }
    
    # 测试推理时间
    start_time = time.time()
    response = requests.post(url, headers=headers, json=data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if response.status_code == 200:
        results = response.json()
        print(f"vLLM Inference Time: {elapsed_time:.2f} seconds")
        total_generated_tokens = sum(len(choice['text'].split()) for choice in results["choices"])
        print(f"Total Tokens Generated: {total_generated_tokens}")
        for i, choice in enumerate(results["choices"]):
            print(f"Prompt {i}: {choice['text']}")
        # 保存统计结果
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Specified")
        with open("vllm_inference_stats.txt", "a") as f:
            f.write(f"Prompts: {len(prompts)} | Tokens Generated: {total_generated_tokens} | Time: {elapsed_time:.2f} seconds\n")
    else:
        print(f"Error: {response.status_code}, {response.text}")

# 测试数据
prompts = [
    "Tell me about quantum mechanics.",
    "Write a short story about a space mission.",
    "Describe how photosynthesis works.",
    "Summarize the theory of relativity.",
    "Explain the significance of AI in modern technology."
]

# 调用推理函数

vllm_client_inference(prompts * 2)
vllm_client_inference(prompts * 2)
vllm_client_inference(prompts * 2)
