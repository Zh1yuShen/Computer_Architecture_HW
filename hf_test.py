import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型路径
model_name = "/mnt/rao/home/szy/HBV/model/Llama-3.1-8B-Instruct"

def set_gpu_devices(devices):
    """
    设置使用的 GPU 设备。
    Args:
        devices (str): GPU 设备的 ID 列表，例如 "0,1"。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

# 设置使用的 GPU 卡数量（例如：使用 GPU 0 和 GPU 1）
set_gpu_devices("2")

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 设置 pad_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto",low_cpu_mem_usage=True)
print(model.dtype)

def hf_inference(prompts, max_length=1024, batch_size=100):
    """
    Hugging Face 推理函数，支持分批处理。
    Args:
        prompts (list): 输入文本列表
        max_length (int): 最大生成序列长度
        batch_size (int): 每批处理的 Prompt 数量
    """
    total_generated_tokens = 0
    start_time = time.time()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # 转换为张量格式
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        # 推理
        generate_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # 解码生成结果
        results = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # 打印生成结果
        for j, res in enumerate(results):
            print(f"Prompt {i + j + 1}: {res}")

        # 更新生成 Token 数量
        total_generated_tokens += sum(len(output) for output in generate_ids.tolist())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Hugging Face Inference Time: {elapsed_time:.2f} seconds")
    print(f"Total Tokens Generated: {total_generated_tokens}")

    # 保存统计结果
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Specified")
    with open("inference_stats.txt", "a") as f:
        f.write(f"Prompts: {len(prompts)} | Tokens Generated: {total_generated_tokens} | Time: {elapsed_time:.2f} seconds | GPUs: {visible_devices}\n")

# 测试数据
prompts = [
    "Tell me about quantum mechanics.",
    "Write a short story about a space mission.",
    "Describe how photosynthesis works.",
    "Summarize the theory of relativity.",
    "Explain the significance of AI in modern technology."
]

# 调用推理函数
hf_inference(prompts*20, max_length=1024 , batch_size=20)
