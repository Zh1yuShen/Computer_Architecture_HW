CUDA_VISIBLE_DEVICES=2,3 python -u -m vllm.entrypoints.openai.api_server \
    --host 127.0.0.1 \
    --port 8000 \
    --model /mnt/rao/home/szy/HBV/model/Llama-3.1-8B-Instruct \
    --tokenizer /mnt/rao/home/szy/HBV/model/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2
