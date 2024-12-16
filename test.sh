curl http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/mnt/rao/home/szy/HBV/model/Llama-3.1-8B-Instruct",
"messages": [{"role": "user", "content": "晚上睡不着觉怎么办？"}],
"max_tokens": 1024,
"temperature": 0
}'
