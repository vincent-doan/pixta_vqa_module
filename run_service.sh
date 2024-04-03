CUDA_VISIBLE_DEVICES=1 \
HF_HOME=/home/vinhdoanthe/.cache/huggingface \
python3 -m uvicorn vqa_module.main:app --host 192.168.100.141 --port 2503 --reload