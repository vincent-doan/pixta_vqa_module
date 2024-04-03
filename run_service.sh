if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <host> <port>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/home/vinhdoanthe/.cache/huggingface 

python3 -m uvicorn vqa_module.main:app --host "$1" --port "$2" --reload