if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <cuda_device_id> <host> <port>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"
export HF_HOME=/home/vinhdoanthe/.cache/huggingface 

python3 -m uvicorn vqa_module.main:app --host "$2" --port "$3" --reload
