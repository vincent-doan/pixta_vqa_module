if [ "$#" -ne 2 ]; then
    echo "Usage: <host> <port>"
    exit 1
fi

export HF_HOME=/home/vinhdoanthe/.cache/huggingface 

python3 -m uvicorn vqa_module.main:app --host "$1" --port "$2" --reload
