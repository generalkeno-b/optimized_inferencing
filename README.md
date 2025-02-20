# Llama Model API

## How to Run

1. Install dependencies:
pip install -r requirements.txt
mkdir models && cd models && git clone --depth 1 --filter=blob:none --sparse https://huggingface.co/mradermacher/vicuna-7b-v1.3-GGUF && cd vicuna-7b-v1.3-GGUF && git sparse-checkout set vicuna-7b-v1.3.IQ3_M.gguf


3. Run the application:
   
python inference.py

5. Access the app:
   
http://localhost:8000/

