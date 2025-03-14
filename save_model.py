# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os

# # 저장할 경로 설정
# model_name = "Llama-2-7b-hf"
# save_directory = f"./my_models/{model_name}"
# os.makedirs(save_directory, exist_ok=True)

# # 모델과 토크나이저 다운로드
# print("Downloading model and tokenizer...")
# model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{model_name}")
# tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/{model_name}")

# # 저장
# print(f"Saving model and tokenizer to {save_directory}...")
# model.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)

# print("Model and tokenizer saved successfully!")

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'meta-llama/Llama-2-7b-hf'
quant_path = './my_models/Llama-2-7b-hf-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')