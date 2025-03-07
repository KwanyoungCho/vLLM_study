import os, time
from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)
import pdb; pdb.set_trace()
llm = LLM(
    model="lmsys/vicuna-7b-v1.3",
    tensor_parallel_size=2,
    speculative_model="yuhuili/EAGLE-Vicuna-7B-v1.3",
    speculative_draft_tensor_parallel_size=1,
    num_speculative_tokens=4,
    # use_v2_block_manager=True,
    disable_log_stats=False,
    # enforce_eager=True,
)

# import pdb; pdb.set_trace()

outputs = llm.generate(prompts, sampling_params)

import time
time.sleep(5)

# llm.start_profile()
# import pdb; pdb.set_trace()
outputs = llm.generate(prompts, sampling_params)

# llm.stop_profile()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

time.sleep(10)


# from vllm import LLM, SamplingParams

# prompts = [
#     "The future of AI is",
# ]
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

# llm = LLM(
#     model="facebook/opt-6.7b",
#     tensor_parallel_size=1,
#     speculative_model="facebook/opt-125m",
#     num_speculative_tokens=5,
#     use_v2_block_manager=True,
#     disable_log_stats=False,
# )

# outputs = llm.generate(prompts, sampling_params)

# import time
# time.sleep(5)

# outputs = llm.generate(prompts, sampling_params)

# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")