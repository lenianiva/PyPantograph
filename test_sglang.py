# copy pasted from https://docs.vllm.ai/en/latest/getting_started/quickstart.html

# do export VLLM_USE_MODELSCOPE=True
import argparse
from typing import Dict, List
import os
import sglang as sgl
from sglang import OpenAI, assistant, gen, set_default_backend, system, user


def test_pytorch():
    print('\n----- Test PyTorch ---')
    # Print the PyTorch version and CUDA version
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Perform a matrix multiplication on CUDA and print the result
    result = torch.randn(2, 4).cuda() @ torch.randn(4, 1).cuda()
    print(f"Matrix multiplication result: {result}")
    
    # Check CUDA availability and device details
    print(f'Number of CUDA devices: {torch.cuda.device_count()}')
    if torch.cuda.device_count() > 0:
        print(f'Device name: {torch.cuda.get_device_name(0)}')
    else:
        print("No CUDA devices available.")

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))



def test_sglang():
    print('\n----- Test sglang ---')
    state = multi_turn_question.run(
        question_1="What is the capital of the United States?",
        question_2="List two local attractions.",
    )

    for m in state.messages():
        print(m["role"], ":", m["content"])

    print("\n-- answer_1 --\n", state["answer_1"])


if __name__ == "__main__":
    import time
    start_time = time.time()
    sgl.set_default_backend(sgl.OpenAI("gpt-4"))

    test_sglang()
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")