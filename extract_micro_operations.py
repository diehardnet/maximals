#!/usr/bin/python3
import os
import re
from typing import Union, Any

import torch

import configure

from setuppuretorch import check_and_setup_gpu
import setupmicrobenchmarks

_COUNT_OPERATIONS = {}
_HOOK_ATTENTION_COUNTER = 0


def atomic_operation_hook_fn(module: torch.nn.Module, module_input: torch.tensor,
                             module_output: torch.tensor) -> Union[torch.tensor, Any]:
    # TODO: Implement the saving of the microbenchmark code and input/outputs
    module_name = module.__class__.__name__
    global _COUNT_OPERATIONS, _HOOK_ATTENTION_COUNTER
    if module_name in setupmicrobenchmarks.ALL_MICRO_OPS:
        if module_name not in _COUNT_OPERATIONS:
            _COUNT_OPERATIONS[module_name] = 0
        _COUNT_OPERATIONS[module_name] += 1

    if setupmicrobenchmarks.ATTENTION in module_name:
        _HOOK_ATTENTION_COUNTER += 1
        print(module_name)


def get_all_layers(model: torch.nn.Module) -> int:
    attentions = 0
    for name, layer in model.named_modules():
        if re.match(r'.*\.attn(?:_block|_grid)?$', name):
            # layer.register_forward_hook(attention_module_hook_fn)
            attentions += 1
        layer.register_forward_hook(atomic_operation_hook_fn)
    return attentions


# Force no grad
@torch.no_grad()
def main():
    # Check if device is ok and disable grad
    check_and_setup_gpu()
    global _HOOK_ATTENTION_COUNTER
    # Main setup loop
    current_directory = os.getcwd()
    for torch_compile in configure.TORCH_COMPILE_CONFIGS:
        for dnn_model in configure.ALL_DNNS:
            configuration_name = f"{dnn_model}_torch_compile_{torch_compile}"
            data_dir = f"{current_directory}/data"
            gold_path = f"{data_dir}/{configuration_name}.pt"
            print(f"Extracting layers for {configuration_name}")
            [golden, input_list, input_labels, model, original_dataset_order] = torch.load(gold_path)
            model.zero_grad(set_to_none=True)
            # TODO: Fix hook pass
            attention_num = get_all_layers(model)
            model(input_list[0])
            print(f"Attention from hooks {_HOOK_ATTENTION_COUNTER} from counter {attention_num}")
            assert _HOOK_ATTENTION_COUNTER == attention_num
            _HOOK_ATTENTION_COUNTER = 0
    print("Total operations\n", "\n".join(f"{k}={v}" for k, v in _COUNT_OPERATIONS.items()))
    print("Finish computation.")


if __name__ == '__main__':
    main()
