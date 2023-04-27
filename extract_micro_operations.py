#!/usr/bin/python3
import os
from typing import Union, Any

import torch

import configure

from setuppuretorch import check_and_setup_gpu


def hook_fn(module: torch.nn.Module, module_input: torch.tensor,
            module_output: torch.tensor) -> Union[torch.tensor, Any]:
    print(module)
    print("------------Input Grad------------")
    for grad in module_input:
        try:
            print(grad.shape)
        except AttributeError:
            print("None found for Gradient")

    print("------------Output Grad------------")
    for grad in module_output:
        try:
            print(grad.shape)
        except AttributeError:
            print("None found for Gradient")
    print("\n")


def main():
    # Check if device is ok and disable grad
    check_and_setup_gpu()

    # Main setup loop
    current_directory = os.getcwd()
    for torch_compile in configure.TORCH_COMPILE_CONFIGS:
        for dnn_model in configure.ALL_DNNS:
            configuration_name = f"{dnn_model}_torch_compile_{torch_compile}"
            data_dir = f"{current_directory}/data"
            gold_path = f"{data_dir}/{configuration_name}.pt"
            print(f"Extracting layers for {configuration_name}")
            [golden, input_list, input_labels, model, original_dataset_order] = torch.load(gold_path)
            # TODO: Fix hook pass
            for name, param in model.named_parameters():
                # if the param is from a linear and is a bias
                # if "fc" in name and "bias" in name:
                print(name)
                param.register_hook(hook_fn)
            model(input_list[0])
            break
        break

    print("Finish computation.")


if __name__ == '__main__':
    main()
