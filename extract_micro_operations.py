#!/usr/bin/python3
import os
import re
import typing

import pandas as pd
import torch

import configure

import configs


def get_hook_fn(layer_num: str, op_name: str, save_path: str) -> typing.Callable:
    """
    Returns a PyTorch hook function that saves intermediate states of a model
    based on the layer number and operation name.

    Arguments:
        layer_num: layer number
        op_name: operation name
        save_path: file path to save the intermediate states.

    Returns:
        A PyTorch hook function that saves intermediate states of a model.
    """

    def hook_fn(module: torch.nn.Module, module_input: torch.tensor,
                module_output: torch.tensor):
        print(layer_num, op_name, save_path)
        if module.__class__.__name__ == op_name and module.layer_num == layer_num:
            module_save = [module_input, module.parameters(), module, module_output]
            torch.save(module_save, save_path)

    return hook_fn


def get_all_layers(model: torch.nn.Module, layers_to_extract_from_model: pd.DataFrame) -> None:
    for name, layer in model.named_modules():
        # if re.match(r'.*\.attn(?:_block|_grid)?$', name):
        #     # layer.register_forward_hook(attention_module_hook_fn)
        #     attentions += 1 __class__.__name__
        class_name = layer.__class__.__name__
        # TODO: Need to select the layers that will be saved


def select_layers_to_extract(csv_path: str, models_to_evaluate: list[str]) -> pd.DataFrame:
    print("Pre-selecting the layers that will be extracted")
    df = pd.read_csv(csv_path)
    df["var_name_layer"] = df["var_name"]
    df.loc[df["layer"].str.lower().str.contains("block"), "var_name_layer"] = "Block"
    var_names = ['norm', 'attn', 'block', 'mlp', 'stage', "swiglu", "gelu", "act"]
    # filter the dataframe by var_name substrings
    df_filtered = df[df['var_name_layer'].str.lower().str.contains('|'.join(var_names))]

    # group by 'net' and 'var_name', and get the index of the row with the highest 'layer_params'
    idx = df_filtered.groupby(['net', 'var_name_layer'])['output_size'].idxmax()

    # select the rows with the highest 'layer_params' using the index
    result = df_filtered.loc[idx]
    result = result[(result["layer"] != "Identity") &
                    (result["layer"] != "Dropout") &
                    (result["layer"] != "Sequential") &
                    (result["net"].isin(models_to_evaluate))
                    ]
    result = result[~result["var_name"].isin(["norm1", "norm2", "norm_pre"])]
    return result


@torch.no_grad()
def generate_micro_operations_files(layers_to_extract: pd.DataFrame, models_to_evaluate: list[str]) -> None:
    print("Extracting the layers")

    device = "cuda:0"
    # save the selected layers
    current_directory = os.getcwd()
    for torch_compile in configure.TORCH_COMPILE_CONFIGS:
        for dnn_model in models_to_evaluate:
            configuration_name = f"{dnn_model}_torch_compile_{torch_compile}"
            data_dir = f"{current_directory}/data"
            gold_path = f"{data_dir}/{configuration_name}.pt"
            print(f"Extracting layers for {configuration_name}")
            [golden, input_list, input_labels, model, original_dataset_order] = torch.load(gold_path)
            model.zero_grad(set_to_none=True)
            model = model.to(device)
            # Select data to extract from specific model
            layers_to_extract_from_model = layers_to_extract[layers_to_extract["net"] == dnn_model]
            get_all_layers(model=model, layers_to_extract_from_model=layers_to_extract_from_model)
            input_cuda = input_list[0].to(device)
            model(input_cuda)


# Force no grad
@torch.no_grad()
def main():
    models_to_evaluate = [configs.EVA_LARGE_PATCH14_448_MIM, configs.VIT_HUGE_PATCH14_CLIP_224]
    # Select specific layers that are most resource demanding
    layers_to_extract = select_layers_to_extract(csv_path="data/profile_layers.csv",
                                                 models_to_evaluate=models_to_evaluate)

    # Generate the layers based on the data
    generate_micro_operations_files(layers_to_extract=layers_to_extract, models_to_evaluate=models_to_evaluate)


if __name__ == '__main__':
    main()
