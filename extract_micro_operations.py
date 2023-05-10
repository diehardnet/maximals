#!/usr/bin/python3
import os
import typing
import pandas as pd
import torch
import configure
import configs

from setuppuretorch import load_data_at_test

_MICRO_BENCHMARKS_DATA = dict()


def get_hook_fn(base_path: str) -> typing.Callable:
    """
    Returns a PyTorch hook function that saves intermediate states of a model
    based on the layer number and operation name.
    """

    def hook_fn(module: torch.nn.Module, module_input: torch.tensor, module_output: torch.tensor):
        global _MICRO_BENCHMARKS_DATA
        device = "cpu"
        # print(module_output.shape)
        # if module.__class__.__name__ == op_name and module.layer_num == layer_num:
        save_path = f"{base_path}_output_size_{module_output.numel()}.pt"
        module_input_cpu = (md_input_i.to(device) for md_input_i in module_input)
        module_output_cpu = (md_output_i.to(device) for md_output_i in module_output)

        _MICRO_BENCHMARKS_DATA[save_path] = [
            module_input_cpu,
            # module.to(device),
            module_output_cpu
        ]

    return hook_fn


def get_all_layers(model: torch.nn.Module, layers_to_extract_from_model: pd.DataFrame,
                   micro_benchmarks_dir: str) -> None:
    layer_types = layers_to_extract_from_model['layer'].to_list()
    parameter_size = layers_to_extract_from_model['layer_params'].to_list()
    for layer_id, (name, layer) in enumerate(model.named_modules()):
        # if re.match(r'.*\.attn(?:_block|_grid)?$', name):
        #     # layer.register_forward_hook(attention_module_hook_fn)
        #     attentions += 1 __class__.__name__
        class_name = layer.__class__.__name__.strip()
        pytorch_total_params = sum(p.numel() for p in layer.parameters())
        op_base_path = f"{micro_benchmarks_dir}/name_{name}_class_name_{class_name}_params_{pytorch_total_params}"
        if class_name in layer_types and pytorch_total_params in parameter_size:
            layer.register_forward_hook(get_hook_fn(base_path=op_base_path))


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
def generate_micro_operations_files(layers_to_extract: pd.DataFrame, models_to_evaluate: list[str],
                                    micro_data_dir: str) -> None:
    print("Extracting the layers")

    device = "cuda:0"
    torch.set_grad_enabled(mode=False)
    # save the selected layers
    current_directory = os.getcwd()
    for torch_compile in configure.TORCH_COMPILE_CONFIGS:
        for dnn_model in models_to_evaluate:
            configuration_name = f"{dnn_model}_torch_compile_{torch_compile}"
            data_dir = f"{current_directory}/data"
            gold_path = f"{data_dir}/{configuration_name}.pt"
            print(f"Extracting layers for {configuration_name}")
            _, _, input_list, model, _ = load_data_at_test(gold_path=gold_path)
            model.zero_grad(set_to_none=True)
            model = model.to(device)
            model.eval()
            # Select data to extract from specific model
            layers_to_extract_from_model = layers_to_extract[layers_to_extract["net"] == dnn_model]
            # Set the path to save
            micro_benchmarks_dir = f"{micro_data_dir}/{dnn_model}"
            os.makedirs(micro_benchmarks_dir, exist_ok=True)

            get_all_layers(model=model, layers_to_extract_from_model=layers_to_extract_from_model,
                           micro_benchmarks_dir=micro_benchmarks_dir)
            input_cuda = input_list[0].to(device)
            model(input_cuda)
    # Saving step
    for path, data in _MICRO_BENCHMARKS_DATA.items():
        print("Saving", path)
        torch.save(data, path)


# Force no grad
@torch.no_grad()
def main():
    micro_data_dir = "data/microbenchmarks"

    models_to_evaluate = [
        # configs.EVA_BASE_PATCH14_448_MIM,
        configs.VIT_LARGE_PATCH14_CLIP_224
    ]

    # Select specific layers that are most resource demanding
    layers_to_extract = select_layers_to_extract(csv_path="data/profile_layers.csv",
                                                 models_to_evaluate=models_to_evaluate)

    # Generate the layers based on the data
    generate_micro_operations_files(layers_to_extract=layers_to_extract, models_to_evaluate=models_to_evaluate,
                                    micro_data_dir=micro_data_dir)


if __name__ == '__main__':
    main()
