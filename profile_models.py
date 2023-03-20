#!/usr/bin/python3
import torch
import torchvision.transforms
import pandas as pd

import configs
import torchinfo
import timm
import gc


def main():
    # Disable all torch grad
    torch.set_grad_enabled(mode=False)
    # Terminal console
    dev_capability = torch.cuda.get_device_capability()
    if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
        raise ValueError(f"Device cap:{dev_capability} is too old.")

    input_sample = torch.rand(3, 600, 600)
    to_pil_transform = torchvision.transforms.ToPILImage()
    input_sample_pil = to_pil_transform(input_sample)
    data_list = list()
    for model_name in configs.ALL_POSSIBLE_MODELS:
        print(f"Profiling {model_name}")
        model = timm.create_model(model_name, pretrained=True).eval().to("cuda:0")
        config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.transforms_factory.create_transform(**config)
        out_sample = torch.stack([transform(input_sample_pil)], dim=0).to("cuda:0")
        info = torchinfo.summary(model=model, input_size=list(out_sample.shape), verbose=0)
        # Freeing must be in this order
        model.cpu()
        out_sample.cpu()
        del out_sample, model, transform, config
        gc.collect()
        torch.cuda.empty_cache()
        for info_i in info.summary_list:
            data_list.append({"net": model_name, "layer": info_i.class_name})

    df = pd.DataFrame(data_list)
    df["count"] = 1
    print(df.groupby(["net", "layer"]).sum())


if __name__ == '__main__':
    main()
