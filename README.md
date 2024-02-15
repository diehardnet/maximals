# Maximum corrupted Malicious values (MaxiMals)

MaxiMals is a set of Transformers hardened for reliability.

# Getting started

## Requirements
First, you have to have the following requirements:

- Python 3.10
- Python pip
- [ImageNet dataset](https://www.image-net.org/index.php)

### Reliability evaluation requirements

For the fault simulations and beam experiments:

- For the beam experiments, you will need the scripts from [radhelper](https://github.com/radhelper) repositories 
to control the boards inside the beamline
  - You must have [libLogHelper](https://github.com/radhelper/libLogHelper) 
  installed on the host that controls the GPU and a socket server set up outside the beam room. 
  You can use [radiation-setup](https://github.com/radhelper/radiation-setup) as a socket server.
- For fault simulations, you can use the official version of 
[NVIDIA Bit Fault Injector](https://github.com/NVlabs/nvbitfi) (works until Volta micro-architecture) or 
the version
  we updated for [Ampere evaluations](https://github.com/fernandoFernandeSantos/nvbitfi/tree/new_gpus_support).


### Python libraries installation

Then install all the Python requirements.

```shell
python3 -m pip install -r requeriments.txt
```

## Executing the fault injection/radiation setup

```shell
usage: setuppuretorch.py [-h] [--iterations ITERATIONS] [--testsamples TESTSAMPLES] [--generate] [--disableconsolelog] [--goldpath GOLDPATH] [--checkpointdir CHECKPOINTDIR] [--model MODEL] [--batchsize BATCHSIZE] [--usetorchcompile] [--hardenedid]

PyTorch Maximals radiation setup

options:
  -h, --help            show this help message and exit
  --iterations ITERATIONS
                        Iterations to run forever
  --testsamples TESTSAMPLES
                        Test samples to be used in the test.
  --generate            Set this flag to generate the gold
  --disableconsolelog   Set this flag disable console logging
  --goldpath GOLDPATH   Path to the gold file
  --checkpointdir CHECKPOINTDIR
                        Path to checkpoint dir
  --model MODEL         Model name: resnet50d, tf_efficientnet_b7, vit_base_patch16_224, vit_base_patch32_224.sam, vit_base_patch16_384, vit_large_patch14_clip_224.laion2b_ft_in12k_in1k, vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k,
                        maxvit_large_tf_384.in21k_ft_in1k, maxvit_large_tf_512.in21k_ft_in1k, swinv2_large_window12to16_192to256.ms_in22k_ft_in1k, swinv2_large_window12to24_192to384.ms_in22k_ft_in1k, swinv2_base_window12to16_192to256.ms_in22k_ft_in1k,
                        swinv2_base_window12to24_192to384.ms_in22k_ft_in1k, eva02_large_patch14_448.mim_in22k_ft_in22k_in1k, eva02_base_patch14_448.mim_in22k_ft_in22k_in1k, eva02_small_patch14_336.mim_in22k_ft_in1k
  --batchsize BATCHSIZE
                        Batch size to be used.
  --usetorchcompile     Disable or enable torch compile (GPU Arch >= 700) <-- not currently working
  --hardenedid          Disable or enable HardenedIdentity. Work only for the profiled models.
```

For example, if you want to generate the golden file for the vit_base_patch32_224.sam model using 
the Identity layer hardening, with 40 Images from ImageNet divided into 4 batches, you can use the following command:

```shell
./setuppuretorch.py --testsamples 40 --batchsize 4 \ 
                    --hardenedid --generate \
                    --checkpointdir ./data/checkpoints \ 
                    --goldpath ./data/vit_base_patch32_224.sam_torch_compile_False_hardening_hardenedid.pt \
                    --model vit_base_patch32_224.sam
```

Then to run the same model for 20 iterations:

```shell
./setuppuretorch.py --testsamples 40 --batchsize 4 \ 
                    --hardenedid --iterations 20 \
                    --checkpointdir ./data/checkpoints \ 
                    --goldpath ./data/vit_base_patch32_224.sam_torch_compile_False_hardening_hardenedid.pt \
                    --model vit_base_patch32_224.sam
```

### Using the libLogHelper free version for profiling

TODO: Lucas to put the sample-tool




# Citation

To cite this work:

```bibtex
@unpublished{roquet2023,
  TITLE = {{Cross-Layer Reliability Evaluation and Efficient Hardening of Large Vision Transformers Models}},
  AUTHOR = {Roquet, Lucas and Fernandes dos Santos, Fernando and Rech, Paolo and Traiola, Marcello and Sentieys, Olivier and Kritikakou, Angeliki},
  URL = {https://hal.science/hal-04456702},
  NOTE = {working paper or preprint},
  YEAR = {2024},
  MONTH = Feb,
  KEYWORDS = {Reliability ; Vision transformers ; GPU ; Radiation-induced effects},
}
```

# Colaboration

If you encounter any issues with the code or feel that there is room for improvement,
please feel free to submit a pull request or open an issue.
