# Maximum Malicious valueS

MaxiMalS is a set of Transformers hardened for reliability.

# Getting started

## Requirements
First, you have to have the following requirements:

- Python 3.10
- Python Pip
- [libLogHelper](https://github.com/radhelper/libLogHelper)
- ImageNet dataset

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

```

