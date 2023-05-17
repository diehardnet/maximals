#!/usr/bin/python3
import logging
import os
import re
from typing import Tuple

import timm
import torch

import configs
import console_logger
import dnn_log_helper
from setuppuretorch import parse_args, Timer, equal, check_and_setup_gpu, print_setup_iteration, describe_error

# Best candidates to be evaluated
# Linear, LayerNorm, Conv2d --> More compute intensive
# GELU, ReLU, Softmax, Sigmoid --> Less resource demanding activation functions
# Composed modules: Attention, Block, Mlp, SwiGLU

# Error geometry
SINGLE, LINE, SQUARE, CUBIC, RANDOM, NDIM = "SINGLE", "LINE", "SQUARE", "CUBIC", "RANDOM", "{}-DIM"


def find_geometric_format(lhs: torch.tensor, rhs: torch.tensor) -> str:
    """
    Compares two Pytorch tensors of equal dimensions and returns the geometric format of their difference.
    The function first verifies that the tensors have equal dimensions. Then, it finds the index of the first
     non-matching element between the two tensors using the Pytorch function torch.nonzero(torch.ne(tensor1, tensor2)).
     If there is only one non-matching element, the function returns "Single".
    Next, the function checks if the non-matching elements form a line by verifying that all non-matching elements have
    the same second and higher dimensions. If so, the function returns "Line".
    If the non-matching elements form a square, the function verifies that all non-matching elements have the same
    third and higher dimensions and that the second dimension forms a contiguous block. If so, the function returns
    "Square".
    If the non-matching elements form a cube, the function verifies that all non-matching elements have the same
    fourth and higher dimensions, the second dimension forms a contiguous block, and the third dimension forms a
    contiguous block. If so, the function returns "Cube".
    If the non-matching elements are higher-dimensional, the function returns the number of dimensions as
     "<number>-dimension".
    If the non-matching elements do not fit any of the above formats, the function returns "Irregular".
    """
    # Find the index of the first non-matching element
    index = torch.nonzero(torch.ne(lhs, rhs))
    # Get the coordinates of the first non-matching element
    # coords = tuple(index[0].tolist())

    # Check if the difference is a single element
    if len(index) == 1:
        return SINGLE

    # Check if the difference is a line
    if all(index[:, 1:] == index[0, 1:]):
        return LINE

    # Check if the difference is a square
    if all(index[:, 2:] == index[0, 2:]) and all(index[:, 1] == index[:, 0]):
        return SQUARE

    # Check if the difference is a cube
    if all(index[:, 3:] == index[0, 3:]) and all(index[:, 1:3] == index[0, 1:3]):
        return CUBIC

    # Check if the difference is higher-dimensional
    if len(index[0]) > 4:
        return NDIM.format(len(index[0]))

    # Difference has no special format
    return RANDOM


def compare(output_tensor: torch.tensor, golden_tensor: torch.tensor, float_threshold: float,
            output_logger: logging.Logger, iteration: int) -> int:
    output_errors = 0
    # Make sure that they are on CPU
    out_is_cuda, golden_is_cuda = output_tensor.is_cuda, golden_tensor.is_cuda
    if out_is_cuda or golden_is_cuda:
        dnn_log_helper.log_and_crash(
            fatal_string=f"Tensors are not on CPU. OUT IS CUDA:{out_is_cuda} GOLDEN IS CUDA:{golden_is_cuda}")

    # First check if the tensors are equal or not
    if equal(rhs=output_tensor, lhs=golden_tensor, threshold=float_threshold) is True:
        return 0

    diff_indices = torch.nonzero(torch.ne(output_tensor, golden_tensor))

    # ------------ Check error on the whole output -------------------------------------------------------------
    for idx in diff_indices:
        index = tuple(idx.tolist())
        found = output_tensor[index].item()
        gold = golden_tensor[index].item()
        output_errors += 1
        error_detail = f"i:{index} g:{gold:.6e} o:{found:.6e}"
        if output_logger and output_errors <= 10:
            output_logger.error(error_detail)
        if output_errors < configs.MAXIMUM_ERRORS_PER_ITERATION - 2:  # Minus 2 because the next 2 errors
            dnn_log_helper.log_error_detail(error_detail=error_detail)

    # Data on output tensor
    has_nan, has_inf, min_val, max_val = describe_error(input_tensor=output_tensor)
    error_detail_out = f"output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
    # Data on abs differences
    abs_diff = torch.abs(torch.subtract(golden_tensor, output_tensor))
    has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = describe_error(input_tensor=abs_diff)
    error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
    dnn_log_helper.log_error_detail(error_detail=error_detail_out)
    # # Log the geometry
    # dnn_log_helper.log_error_detail(
    #     f"geometry:{find_geometric_format(lhs=output_tensor, rhs=golden_tensor)}"
    # )
    # Dump the file
    log_helper_file = re.match(r".*LOCAL:(\S+).log.*", dnn_log_helper.log_file_name).group(1)
    save_file = f"{log_helper_file}_sdcit_{iteration}.pt"
    torch.save(output_tensor, save_file)
    dnn_log_helper.log_error_count(output_errors)
    return output_errors


def load_microbenchmark(gold_path: str) -> Tuple[torch.tensor, torch.tensor, torch.nn.Module]:
    input_tensor, output_tensor, forward_call = torch.load(gold_path)
    output_tensor = output_tensor.to("cpu")
    forward_call.zero_grad(set_to_none=True)
    return input_tensor, output_tensor, forward_call


# Force no grad
@torch.no_grad()
def main():
    args, args_text_list = parse_args()
    # Define DNN goal
    configs_threshold = configs.DNN_THRESHOLD[configs.MICROBENCHMARK]
    dnn_log_helper.start_setup_log_file(framework_name="PyTorch",
                                        torch_version=torch.__version__, timm_version=timm.__version__,
                                        gpu=torch.cuda.get_device_name(), args_conf=args_text_list,
                                        dnn_name=configs.MICROBENCHMARK, activate_logging=not args.generate,
                                        dnn_goal="", dataset="", float_threshold=configs_threshold)

    # Check if device is ok and disable grad
    check_and_setup_gpu()

    # Defining a timer
    timer = Timer()
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    # First step is to load the inputs in the memory
    timer.tic()
    input_tensor, golden_tensor, forward_call = load_microbenchmark(gold_path=args.goldpath)
    timer.toc()
    if terminal_logger:
        terminal_logger.debug("\n".join(args_text_list))
        terminal_logger.debug(f"Time necessary to load the inputs and golden: {timer.diff_time_str}")

    # Main setup loop
    setup_iteration = 0
    while setup_iteration < args.iterations:
        # Loop over the input list
        timer.tic()
        dnn_log_helper.start_iteration()
        output_tensor = forward_call(input_tensor)
        torch.cuda.synchronize(device=configs.DEVICE)
        dnn_log_helper.end_iteration()
        timer.toc()
        kernel_time = timer.diff_time
        # Always copy to CPU
        timer.tic()
        output_tensor_cpu = output_tensor.to("cpu")
        timer.toc()
        copy_to_cpu_time = timer.diff_time
        # Then compare the golden with the output
        timer.tic()
        errors = 0
        if args.generate is False:
            errors = compare(output_tensor=output_tensor_cpu, golden_tensor=golden_tensor,
                             float_threshold=configs_threshold, output_logger=terminal_logger,
                             iteration=setup_iteration)

        timer.toc()
        comparison_time = timer.diff_time

        # Reload all the memories after error
        if errors != 0:
            if terminal_logger:
                terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
            del input_tensor
            # Free cuda memory
            torch.cuda.empty_cache()
            input_tensor, golden_tensor, forward_call = load_microbenchmark(gold_path=args.goldpath)

        # Printing timing information
        print_setup_iteration(batch_id=1, comparison_time=comparison_time, copy_to_cpu_time=copy_to_cpu_time,
                              errors=errors, kernel_time=kernel_time, setup_iteration=setup_iteration,
                              terminal_logger=terminal_logger)
        setup_iteration += 1

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
