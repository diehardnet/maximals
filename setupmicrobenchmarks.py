#!/usr/bin/python3
import enum
import logging
import os
import torch
import timm
import configs
import console_logger
import dnn_log_helper
from typing import Tuple

from setuppuretorch import parse_args, Timer, equal, check_and_setup_gpu, print_setup_iteration, describe_error

# Best candidates to be evaluated
# Linear, LayerNorm, Conv2d --> More compute intensive
# GELU, ReLU, Softmax, Sigmoid --> Less resource demanding activation functions
# Compute layers
LINEAR = "Linear"
CONV2D = "Conv2d"
LAYER_NORM = "LayerNorm"

# Activation layers
GELU = "GELU"
RELU = "ReLU"
SOFTMAX = "Softmax"
SIGMOID = "Sigmoid"

# Composed modules
ATTENTION = "Attention"
BLOCK = "Block"
MLP = "Mlp"
SWIGLU = "SwiGLU"

ALL_MICRO_OPS = [LINEAR, CONV2D, LAYER_NORM, GELU, RELU, SOFTMAX, SIGMOID, ATTENTION]

# Error geometry
SINGLE, LINE, SQUARE, RANDOM, CUBIC = range(5)


def geometry_comparison(output_tensor: torch.tensor, golden_tensor: torch.tensor) -> int:
    # FIXME: This function must be verified
    diff = torch.eq(output_tensor, golden_tensor).to(dtype=int)
    count_non_zero_diff = torch.count_nonzero(diff)
    if count_non_zero_diff == 1:
        return SINGLE
    elif count_non_zero_diff > 1:
        # Use label function to labelling the matrix
        where_is_corrupted = torch.argwhere(diff != 0)

        # Get all positions of X and Y
        all_x_positions = where_is_corrupted[:, 0]
        all_y_positions = where_is_corrupted[:, 1]
        all_z_positions = where_is_corrupted[:, 1]

        # Count how many times each value is in the list
        _, counter_x_positions = torch.unique(all_x_positions, return_counts=True)
        _, counter_y_positions = torch.unique(all_y_positions, return_counts=True)
        _, counter_z_positions = torch.unique(all_z_positions, return_counts=True)

        # Check if any value is in the list more than one time
        row_error = torch.any(counter_x_positions > 1).flatten()
        col_error = torch.any(counter_y_positions > 1).flatten()
        depth_error = torch.any(counter_z_positions > 1).flatten()

        if row_error and col_error:  # square error
            return SQUARE
        elif row_error or col_error:  # row/col error
            return LINE
        elif depth_error:  # cubic
            return CUBIC
        else:  # Random
            return RANDOM


def compare(output_tensor: torch.tensor, golden_tensor: torch.tensor, float_threshold: float,
            output_logger: logging.Logger, ) -> int:
    output_errors = 0

    # First check if the tensors are equal or not
    if equal(rhs=output_tensor, lhs=golden_tensor, threshold=float_threshold) is True:
        return 0
    # ------------ Check error on the whole output -------------------------------------------------------------
    for i, (gold, found) in enumerate(zip(output_tensor, golden_tensor)):
        if abs(gold - found) > float_threshold and output_errors < configs.MAXIMUM_ERRORS_PER_ITERATION:
            output_errors += 1
            if output_logger:
                output_logger.error(f"i:{i} g:{gold:.6e} o:{found:.6e}")
            dnn_log_helper.log_error_detail(f"i:{i} g:{gold:.6e} o:{found:.6e}")

    # Data on output tensor
    has_nan, has_inf, min_val, max_val = describe_error(input_tensor=output_tensor)
    error_detail_out = f"output_t nan:{has_nan} inf:{has_inf} min:{min_val} max:{max_val} "
    # Data on abs differences
    abs_diff = torch.abs(torch.subtract(golden_tensor, output_tensor))
    has_nan_diff, has_inf_diff, min_val_diff, max_val_diff = describe_error(input_tensor=abs_diff)
    error_detail_out += f"diff_t nan:{has_nan_diff} inf:{has_inf_diff} min:{min_val_diff} max:{max_val_diff}"
    return output_errors


def load_microbenchmark(gold_path: str, micro_name: str) -> Tuple[torch.tensor, torch.tensor, torch.nn.Module]:
    input_tensor, output_tensor, forward_call = torch.load(gold_path)
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
                                        dnn_goal=args.microtype, dataset="random",
                                        float_threshold=configs_threshold)

    # Check if device is ok and disable grad
    check_and_setup_gpu()

    # Defining a timer
    timer = Timer()
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    # First step is to load the inputs in the memory
    timer.tic()
    input_tensor, golden_tensor, forward_call = load_microbenchmark(gold_path=args.goldpath,
                                                                    micro_name=args.microbenchmark)
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
        output_tensor = forward_call(input_tensor=input_tensor)
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
                             float_threshold=configs_threshold, output_logger=terminal_logger)

        timer.toc()
        comparison_time = timer.diff_time

        # Reload all the memories after error
        if errors != 0:
            if terminal_logger:
                terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
            del input_tensor
            # Free cuda memory
            torch.cuda.empty_cache()
            input_tensor, golden_tensor, forward_call = load_microbenchmark(gold_path=args.goldpath,
                                                                            micro_name=args.microbenchmark)

        # Printing timing information
        print_setup_iteration(batch_id=None, comparison_time=comparison_time, copy_to_cpu_time=copy_to_cpu_time,
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
