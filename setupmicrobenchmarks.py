#!/usr/bin/python3
import logging
import os
import torch
import timm
import configs
import console_logger
import dnn_log_helper

from setuppuretorch import parse_args, Timer, equal, check_and_setup_gpu, print_setup_iteration

CONV2D = "conv2d"
BATCH_NORM2D = "batchnorm2d"
RELU = "relu"
LINEAR = "linear"
ALL_MICRO_OPS = [CONV2D, BATCH_NORM2D, RELU, LINEAR]
MICROBENCHMARK_FLOAT_THRESHOLD = 0


# MaxPool2d, Identity, AvgPool2d, AdaptiveAvgPool2d , Flatten,SiLU ,Sigmoid = enum.auto()
# Dropout , LayerNorm ,GELU ,Softmax ,DropPath = enum.auto()


def load_micro_forward_and_input(generate: bool, microbenchmark_type: str) -> tuple[torch.tensor, callable]:
    input_tensor = torch.empty()
    microbenchmark_type = microbenchmark_type.lower()
    if microbenchmark_type == CONV2D:
        forward_operation = torch.nn.Conv2d
    elif microbenchmark_type == BATCH_NORM2D:
        forward_operation = torch.nn.BatchNorm2d
    elif microbenchmark_type == RELU:
        forward_operation = torch.nn.ReLU
    elif microbenchmark_type == LINEAR:
        forward_operation = torch.nn.Linear
    else:
        raise NotImplementedError("Type not implemented")
    return input_tensor, forward_operation


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
    return output_errors


def main():
    args, args_text_list = parse_args()
    # Define DNN goal
    dnn_log_helper.start_setup_log_file(framework_name="PyTorch",
                                        torch_version=torch.__version__, timm_version=timm.__version__,
                                        gpu=torch.cuda.get_device_name(), args_conf=args_text_list,
                                        dnn_name="Microbenchmark", activate_logging=not args.generate,
                                        dnn_goal=args.microtype, dataset="random",
                                        float_threshold=MICROBENCHMARK_FLOAT_THRESHOLD)

    # Check if device is ok and disable grad
    check_and_setup_gpu()

    # Defining a timer
    timer = Timer()
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    # First step is to load the inputs in the memory
    timer.tic()
    input_micro, forward_call = load_micro_forward_and_input(generate=args.generate, microbenchmark_type=args.microtype)
    # Load if it is not a gold generating op
    golden: torch.tensor = torch.load(args.goldpath) if args.generate is False else torch.empty(0)
    output_tensor_cpu: torch.tensor = torch.empty()
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
        output_tensor = forward_call(input_tensor=input_micro)
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
            errors = compare(output_tensor=output_tensor_cpu, golden_tensor=golden,
                             float_threshold=MICROBENCHMARK_FLOAT_THRESHOLD, output_logger=terminal_logger)

        timer.toc()
        comparison_time = timer.diff_time

        # Reload all the memories after error
        if errors != 0:
            if terminal_logger:
                terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
            del input_micro
            # Free cuda memory
            torch.cuda.empty_cache()
            input_micro = load_micro_forward_and_input(generate=args.generate, microbenchmark_type=args.microtype)

        # Printing timing information
        print_setup_iteration(batch_id=None, comparison_time=comparison_time, copy_to_cpu_time=copy_to_cpu_time,
                              errors=errors, kernel_time=kernel_time, setup_iteration=setup_iteration,
                              terminal_logger=terminal_logger)
        setup_iteration += 1

    if args.generate is True:
        torch.save(output_tensor_cpu, args.goldpath)

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
