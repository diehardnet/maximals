#!/usr/bin/python3
import os
import torch
import configs
import console_logger
import dnn_log_helper

from setuppuretorch import parse_args, Timer, equal


def load_micro_input(generate: bool, microbenchmark_type: str) -> torch.tensor:
    return torch.rand(1, 3, 224, 224)


def compare(output_tensor: torch.tensor, golden_tensor: torch.tensor, float_threshold: float) -> int:
    output_errors = 0

    # First check if the tensors are equal or not
    if equal(rhs=output_tensor, lhs=golden_tensor, threshold=float_threshold) is True:
        return 0
    return output_errors


def micro_forward(input_tensor: torch.tensor) -> torch.tensor:
    return input_tensor


def main():
    args, args_text_list = parse_args()
    # Starting the setup
    args_text_list.append(f"GPU:{torch.cuda.get_device_name()}")
    # Define DNN goal
    dnn_goal = configs.DNN_GOAL[args.model]
    dataset = configs.DATASETS[dnn_goal]
    float_threshold = configs.DNN_THRESHOLD[dnn_goal]
    dnn_log_helper.start_setup_log_file(framework_name="PyTorchMicrobenchmarks",
                                        framework_version=str(torch.__version__), args_conf=args_text_list,
                                        dnn_name=args.model, activate_logging=not args.generate, dnn_goal=dnn_goal,
                                        dataset=dataset, float_threshold=float_threshold)

    # Disable all torch grad
    torch.set_grad_enabled(mode=False)
    if torch.cuda.is_available() is False:
        dnn_log_helper.log_and_crash(fatal_string=f"Device {configs.DEVICE} not available.")
    dev_capability = torch.cuda.get_device_capability()
    if dev_capability[0] < configs.MINIMUM_DEVICE_CAPABILITY:
        dnn_log_helper.log_and_crash(fatal_string=f"Device cap:{dev_capability} is too old.")

    # Defining a timer
    timer = Timer()
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    # First step is to load the inputs in the memory
    timer.tic()
    input_micro = load_micro_input(generate=args.generate, microbenchmark_type=args.microtype)
    # Load if it is not a gold generating op
    golden: torch.tensor = torch.load(args.goldpath) if args.generate is False else torch.empty(0)
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
        output_tensor = micro_forward(input_tensor=input_micro)
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
            errors = compare(output_tensor=output_tensor_cpu, golden_tensor=golden, float_threshold=float_threshold)

        timer.toc()
        comparison_time = timer.diff_time

        # Reload all the memories after error
        if errors != 0:
            if terminal_logger:
                terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
            del input_micro
            # Free cuda memory
            torch.cuda.empty_cache()
            input_micro = load_micro_input(generate=args.generate, microbenchmark_type=args.microtype)

        # Printing timing information
        if terminal_logger:
            wasted_time = comparison_time + copy_to_cpu_time
            time_pct = (wasted_time / (wasted_time + kernel_time)) * 100.0
            iteration_out = f"It:{setup_iteration:<3} micro time:{kernel_time:.5f}, "
            iteration_out += f"compare time:{comparison_time:.5f} copy time:{copy_to_cpu_time:.5f} "
            iteration_out += f"(wasted:{time_pct:.1f}%) errors:{errors}"
            terminal_logger.debug(iteration_out)
        setup_iteration += 1

    if args.generate is True:
        torch.save(golden, args.goldpath)

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
