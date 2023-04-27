#!/usr/bin/python3
import os
from typing import List, Dict
import timm
import torch
from torchvision import transforms as tv_transforms
import torch_tensorrt

import configs
import console_logger
import dnn_log_helper

from setuppuretorch import Timer, load_dataset, compare, copy_output_to_cpu
from setuppuretorch import parse_args, update_golden, check_dnn_accuracy, check_and_setup_gpu


def load_model(model_name: str, generate: bool, model_tensorrt_path: str) -> [torch.nn.Module, tv_transforms.Compose]:
    if generate is False:
        model = torch.jit.load(model_tensorrt_path)
    else:
        # First option is the baseline option
        model = timm.create_model(model_name, pretrained=True)
        model.eval()

    model = model.to(configs.DEVICE)
    config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.transforms_factory.create_transform(**config)

    return model, transform


def main():
    args, args_text_list = parse_args()
    # Starting the setup
    args_text_list.append(f"GPU:{torch.cuda.get_device_name()}")
    # Define DNN goal
    dnn_goal = configs.DNN_GOAL[args.model]
    dataset = configs.DATASETS[dnn_goal]
    float_threshold = configs.DNN_THRESHOLD[dnn_goal]
    dnn_log_helper.start_setup_log_file(framework_name="PyTorchTensorRT", torch_version=str(torch.__version__),
                                        args_conf=args_text_list, dnn_name=args.model,
                                        activate_logging=not args.generate, dnn_goal=dnn_goal, dataset=dataset,
                                        float_threshold=float_threshold)

    # Check if device is ok and disable grad
    check_and_setup_gpu()
    # Defining a timer
    timer = Timer()
    model_tensorrt_path = args.goldpath.replace(".pt", configs.TENSORRT_FILE_POSFIX)

    # Load the model
    model, transform = load_model(model_name=args.model, generate=args.generate,
                                  model_tensorrt_path=model_tensorrt_path)
    # First step is to load the inputs in the memory
    timer.tic()
    input_list, input_labels = load_dataset(batch_size=args.batchsize, dataset=dataset, test_sample=args.testsamples,
                                            transform=transform)
    timer.toc()
    input_load_time = timer.diff_time_str

    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name) if args.disableconsolelog is False else None

    # Load if it is not a gold generating op
    golden: Dict[str, List[torch.tensor]] = dict(output_list=list(), top_k_labels=list())
    timer.tic()
    if args.generate is False:
        golden = torch.load(args.goldpath)
    else:
        # If tensorrt is selected then convert, only at generate
        input_shape = input_list[-1].shape
        tensorrt_inputs = [torch_tensorrt.Input(input_shape, dtype=torch.float32)]
        model = torch_tensorrt.compile(model, inputs=tensorrt_inputs, enabled_precisions=torch.float32)
    timer.toc()
    golden_load_diff_time = timer.diff_time_str

    if terminal_logger:
        terminal_logger.debug("\n".join(args_text_list))
        terminal_logger.debug(f"Time necessary to load the inputs: {input_load_time}")
        terminal_logger.debug(f"Time necessary to load the golden outputs: {golden_load_diff_time}")

    # Main setup loop
    setup_iteration = 0
    while setup_iteration < args.iterations:
        # Loop over the input list
        batch_id = 0  # It must be like this, because I may reload the list in the middle of the process
        while batch_id < len(input_list):
            timer.tic()
            dnn_log_helper.start_iteration()
            dnn_output = model(input_list[batch_id])
            torch.cuda.synchronize(device=configs.DEVICE)
            dnn_log_helper.end_iteration()
            timer.toc()
            kernel_time = timer.diff_time
            # Always copy to CPU
            timer.tic()
            dnn_output_cpu = copy_output_to_cpu(dnn_output=dnn_output, dnn_goal=dnn_goal)
            timer.toc()
            copy_to_cpu_time = timer.diff_time
            # Then compare the golden with the output
            timer.tic()
            errors = 0
            if args.generate is False:
                errors = compare(output_tensor=dnn_output_cpu,
                                 golden=golden,
                                 ground_truth_labels=input_labels,
                                 batch_id=batch_id,
                                 output_logger=terminal_logger, dnn_goal=dnn_goal, setup_iteration=setup_iteration,
                                 float_threshold=float_threshold)
            else:
                golden = update_golden(golden=golden, output=dnn_output_cpu, dnn_goal=dnn_goal)

            timer.toc()
            comparison_time = timer.diff_time

            # Reload all the memories after error
            if errors != 0:
                if terminal_logger:
                    terminal_logger.info("RELOADING THE MODEL AND THE INPUTS AFTER ERROR")
                del input_list
                del model
                # Free cuda memory
                torch.cuda.empty_cache()
                model, _ = load_model(model_name=args.model, generate=args.generate,
                                      model_tensorrt_path=model_tensorrt_path)
                input_list, input_labels = load_dataset(batch_size=args.batchsize, dataset=dataset,
                                                        test_sample=args.testsamples, transform=transform)

            # Printing timing information
            if terminal_logger:
                wasted_time = comparison_time + copy_to_cpu_time
                time_pct = (wasted_time / (wasted_time + kernel_time)) * 100.0
                iteration_out = f"It:{setup_iteration:<3} batch_id:{batch_id:<3} inference time:{kernel_time:.5f}, "
                iteration_out += f"compare time:{comparison_time:.5f} copy time:{copy_to_cpu_time:.5f} "
                iteration_out += f"(wasted:{time_pct:.1f}%) errors:{errors}"
                terminal_logger.debug(iteration_out)
            batch_id += 1
        setup_iteration += 1

    if args.generate is True:
        torch.save(golden, args.goldpath)
        check_dnn_accuracy(predicted=golden, ground_truth=input_labels, output_logger=terminal_logger,
                           dnn_goal=dnn_goal)
        if args.usetensorrt is True:
            torch.jit.save(model, model_tensorrt_path)

    if terminal_logger:
        terminal_logger.debug("Finish computation.")

    dnn_log_helper.end_log_file()


if __name__ == '__main__':
    try:
        main()
    except Exception as main_function_exception:
        dnn_log_helper.log_and_crash(fatal_string=f"EXCEPTION:{main_function_exception}")
