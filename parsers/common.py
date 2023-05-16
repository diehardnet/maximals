import copy
import datetime
import os
import re
from typing import List, Tuple, Union


def search_header(lines: List[str], log) -> Tuple[Union[str, None], Union[int, None]]:
    for line in lines:
        m = re.match(r".*config=.+/(\S+)\.yaml.*batch_size=(\d+).*", line)
        if m:
            return m.group(1), int(m.group(2))
        m = re.match(r"#HEADER.*model=(\S+) batchsize=(\d+).*", line)
        if m:
            return m.group(1), int(m.group(2))

    for line in lines:
        m = re.match(r".*config.+/(\S+)\.yaml.*", line)
        if m:
            return m.group(1), -1
    for line in lines:
        m = re.match(r"#SERVER_HEADE.*--batchsize (\d+).*--model (\S+) {2}--disableconsolelog", line)
        if m:
            return m.group(2), m.group(1)

    return None, None


def search_error_criticality(err_str: str) -> re.Match:
    # diehardnet
    m = re.match(r"#ERR batch:\d+ critical-img:\d+ i:\d+ g:(\d+) o:(\d+) gt:(\d+)", err_str)
    # Maximals
    if m is None:
        m = re.match(r"#ERR.*critical.*g:(\d+) o:(\d+) gt:(\d+)", err_str)
    return m


def parse_log_file(log_path: str) -> List[dict]:
    # ...log/2022_09_15_16_00_43_PyTorch-c100_res44_test_02_relu6-bn_200_epochs_ECC_OFF_carolinria.log
    pattern = r".*/(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_\S+_ECC_(\S+)_(\S+).log"
    m = re.match(pattern, log_path)
    if m:
        year, month, day, hour, minute, seconds, ecc, hostname = m.groups()
        year, month, day, hour, minute, seconds = [int(i) for i in [year, month, day, hour, minute, seconds]]
        start_dt = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=seconds)
        data_list = list()

        with open(log_path) as log_fp:
            lines = log_fp.readlines()

        config, batch_size = search_header(lines=lines, log=log_path)
        if config is None or batch_size is None or config == "urations":
            raise ValueError(f"Problem on parsing header {log_path}")
        data_dict = dict(start_dt=start_dt, config=config, ecc=ecc, hostname=hostname,
                         logfile=os.path.basename(log_path), batch_size=batch_size)

        last_acc_time = 0
        critical_sdc, evil_sdc, benign_sdc = 0, 0, 0
        for line in lines:
            ct_m = search_error_criticality(err_str=line)
            if ct_m:
                critical_sdc += 1
                golden, output, ground_truth = ct_m.group(1), ct_m.group(2), ct_m.group(3)
                evil_sdc += int(output != ground_truth)
                benign_sdc += int(output == ground_truth and golden != ground_truth)
            elif "critical" in line:
                raise ValueError(f"Not a valid line {line}")

            sdc_m = re.match(r"#SDC Ite:(\d+) KerTime:(\S+) AccTime:(\S+) KerErr:(\d+) AccErr:(\d+)", line)
            if sdc_m:
                it, ker_time, acc_time, ker_err, acc_err = sdc_m.groups()
                last_acc_time = float(acc_time)
                curr_data = copy.deepcopy(data_dict)
                curr_data.update(
                    dict(it=it, ker_time=float(ker_time), acc_time=0, ker_err=ker_err, acc_err=acc_err, sdc=1,
                         critical_sdc=int(critical_sdc != 0), evil_sdc=evil_sdc, benign_sdc=benign_sdc,
                         hostname=hostname)
                )
                data_list.append(curr_data)
                critical_sdc, evil_sdc, benign_sdc = 0, 0, 0

        if data_list:
            data_list[-1]["acc_time"] = last_acc_time
        return data_list
