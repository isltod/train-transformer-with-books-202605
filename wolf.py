import os
import time

import numpy as np
import pandas as pd
import torch
from pylab import plt, mpl


def set_yhilfish_env():
    # 이브 힐피시 금융 인공지능 책에서 계속 사용하는 설정들...
    plt.style.use("seaborn-v0_8")
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.family"] = "serif"
    pd.set_option("mode.chained_assignment", None)
    pd.set_option("display.float_format", "{:.4f}".format)
    np.set_printoptions(suppress=True, precision=4)
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"


def print_torch_gpu():
    if torch.cuda.is_available():
        print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}개\n")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA를 사용할 수 없습니다.")


def get_my_gpu_device(gpu_num):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU {gpu_num}: {torch.cuda.get_device_name(gpu_num)} 사용")
    else:
        device = torch.device("cpu")
        print("CPU 사용")
    return device


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        print(f"소요 시간: {time.perf_counter() - self.start:.5f}초")
