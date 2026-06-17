# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import inspect
import logging
import os
from typing import Any, Tuple

import psutil
import torch.distributed as dist

from verl.utils.device import get_torch_device
from verl.utils.logger.aggregate_logger import DecoratorLoggerBase


def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> Tuple[str]:
    """Get current memory usage."""
    assert unit in ["GB", "MB", "KB"]
    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


def log_gpu_memory_usage(head: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"

        if logger is None:
            print(message)
        else:
            logger.log(msg=message, level=level)


def assert_gpu_memory_safe(
    head: str,
    max_used_ratio: float | None = None,
    min_free_gb: float | None = None,
    logger: logging.Logger = None,
):
    """Fail fast when device memory is too close to OOM.

    Set either threshold through arguments or env vars:
    - VERL_GPU_MEMORY_GUARD_MAX_USED_RATIO, e.g. 0.94
    - VERL_GPU_MEMORY_GUARD_MIN_FREE_GB, e.g. 4
    """
    env_ratio = os.getenv("VERL_GPU_MEMORY_GUARD_MAX_USED_RATIO")
    env_free = os.getenv("VERL_GPU_MEMORY_GUARD_MIN_FREE_GB")
    if max_used_ratio is None and env_ratio:
        max_used_ratio = float(env_ratio)
    if min_free_gb is None and env_free:
        min_free_gb = float(env_free)
    if max_used_ratio is None and min_free_gb is None:
        return

    mem_free, mem_total = get_torch_device().mem_get_info()
    used_ratio = 1.0 - (mem_free / mem_total)
    free_gb = mem_free / (1024**3)
    total_gb = mem_total / (1024**3)

    too_full = max_used_ratio is not None and used_ratio >= max_used_ratio
    too_little_free = min_free_gb is not None and free_gb <= min_free_gb
    if not (too_full or too_little_free):
        return

    message = (
        f"GPU memory guard triggered at {head}: "
        f"used_ratio={used_ratio:.4f}, free_gb={free_gb:.2f}, total_gb={total_gb:.2f}, "
        f"max_used_ratio={max_used_ratio}, min_free_gb={min_free_gb}. "
        "Exiting before CUDA OOM."
    )
    if logger is None:
        print(message)
    else:
        logger.error(message)
    raise RuntimeError(message)


def assert_cpu_memory_safe(
    head: str,
    max_used_ratio: float | None = None,
    min_available_gb: float | None = None,
    logger: logging.Logger = None,
):
    """Fail fast when host RAM is too close to OOM.

    Set either threshold through arguments or env vars:
    - VERL_CPU_MEMORY_GUARD_MAX_USED_RATIO, e.g. 0.92
    - VERL_CPU_MEMORY_GUARD_MIN_AVAILABLE_GB, e.g. 32
    """
    env_ratio = os.getenv("VERL_CPU_MEMORY_GUARD_MAX_USED_RATIO")
    env_available = os.getenv("VERL_CPU_MEMORY_GUARD_MIN_AVAILABLE_GB")
    if max_used_ratio is None and env_ratio:
        max_used_ratio = float(env_ratio)
    if min_available_gb is None and env_available:
        min_available_gb = float(env_available)
    if max_used_ratio is None and min_available_gb is None:
        return

    mem = psutil.virtual_memory()
    used_ratio = mem.percent / 100.0
    available_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)

    too_full = max_used_ratio is not None and used_ratio >= max_used_ratio
    too_little_available = min_available_gb is not None and available_gb <= min_available_gb
    if not (too_full or too_little_available):
        return

    message = (
        f"CPU memory guard triggered at {head}: "
        f"used_ratio={used_ratio:.4f}, available_gb={available_gb:.2f}, total_gb={total_gb:.2f}, "
        f"max_used_ratio={max_used_ratio}, min_available_gb={min_available_gb}. "
        "Exiting before host OOM."
    )
    if logger is None:
        print(message)
    else:
        logger.error(message)
    raise RuntimeError(message)


class GPUMemoryLogger(DecoratorLoggerBase):
    """A decorator class to log GPU memory usage.

    Example:
        >>> from verl.utils.debug.performance import GPUMemoryLogger
        >>> @GPUMemoryLogger(role="actor")
        >>> def update_actor(self, batch):
        ...     # real actor update logics
        ...     return
    """

    def __init__(self, role: str, logger: logging.Logger = None, level=logging.DEBUG, log_only_rank_0: bool = True):
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        super().__init__(role, logger, level, rank, log_only_rank_0)

    def __call__(self, decorated_function: callable):
        def f(*args, **kwargs):
            return self.log(decorated_function, *args, **kwargs)

        return f

    def log(self, func, *args, **kwargs):
        name = func.__name__
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = f"Before {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"
        self.logging_function(message)

        output = func(*args, **kwargs)

        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = f"After {name}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"

        self.logging_function(message)
        return output

def log_print(ctn: Any):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    frame = inspect.currentframe().f_back
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    file_name = frame.f_code.co_filename.split('/')[-1]
    print(f"[{file_name}:{line_number}:{function_name}]: {ctn}")
