import datetime
import functools
import os
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.distributed as tdist

import infinity.utils.dist as dist

os_system = functools.partial(subprocess.call, shell=True)
def echo(info):
    os_system(f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
def time_str(fmt='[%m-%d %H:%M:%S]'):
    return datetime.datetime.now().strftime(fmt)


class DistLogger(object):
    def __init__(self, lg):
        self._lg = lg
    
    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
    
    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._lg is not None else DistLogger.do_nothing


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    
    def __init__(self, window_size=30, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        return np.median(self.deque) if len(self.deque) else 0
    
    @property
    def avg(self):
        return sum(self.deque) / (len(self.deque) or 1)
    
    @property
    def global_avg(self):
        return self.total / (self.count or 1)
    
    @property
    def max(self):
        return max(self.deque) if len(self.deque) else 0
    
    @property
    def value(self):
        return self.deque[-1] if len(self.deque) else 0
    
    def time_preds(self, counts) -> Tuple[float, str, str]:
        remain_secs = counts * self.median
        return remain_secs, str(datetime.timedelta(seconds=round(remain_secs))), time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remain_secs))
    
    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value)


class MetricLogger(object):
    def __init__(self):
        self.meters = defaultdict(SmoothedValue)
        self.iter_end_t = time.time()
        self.log_iters = set()
        self.log_every_iter = False
    
    def update(self, **kwargs):
        # if it != 0 and it not in self.log_iters: return
        for k, v in kwargs.items():
            if v is None: continue
            if hasattr(v, 'item'): v = v.item()
            # assert isinstance(v, (float, int)), type(v)
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if len(meter.deque):
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return '  '.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, start_it, max_iters, itrt, log_freq, log_every_iter=False, header=''):    # also solve logging & skipping iterations before start_it
        start_it = start_it % max_iters
        self.log_iters = set(range(start_it, max_iters, log_freq))
        self.log_iters.add(start_it)
        self.log_iters.add(max_iters-1)
        self.log_iters.add(max_iters)
        self.log_every_iter = log_every_iter
        self.iter_end_t = time.time()
        self.iter_time = SmoothedValue(fmt='{value:.4f}')
        self.data_time = SmoothedValue(fmt='{value:.3f}')
        header_fmt = header + ':  [{0:' + str(len(str(max_iters))) + 'd}/{1}]'
        
        start_time = time.time()
        if isinstance(itrt, Iterator) and not hasattr(itrt, 'preload') and not hasattr(itrt, 'set_epoch'):
            for it in range(start_it, max_iters):
                obj = next(itrt)
                if it < start_it: continue
                self.data_time.update(time.time() - self.iter_end_t)
                yield it, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if self.log_every_iter or it in self.log_iters:
                    eta_seconds = self.iter_time.avg * (max_iters - it)
                    print(f'{header_fmt.format(it, max_iters)}  eta: {str(datetime.timedelta(seconds=int(eta_seconds)))}  {str(self)}  T: {self.iter_time.value:.3f}s  dataT: {self.data_time.value*1e3:.1f}ms', flush=True)
                self.iter_end_t = time.time()
        else:
            if isinstance(itrt, int): itrt = range(itrt)
            for it, obj in enumerate(itrt):
                if it < start_it:
                    self.iter_end_t = time.time()
                    continue
                self.data_time.update(time.time() - self.iter_end_t)
                yield it, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if self.log_every_iter or it in self.log_iters:
                    eta_seconds = self.iter_time.avg * (max_iters - it)
                    print(f'{header_fmt.format(it, max_iters)}  eta: {str(datetime.timedelta(seconds=int(eta_seconds)))}  {str(self)}  T: {self.iter_time.value:.3f}s  dataT: {self.data_time.value*1e3:.1f}ms', flush=True)
                self.iter_end_t = time.time()
        cost = time.time() - start_time
        cost_str = str(datetime.timedelta(seconds=int(cost)))
        print(f'{header}   Cost of this ep:      {cost_str}   ({cost / (max_iters-start_it):.3f} s / it)', flush=True)


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def check_randomness(args):
    U = 16384
    t = torch.zeros(dist.get_world_size(), 4, dtype=torch.float32, device=args.device)
    t0 = torch.zeros(1, dtype=torch.float32, device=args.device).random_(U)
    t[dist.get_rank(), 0] = float(random.randrange(U))
    t[dist.get_rank(), 1] = float(np.random.randint(U))
    t[dist.get_rank(), 2] = float(torch.randint(0, U, (1,))[0])
    t[dist.get_rank(), 3] = float(t0[0])
    dist.allreduce(t)
    for rk in range(1, dist.get_world_size()):
        assert torch.allclose(t[rk - 1], t[rk]), f't={t}'
    del t0, t, U
