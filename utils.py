from py3nvml.py3nvml import *
import torch

class Logger:
    def __init__(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(0)
        self.logs = []

    def sizeof_fmt(self, num, suffix='B'):
        for unit in ['','K','M','G','T','P','E','Z']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Y', suffix)

    def sizeof(self, var: torch.Tensor):
        n = torch.numel(var)
        t = var.type()
        if t == 'torch.FloatTensor': return self.sizeof_fmt(n*32)
        elif t == 'torch.LongTensor': return self.sizeof_fmt(n*64)
        elif t == 'torch.DoubleTensor': return self.sizeof_fmt(n*64)
        else: return self.sizeof_fmt(n*32)


    def gpu_mem_log(self, ctx_name:str, var_names:[str], vars:[torch.Tensor]):
        logs = []
        logs.append("{}:\n".format(ctx_name))
        if len(var_names) != len(vars):
            print("Wrong length!")
        for i in range(len(var_names)):
            logs.append("\t{}:\t{}\n".format(var_names[i], self.sizeof(vars[i])))
        info = nvmlDeviceGetMemoryInfo(self.handle)
        logs.append("Usage/Total: {}/{}\n".format(info.used, info.total))
        self.logs += logs