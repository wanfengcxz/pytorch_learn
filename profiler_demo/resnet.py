import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    
    with profile(activities=[ProfilerActivity.CPU], 
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=True,
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
                 ) as prof:
        with record_function("model_inference"):
            model(inputs)
            
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # prof.export_chrome_trace("trace.json")
    # tensorboard --logdir=./log
