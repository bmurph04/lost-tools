import torch
import time

class SystemEvaluator:
    """
    System evaluator to get various metrics
    """
    def __init__(self, device):
        
        self.device = device
        self.eval_dict = {}

    def print_latency_metrics(self, modules=[]):
        if modules:
            for module in modules:
                assert module in self.eval_dict.keys(), f"Module '{module}' has no metrics so cannot be reported on"

            modules_to_report = modules
            
        else:
            modules_to_report = list(self.eval_dict.keys())
        
        assert 'frame' in modules_to_report, "To print latency metrics, please include total frame latency for FPS computation."

        num_frames = len(self.eval_dict['frame']['latencies'])
        total_avg_latency = self.eval_dict['frame']['avg_latency']

        avg_module_latencies = []

        for module in modules_to_report:
            avg_module_latency = self.eval_dict[module].get('avg_latency')
            avg_module_latencies.append((module, avg_module_latency))

        print("\n" + "="*50)
        print("           LATENCY BENCHMARK REPORT          ")
        print("="*50)
        print(f" Frames Evaluated:      {num_frames}")
        print("-" * 50)

        for (module, avg_latency) in avg_module_latencies:
            if module == 'frame': continue
            print(f"{module}:    {avg_latency:6.2f} ms  ({(avg_latency/total_avg_latency)*100:4.1f}%)")
        
        print("-" * 50)
        print(f" Average Frame Latency:   {total_avg_latency:6.2f} ms")
        print(f"     System FPS:        {(1000.0/total_avg_latency):6.2f} FPS")
        print("="*50 + "\n")


        
        # print("           LATENCY BENCHMARK REPORT          ")
        # print("="*50)
        # print(f" Frames Evaluated:      {len(frames_dir)} (Skipped {WARMUP_FRAMES} warmup frames)")
        # print("-" * 50)
        # print(f" Detector (RF-DETR):    {avg_det:6.2f} ms  ({(avg_det/avg_total)*100:4.1f}%)")
        # print(f" Tracker (Track-On2):   {avg_track:6.2f} ms  ({(avg_track/avg_total)*100:4.1f}%)")
        # print("-" * 50)
        # print(f" Total Frame Latency:   {avg_total:6.2f} ms")
        # print(f" Pure Model FPS:        {fps_inference_only:6.2f} FPS (Detector + Tracker)")
        # print("="*50 + "\n")
    
    
    def get_avg_latency(self, name):
        metrics = self.get_module(name)
        assert metrics is not None, f"{name} hasn't been evaluated yet"
        
        avg_latency = metrics.get('avg_latency')
        if avg_latency is None:
            raise KeyError(f"Error: No latency metrics produced for {name} so cannot report avg_latency")

        return avg_latency

    def start_speed_test(self, name=None):
        metrics = self.get_module(name, init=True)
        metrics['temp_latency_start'] = self.get_sync_time(self.device) * 1000.0
        
    def end_speed_test(self, name=None): 
        # Get the temp_latency_start value for the current metric, raise an error if it cannot be reached
        metrics = self.get_module(name)
        start_time = metrics.get('temp_latency_start')
        
        if start_time is None:
            raise KeyError(f"Error: Tried to run end_speed_test but found no start_speed_test initialization for {name}")

        # Get the end time
        end_time = self.get_sync_time(self.device) * 1000.0

        # Get the list of latencies for the current metric
        metric_latencies = metrics.get('latencies')
        if metric_latencies is None:
            metrics['latencies'] = []
            metric_latencies = metrics['latencies']

        # Add the current latency to the list
        metric_latencies.append(end_time - start_time)
        
        # Recalculate the average latency for the current metric and save it
        avg_metric_latency = sum(metric_latencies) / len(metric_latencies)
        metrics['avg_latency'] = avg_metric_latency

        # Remove temp_latency_start from dict
        del metrics['temp_latency_start']

    def get_module(self, name, init=False):
        
        if self.eval_dict.get(name) is None and init:
            self.eval_dict[name] = {}
        
        return self.eval_dict.get(name, {})


    def get_sync_time(self, device: str) -> float:
        """Helper function to synchronize CUDA before taking a timestamp."""
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()