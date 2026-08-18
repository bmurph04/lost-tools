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

        frame_latencies = self.eval_dict['frame']['latencies']
        num_frames = len(frame_latencies)
        total_avg_latency = sum(frame_latencies) / num_frames

        print("\n" + "="*62)
        print("LATENCY BENCHMARK REPORT")
        print("="*62)
        print(f" {'Module':<18} | {'Exec Time':<11} | {'Per-Frame Share':<17} | {'Duty'}")
        print("-" * 62)

        for module in modules_to_report:
            if module == 'frame': continue
            module_raw_latencies = self.eval_dict[module].get('latencies')
            avg_execution_time = sum(module_raw_latencies) / len(module_raw_latencies)

            call_count = len(module_raw_latencies)
            call_frequency = call_count / num_frames
            amortized_latency = avg_execution_time * call_frequency
            frame_percent = (amortized_latency / total_avg_latency) * 100.0

            freq_label = '100%' if call_frequency >= 0.99 else f'{call_frequency*100:3.0f}%'

            print(f" {module:<18} | {avg_execution_time:6.2f} ms | {amortized_latency:6.2f} ms ({frame_percent:4.1f}%) | {freq_label}")
        
        print("-" * 62)
        print(f" Total Frames: {num_frames}")
        print(f" Average Frame Latency: {total_avg_latency:6.2f} ms")
        print(f" True System FPS:       {(1000.0 / total_avg_latency):6.2f} FPS")
        print("="*62 + "\n")


        
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
        if 'latencies' not in metrics:
            metrics['latencies'] = []
            metrics['running_sum'] = 0.0

        # Add the current latency to the list
        metrics['latencies'].append(end_time - start_time)
        metrics['running_sum'] += (end_time - start_time)
        # Recalculate the average latency for the current metric and save it
        metrics['avg_latency'] = metrics['running_sum'] / len(metrics['latencies'])

        # Remove temp_latency_start from dict
        del metrics['temp_latency_start']

    def get_module(self, name, init=False):
        
        if init and name not in self.eval_dict:
            self.eval_dict[name] = {}
        
        return self.eval_dict[name]


    def get_sync_time(self, device: str) -> float:
        """Helper function to synchronize CUDA before taking a timestamp."""
        if 'cuda' in device.lower() and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()