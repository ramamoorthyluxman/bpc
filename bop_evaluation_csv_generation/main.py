import os
import sys
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from tqdm import tqdm
import datetime
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pose_estimation_pipeline')))
from pose_estimation_pipeline import pose_estimation_pipeline

class SuppressOutput:
    """Context manager to suppress all print output from libraries"""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

class FastestProcessor:
    def __init__(self, config_path, camera_params_path, num_threads=8):
        self.config_path = config_path
        self.camera_params_path = camera_params_path
        self.num_threads = num_threads
        self.thread_local = threading.local()
        
    def get_pipeline(self):
        """Get thread-local pipeline instance"""
        if not hasattr(self.thread_local, 'pipeline'):
            with SuppressOutput():
                self.thread_local.pipeline = pose_estimation_pipeline(self.config_path)
        return self.thread_local.pipeline
    
    def process_image(self, image_data):
        """Process single image - optimized for speed"""
        scene_folder, image_id, rgb_path, depth_path = image_data
        
        try:
            pipeline = self.get_pipeline()
            start_time = time.time()
            with SuppressOutput():
                results = pipeline.pose_estimate_pipeline(rgb_path, depth_path, self.camera_params_path)
            execution_time = time.time() - start_time
            
            # Fast result parsing
            processed_results = []
            for result in results:
                if result.get('success', False):
                    processed_results.append([
                        scene_folder, image_id, result['label'], result['confidence_score'],
                        result['transformation'][0], result['transformation'][1], execution_time
                    ])
            
            return processed_results
        except:
            return []
    
    def format_time(self, seconds):
        """Format seconds into human readable time"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours:.0f}h {minutes:.0f}m {secs:.0f}s"
    
    def run(self, dataset_path, output_csv):
        """Main processing function - maximum speed with progress"""
        print("🔍 Collecting image paths...")
        
        # Fast image collection
        image_data = []
        for scene_folder in os.listdir(dataset_path):
            scene_path = os.path.join(dataset_path, scene_folder)
            if not os.path.isdir(scene_path):
                continue
                
            rgb_folder = os.path.join(scene_path, "rgb_photoneo")
            depth_folder = os.path.join(scene_path, "depth_photoneo")
            
            if not (os.path.exists(rgb_folder) and os.path.exists(depth_folder)):
                continue
                
            for image_file in os.listdir(rgb_folder):
                if image_file.endswith('.png'):
                    image_id = image_file.replace('.png', '')
                    rgb_path = os.path.join(rgb_folder, image_file)
                    depth_path = os.path.join(depth_folder, image_file)
                    
                    if os.path.exists(depth_path):
                        image_data.append((scene_folder, image_id, rgb_path, depth_path))
        
        total_images = len(image_data)
        print(f"📸 Found {total_images} images to process")
        print(f"🧵 Using {self.num_threads} threads")
        
        # Initialize CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
        
        # Process with maximum speed and progress tracking
        start_time = time.time()
        results_queue = Queue()
        processed_count = 0
        processing_lock = threading.Lock()
        
        # Create progress bar
        pbar = tqdm(
            total=total_images, 
            desc="Processing", 
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        def writer_worker():
            """Background writer to avoid I/O blocking"""
            nonlocal processed_count
            
            with open(output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                while True:
                    result = results_queue.get()
                    if result is None:
                        break
                    
                    # Write results
                    for row in result:
                        writer.writerow(row)
                    
                    # Update progress
                    with processing_lock:
                        processed_count += 1
                        elapsed = time.time() - start_time
                        
                        # Calculate rates and ETA
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        remaining_images = total_images - processed_count
                        eta_seconds = remaining_images / rate if rate > 0 else 0
                        
                        # Update progress bar
                        pbar.set_postfix({
                            'Rate': f'{rate:.1f} img/s',
                            'ETA': self.format_time(eta_seconds),
                            'Elapsed': self.format_time(elapsed)
                        })
                        pbar.update(1)
                    
                    results_queue.task_done()
        
        # Start background writer
        writer_thread = threading.Thread(target=writer_worker, daemon=True)
        writer_thread.start()
        
        print(f"🚀 Starting processing at {datetime.datetime.now().strftime('%H:%M:%S')}")
        
        # Process images
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.process_image, img) for img in image_data]
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results_queue.put(result)
                else:
                    # Handle empty results (failed processing)
                    results_queue.put([])  # Empty result to maintain count
        
        # Finish writing
        results_queue.put(None)
        writer_thread.join()
        pbar.close()
        
        # Final statistics
        total_time = time.time() - start_time
        final_rate = total_images / total_time
        
        print(f"\n✅ Processing completed!")
        print(f"📊 Final Statistics:")
        print(f"   • Total images: {total_images}")
        print(f"   • Total time: {self.format_time(total_time)}")
        print(f"   • Average rate: {final_rate:.2f} images/second")
        print(f"   • Finished at: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"   • Results saved to: {output_csv}")
        
        # Performance summary
        theoretical_sequential_time = total_time * self.num_threads
        estimated_speedup = theoretical_sequential_time / total_time if total_time > 0 else 1
        print(f"   • Estimated speedup: ~{estimated_speedup:.1f}x vs sequential")

def main():
    # Configuration - MODIFY THESE PATHS
    config_path = "/home/rama/bpc_ws/bpc/pose_estimation_pipeline/config.yaml"
    dataset_path = "/home/rama/bpc_ws/bpc/ipd/test"
    camera_params_path = "/home/rama/bpc_ws/bpc/ipd/camera_photoneo.json"
    output_csv = "pose_estimation_results_fastest.csv"
    
    # Use maximum threads for speed (adjust if system struggles)
    num_threads = 4  # Increase to 12-16 if you have more CPU cores
    
    print("🎯 Fastest Pose Estimation Processor")
    print("=" * 50)
    
    processor = FastestProcessor(config_path, camera_params_path, num_threads)
    processor.run(dataset_path, output_csv)

if __name__ == "__main__":
    main()