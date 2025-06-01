import os
import sys
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pose_estimation_pipeline')))
from pose_estimation_pipeline import pose_estimation_pipeline

class PoseEstimationProcessor:
    def __init__(self, config_path, camera_params_path, num_threads=4):
        self.config_path = config_path
        self.camera_params_path = camera_params_path
        self.num_threads = num_threads
        self.pipeline = pose_estimation_pipeline(config_path)  # Single pipeline instance
        self.pipeline_lock = threading.Lock()  # Thread safety
        self.results_queue = queue.Queue()
        self.csv_lock = threading.Lock()
        
    def process_single_image(self, image_data):
        """Process a single image"""
        scene_folder, image_id, rgb_image_path, depth_image_path = image_data
        
        try:
            start_time = time.time()
            
            # Use lock if pipeline is not thread-safe
            with self.pipeline_lock:
                results = self.pipeline.pose_estimate_pipeline(
                    rgb_image_path, depth_image_path, self.camera_params_path
                )
            
            execution_time = time.time() - start_time
            
            # Process results
            processed_results = []
            for result in results:
                if result.get('success', False):
                    obj_id = result['label']
                    score = result['confidence_score']
                    transformation = result['transformation']
                    R = transformation[0]
                    t = transformation[1]
                    
                    processed_results.append([
                        scene_folder, image_id, obj_id, score, R, t, execution_time
                    ])
            
            return processed_results
            
        except Exception as e:
            print(f"Error processing {scene_folder}/{image_id}: {e}")
            return []
    
    def collect_image_paths(self, dataset_path):
        """Collect all image paths"""
        image_data = []
        
        for scene_folder in sorted(os.listdir(dataset_path)):
            scene_path = os.path.join(dataset_path, scene_folder)
            if os.path.isdir(scene_path):
                rgb_folder = os.path.join(scene_path, "rgb_photoneo")
                depth_folder = os.path.join(scene_path, "depth_photoneo")
                
                if os.path.exists(rgb_folder) and os.path.exists(depth_folder):
                    for image_file in sorted(os.listdir(rgb_folder)):
                        if image_file.endswith('.png'):
                            image_id = image_file.replace('.png', '')
                            rgb_image_path = os.path.join(rgb_folder, image_file)
                            depth_image_path = os.path.join(depth_folder, image_file)
                            
                            if os.path.exists(depth_image_path):
                                image_data.append((scene_folder, image_id, rgb_image_path, depth_image_path))
        
        return image_data
    
    def process_dataset(self, dataset_path, output_csv):
        """Process entire dataset with threading"""
        print("Collecting image paths...")
        image_data = self.collect_image_paths(dataset_path)
        print(f"Found {len(image_data)} images to process")
        
        # Initialize CSV file
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
        
        start_time = time.time()
        processed_count = 0
        
        # Process with thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.process_single_image, img_data): img_data 
                for img_data in image_data
            }
            
            # Collect results as they complete
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                for future in as_completed(future_to_image):
                    img_data = future_to_image[future]
                    scene_folder, image_id = img_data[0], img_data[1]
                    
                    try:
                        results = future.result()
                        
                        # Write results to CSV (thread-safe with context manager)
                        with self.csv_lock:
                            for result_row in results:
                                writer.writerow(result_row)
                        
                        processed_count += 1
                        if processed_count % 10 == 0:  # Progress update every 10 images
                            progress = processed_count / len(image_data) * 100
                            elapsed = time.time() - start_time
                            rate = processed_count / elapsed
                            eta = (len(image_data) - processed_count) / rate if rate > 0 else 0
                            
                            print(f"Progress: {processed_count}/{len(image_data)} "
                                  f"({progress:.1f}%) - "
                                  f"Rate: {rate:.1f} img/s - "
                                  f"ETA: {eta:.0f}s")
                        
                    except Exception as e:
                        print(f"Failed to process {scene_folder}/{image_id}: {e}")
        
        total_time = time.time() - start_time
        print(f"Processing completed in {total_time:.2f} seconds")
        print(f"Average rate: {len(image_data)/total_time:.2f} images/second")
        print(f"Results saved to {output_csv}")

# Alternative: No-lock version if your pipeline is thread-safe
class PipelinePerThread:
    def __init__(self, config_path, camera_params_path, num_threads=4):
        self.config_path = config_path
        self.camera_params_path = camera_params_path
        self.num_threads = num_threads
        self.thread_local = threading.local()
        self.csv_lock = threading.Lock()
    
    def get_pipeline(self):
        """Get or create pipeline for current thread"""
        if not hasattr(self.thread_local, 'pipeline'):
            self.thread_local.pipeline = pose_estimation_pipeline(self.config_path)
        return self.thread_local.pipeline
    
    def process_single_image(self, image_data):
        """Process single image with thread-local pipeline"""
        scene_folder, image_id, rgb_image_path, depth_image_path = image_data
        
        try:
            start_time = time.time()
            pipeline = self.get_pipeline()
            results = pipeline.pose_estimate_pipeline(
                rgb_image_path, depth_image_path, self.camera_params_path
            )
            execution_time = time.time() - start_time
            
            processed_results = []
            for result in results:
                if result.get('success', False):
                    obj_id = result['label']
                    score = result['confidence_score']
                    transformation = result['transformation']
                    R = transformation[0]
                    t = transformation[1]
                    
                    processed_results.append([
                        scene_folder, image_id, obj_id, score, R, t, execution_time
                    ])
            
            return processed_results
            
        except Exception as e:
            print(f"Error processing {scene_folder}/{image_id}: {e}")
            return []
    
    def collect_image_paths(self, dataset_path):
        """Collect all image paths"""
        image_data = []
        
        for scene_folder in sorted(os.listdir(dataset_path)):
            scene_path = os.path.join(dataset_path, scene_folder)
            if os.path.isdir(scene_path):
                rgb_folder = os.path.join(scene_path, "rgb_photoneo")
                depth_folder = os.path.join(scene_path, "depth_photoneo")
                
                if os.path.exists(rgb_folder) and os.path.exists(depth_folder):
                    for image_file in sorted(os.listdir(rgb_folder)):
                        if image_file.endswith('.png'):
                            image_id = image_file.replace('.png', '')
                            rgb_image_path = os.path.join(rgb_folder, image_file)
                            depth_image_path = os.path.join(depth_folder, image_file)
                            
                            if os.path.exists(depth_image_path):
                                image_data.append((scene_folder, image_id, rgb_image_path, depth_image_path))
        
        return image_data
    
    def process_dataset(self, dataset_path, output_csv):
        """Process entire dataset with threading"""
        print("Collecting image paths...")
        image_data = self.collect_image_paths(dataset_path)
        print(f"Found {len(image_data)} images to process")
        
        # Initialize CSV file
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
        
        start_time = time.time()
        processed_count = 0
        
        # Process with thread pool
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            future_to_image = {
                executor.submit(self.process_single_image, img_data): img_data 
                for img_data in image_data
            }
            
            # Collect results as they complete
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                for future in as_completed(future_to_image):
                    img_data = future_to_image[future]
                    scene_folder, image_id = img_data[0], img_data[1]
                    
                    try:
                        results = future.result()
                        
                        # Write results to CSV (thread-safe with context manager)
                        with self.csv_lock:
                            for result_row in results:
                                writer.writerow(result_row)
                        
                        processed_count += 1
                        if processed_count % 10 == 0:  # Progress update every 10 images
                            progress = processed_count / len(image_data) * 100
                            elapsed = time.time() - start_time
                            rate = processed_count / elapsed
                            eta = (len(image_data) - processed_count) / rate if rate > 0 else 0
                            
                            print(f"Progress: {processed_count}/{len(image_data)} "
                                  f"({progress:.1f}%) - "
                                  f"Rate: {rate:.1f} img/s - "
                                  f"ETA: {eta:.0f}s")
                        
                    except Exception as e:
                        print(f"Failed to process {scene_folder}/{image_id}: {e}")
        
        total_time = time.time() - start_time
        print(f"Processing completed in {total_time:.2f} seconds")
        print(f"Average rate: {len(image_data)/total_time:.2f} images/second")
        print(f"Results saved to {output_csv}")

def main():
    # Configuration
    config_path = "/home/rama/bpc_ws/bpc/pose_estimation_pipeline/config.yaml"
    dataset_path = "/home/rama/bpc_ws/bpc/datasets/dummy"
    camera_params_path = "/home/rama/bpc_ws/bpc/ipd/camera_photoneo.json"
    output_csv = "pose_estimation_results_threaded.csv"
    
    # Choose number of threads (start conservative)
    num_threads = 5  # Increase if your pipeline is truly thread-safe
    
    print(f"Starting threaded processing with {num_threads} threads")
    
    # Use the appropriate processor based on thread safety
    # processor = PoseEstimationProcessor(config_path, camera_params_path, num_threads)
    processor = PipelinePerThread(config_path, camera_params_path, num_threads)  # If thread-safe
    
    processor.process_dataset(dataset_path, output_csv)

if __name__ == "__main__":
    main()