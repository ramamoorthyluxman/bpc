import csv
import os
from collections import defaultdict
# Remove unused matplotlib import that can consume memory
# import matplotlib.pyplot as plt
from process_scene_data import process_scene_data
import sys
import time
import gc
import psutil  # For memory monitoring
import tracemalloc  # For memory debugging
import heapq  # For merging sorted files
import tempfile  # For temporary file handling

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utilities')))
from generate_output_csv import CSVWriter

class SceneImageGroupProcessor:
    def __init__(self, csv_file_path, max_memory_mb=4000, chunk_size=100):
        self.csv_file_path = csv_file_path
        self.csv_writer = CSVWriter('results.csv')
        self.max_memory_mb = max_memory_mb
        self.chunk_size = chunk_size
        
        # Start memory tracing for debugging
        tracemalloc.start()
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def check_memory_limit(self):
        """Check if memory usage exceeds limit"""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_mb:
            print(f"Warning: Memory usage ({current_memory:.1f} MB) exceeds limit ({self.max_memory_mb} MB)")
            return True
        return False
    
    def aggressive_cleanup(self):
        """Perform aggressive memory cleanup"""
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Print memory usage for debugging
        current, peak = tracemalloc.get_traced_memory()
        print(f"Memory: Current={current/1024/1024:.1f}MB, Peak={peak/1024/1024:.1f}MB")
    
    def process_in_chunks(self):
        """
        Process CSV data in chunks with proper grouping.
        Handles groups that may span across chunks correctly.
        """
        chunk_groups = defaultdict(list)
        pending_groups = defaultdict(list)  # Groups that span chunks
        row_count = 0
        group_count = 0
        
        try:
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    scene_id = row['scene_id']
                    image_id = row['image_id']
                    key = (scene_id, image_id)
                    
                    # Add to current chunk
                    chunk_groups[key].append(row)
                    row_count += 1
                    
                    # Process chunk when it gets large enough
                    if row_count >= self.chunk_size:
                        print(f"Processing chunk with {len(chunk_groups)} groups...")
                        
                        # Process complete groups and identify incomplete ones
                        complete_groups, incomplete_groups = self.identify_complete_groups(
                            chunk_groups, file, reader
                        )
                        
                        # Process complete groups
                        for group_key, camera_rows in complete_groups.items():
                            # Add any pending rows for this group
                            if group_key in pending_groups:
                                camera_rows = pending_groups[group_key] + camera_rows
                                del pending_groups[group_key]
                            
                            self.process_single_group(group_key, camera_rows, group_count)
                            group_count += 1
                        
                        # Save incomplete groups for next iteration
                        for group_key, camera_rows in incomplete_groups.items():
                            if group_key in pending_groups:
                                pending_groups[group_key].extend(camera_rows)
                            else:
                                pending_groups[group_key] = camera_rows
                        
                        # Clear processed data and cleanup
                        chunk_groups.clear()
                        complete_groups.clear()
                        incomplete_groups.clear()
                        self.aggressive_cleanup()
                        row_count = 0
                        
                        # Check memory usage
                        if self.check_memory_limit():
                            print("Memory limit exceeded, consider reducing chunk size")
                
                # Process remaining groups (including pending ones)
                all_remaining = dict(chunk_groups)
                for group_key, camera_rows in pending_groups.items():
                    if group_key in all_remaining:
                        all_remaining[group_key] = camera_rows + all_remaining[group_key]
                    else:
                        all_remaining[group_key] = camera_rows
                
                for group_key, camera_rows in all_remaining.items():
                    self.process_single_group(group_key, camera_rows, group_count)
                    group_count += 1
                    
        except FileNotFoundError:
            print(f"Error: Could not find the file '{self.csv_file_path}'")
        except KeyError as e:
            print(f"Error: Column {e} not found in the CSV file")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
        finally:
            if hasattr(self.csv_writer, 'close'):
                self.csv_writer.close()
    
    def identify_complete_groups(self, chunk_groups, file, reader):
        """
        Identify which groups are complete in current chunk by peeking ahead.
        """
        current_pos = file.tell()
        complete_groups = {}
        incomplete_groups = {}
        
        # Get all group keys in current chunk
        chunk_keys = set(chunk_groups.keys())
        
        # Peek ahead to see which groups continue
        continuing_groups = set()
        peek_count = 0
        max_peek = 1000  # Limit peek to avoid memory issues
        
        try:
            for next_row in reader:
                if peek_count >= max_peek:
                    break
                    
                scene_id = next_row['scene_id']
                image_id = next_row['image_id']
                key = (scene_id, image_id)
                
                if key in chunk_keys:
                    continuing_groups.add(key)
                
                peek_count += 1
        except StopIteration:
            pass
        
        # Reset file position
        file.seek(current_pos)
        
        # Categorize groups
        for group_key, camera_rows in chunk_groups.items():
            if group_key in continuing_groups:
                incomplete_groups[group_key] = camera_rows
            else:
                complete_groups[group_key] = camera_rows
        
        return complete_groups, incomplete_groups
    
    def process_single_group(self, group_key, camera_rows, group_num):
        """Process a single group of camera rows"""
        scene_id, image_id = group_key
        start_time = time.time()
        
        print(f"\nGroup {group_num + 1}: Scene {scene_id}, Image {image_id}")
        print(f"Number of cameras: {len(camera_rows)}")
        print(f"Memory usage: {self.get_memory_usage():.1f} MB")
        print("-" * 40)
        
        scene_info = None
        try:
            scene_info = process_scene_data(scene_data=camera_rows)
            scene_info.mask_objects()
            scene_info.consolidate_detections()
            scene_info.do_feature_matchings()
            scene_info.compute_6d_poses()

            end_time = time.time()

            # Process results
            for result in scene_info.detections_homographies:
                obj_id = result['object_idx']
                score = result['confidence']
                R = result['R']
                t = result['T']
                time_taken = end_time - start_time
                
                # Add row to CSV and save
                self.csv_writer.add_row(scene_id, image_id, obj_id, score, R, t, time_taken)
            
            # Clear results to free memory
            if hasattr(scene_info, 'detections_homographies'):
                scene_info.detections_homographies.clear()
                
        except Exception as e:
            print(f"Error processing group {group_key}: {e}")
        finally:
            # Cleanup scene_info and all its attributes
            if scene_info is not None:
                # Try to explicitly delete large attributes if they exist
                attrs_to_clear = ['detections_homographies', 'features', 'matches', 'images', 'masks']
                for attr in attrs_to_clear:
                    if hasattr(scene_info, attr):
                        delattr(scene_info, attr)
                
                del scene_info
            
            # Clear camera_rows data
            camera_rows.clear()
            
            # Force garbage collection
            self.aggressive_cleanup()
    
    def iterate_by_scene_image_groups_streaming(self):
        """
        Streaming approach with proper grouping - sorts data first then processes.
        This ensures correct grouping while managing memory efficiently.
        """
        temp_file = 'temp_sorted.csv'
        
        try:
            # Step 1: Sort the CSV file by (scene_id, image_id)
            print("Sorting CSV data by (scene_id, image_id)...")
            self.sort_csv_file(temp_file)
            
            # Step 2: Process the sorted file in streaming fashion
            print("Processing sorted data...")
            last_key = None
            current_group_rows = []
            group_count = 0
            
            with open(temp_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    scene_id = row['scene_id']
                    image_id = row['image_id']
                    current_key = (scene_id, image_id)
                    
                    # If we've moved to a new group, process the previous one
                    if last_key is not None and current_key != last_key:
                        self.process_single_group(last_key, current_group_rows, group_count)
                        group_count += 1
                        current_group_rows = []
                        self.aggressive_cleanup()
                    
                    current_group_rows.append(row)
                    last_key = current_key
                
                # Process the final group
                if current_group_rows:
                    self.process_single_group(last_key, current_group_rows, group_count)
                    
        except Exception as e:
            print(f"Error in streaming processing: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if hasattr(self.csv_writer, 'close'):
                self.csv_writer.close()
            
            self.aggressive_cleanup()
    
    def sort_csv_file(self, output_file):
        """
        Sort CSV file by (scene_id, image_id) using external sort for memory efficiency.
        """
        chunk_size = 10000  # Adjust based on available memory
        temp_files = []
        
        try:
            # Step 1: Split into sorted chunks
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                fieldnames = reader.fieldnames
                
                chunk = []
                chunk_num = 0
                
                for row in reader:
                    chunk.append(row)
                    
                    if len(chunk) >= chunk_size:
                        # Sort this chunk and write to temp file
                        chunk.sort(key=lambda x: (x['scene_id'], x['image_id']))
                        temp_file = f'temp_chunk_{chunk_num}.csv'
                        temp_files.append(temp_file)
                        
                        with open(temp_file, 'w', newline='', encoding='utf-8') as tf:
                            writer = csv.DictWriter(tf, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(chunk)
                        
                        chunk = []
                        chunk_num += 1
                        gc.collect()
                
                # Handle remaining chunk
                if chunk:
                    chunk.sort(key=lambda x: (x['scene_id'], x['image_id']))
                    temp_file = f'temp_chunk_{chunk_num}.csv'
                    temp_files.append(temp_file)
                    
                    with open(temp_file, 'w', newline='', encoding='utf-8') as tf:
                        writer = csv.DictWriter(tf, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(chunk)
            
            # Step 2: Merge sorted chunks
            self.merge_sorted_files(temp_files, output_file, fieldnames)
            
        finally:
            # Clean up temporary chunk files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def merge_sorted_files(self, temp_files, output_file, fieldnames):
        """
        Merge multiple sorted CSV files into one sorted file.
        """
        file_readers = []
        
        try:
            # Open all temp files
            files = [open(tf, 'r', newline='', encoding='utf-8') for tf in temp_files]
            readers = [csv.DictReader(f) for f in files]
            
            # Create heap with first row from each file
            heap = []
            for i, reader in enumerate(readers):
                try:
                    row = next(reader)
                    heapq.heappush(heap, ((row['scene_id'], row['image_id']), i, row))
                except StopIteration:
                    pass
            
            # Write merged output
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                while heap:
                    key, file_idx, row = heapq.heappop(heap)
                    writer.writerow(row)
                    
                    # Get next row from the same file
                    try:
                        next_row = next(readers[file_idx])
                        heapq.heappush(heap, ((next_row['scene_id'], next_row['image_id']), file_idx, next_row))
                    except StopIteration:
                        pass
        
        finally:
            # Close all files
            for f in files:
                f.close()
    
    def iterate_by_scene_image_groups(self):
        """
        Original method with memory optimizations.
        Use this if you need all groups in memory at once.
        """
        grouped_data = defaultdict(list)
        
        try:
            # First pass: Read and group the data
            with open(self.csv_file_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    scene_id = row['scene_id']
                    image_id = row['image_id']
                    key = (scene_id, image_id)
                    grouped_data[key].append(row)
                    
                    # Check memory periodically
                    if len(grouped_data) % 1000 == 0:
                        if self.check_memory_limit():
                            print("Switching to streaming mode due to memory constraints")
                            del grouped_data
                            file.seek(0)  # Reset file pointer
                            return self.iterate_by_scene_image_groups_streaming()
            
            print(f"Total groups to process: {len(grouped_data)}")
            print("Iterating through scene_id, image_id combinations:")
            print("=" * 60)
            
            # Second pass: Iterate through each group
            for group_num, (group_key, camera_rows) in enumerate(grouped_data.items()):
                self.process_single_group(group_key, camera_rows, group_num)
                
                # Remove processed data to free memory
                del grouped_data[group_key]
                
                # Periodic cleanup
                if (group_num + 1) % 10 == 0:
                    self.aggressive_cleanup()
                    
        except FileNotFoundError:
            print(f"Error: Could not find the file '{self.csv_file_path}'")
        except KeyError as e:
            print(f"Error: Column {e} not found in the CSV file")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
        finally:
            if hasattr(self.csv_writer, 'close'):
                self.csv_writer.close()
            
            # Final cleanup
            del grouped_data
            self.aggressive_cleanup()


if __name__ == "__main__":
    csv_file_path = "/home/rama/bpc_ws/bpc/utilities/test_dataset.csv"
    
    # Create processor with memory limit (adjust as needed)
    processor = SceneImageGroupProcessor(csv_file_path, max_memory_mb=4000, chunk_size=1000)
    
    # Choose processing method based on your data size and requirements:
    
    print("Choose processing method:")
    print("1. Original with optimizations (small to medium datasets)")
    print("2. Streaming with sorting (large datasets, ensures correct grouping)")
    print("3. Chunk-based processing (alternative for large datasets)")
    
    # Recommended approach for correct grouping with memory efficiency:
    print("\nUsing streaming approach with sorting for correct grouping...")
    processor.iterate_by_scene_image_groups_streaming()
    
    # Alternative approaches:
    # processor.iterate_by_scene_image_groups()  # Original with optimizations
    # processor.process_in_chunks()  # Chunk-based processing