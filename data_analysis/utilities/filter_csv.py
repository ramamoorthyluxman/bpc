import pandas as pd
import numpy as np
import argparse
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

def rotation_matrix_from_row(row):
    """Convert CSV row to 3x3 rotation matrix"""
    return np.array([
        [row['r11'], row['r12'], row['r13']],
        [row['r21'], row['r22'], row['r23']],
        [row['r31'], row['r32'], row['r33']]
    ])

def angular_distance(R1, R2):
    """Calculate angular distance between two rotation matrices"""
    # Ensure matrices are proper rotation matrices (handle numerical errors)
    R1 = np.array(R1, dtype=float)
    R2 = np.array(R2, dtype=float)
    
    # Calculate R1^T * R2
    R_rel = np.dot(R1.T, R2)
    
    # Calculate trace and clamp to valid range for arccos
    trace_val = np.trace(R_rel)
    trace_val = np.clip(trace_val, -1, 3)  # trace of rotation matrix is between -1 and 3
    
    # Calculate angle using rotation matrix trace formula
    # angle = arccos((trace(R_rel) - 1) / 2)
    cos_angle = (trace_val - 1) / 2
    cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range for arccos
    
    angle = np.arccos(cos_angle)
    return angle

def select_diverse_rotations(group_df, max_rows=5):
    """Select up to max_rows rotations with maximum diversity"""
    if len(group_df) <= max_rows:
        return group_df
    
    # Extract rotation matrices
    rotations = []
    for _, row in group_df.iterrows():
        rotations.append(rotation_matrix_from_row(row))
    
    n_rotations = len(rotations)
    
    # Calculate pairwise angular distances
    distances = np.zeros((n_rotations, n_rotations))
    for i in range(n_rotations):
        for j in range(i + 1, n_rotations):
            dist = angular_distance(rotations[i], rotations[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Greedy selection algorithm
    selected_indices = []
    
    # Start with the rotation that has maximum sum of distances to all others
    sum_distances = np.sum(distances, axis=1)
    first_idx = np.argmax(sum_distances)
    selected_indices.append(first_idx)
    
    # Iteratively add rotations that maximize minimum distance to selected ones
    for _ in range(min(max_rows - 1, n_rotations - 1)):
        remaining_indices = [i for i in range(n_rotations) if i not in selected_indices]
        if not remaining_indices:
            break
            
        best_idx = None
        best_min_dist = -1
        
        for candidate_idx in remaining_indices:
            # Calculate minimum distance to already selected rotations
            min_dist_to_selected = min([distances[candidate_idx, sel_idx] 
                                      for sel_idx in selected_indices])
            
            if min_dist_to_selected > best_min_dist:
                best_min_dist = min_dist_to_selected
                best_idx = candidate_idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
    
    # Return selected rows
    return group_df.iloc[selected_indices].copy()

def filter_dataset(input_csv_path, output_csv_path="simplified_dataset.csv", max_rows_per_object_per_camera=5):
    """Filter dataset to keep max_rows_per_object_per_camera with most diverse rotations per object per camera"""
    
    print(f"Loading dataset from: {input_csv_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: File {input_csv_path} not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Unique objects: {df['object_id'].nunique()}")
    print(f"Unique cameras: {df['camera_type'].nunique()}")
    print(f"Unique object-camera combinations: {df.groupby(['object_id', 'camera_type']).ngroups}")
    
    # Check if required columns exist
    rotation_cols = ['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
    required_cols = rotation_cols + ['object_id', 'camera_type']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Group by object_id and camera_type and apply filtering
    filtered_groups = []
    
    for (object_id, camera_type), group in df.groupby(['object_id', 'camera_type']):
        print(f"Processing object {object_id}, camera {camera_type}: {len(group)} rows -> ", end="")
        
        if len(group) <= max_rows_per_object_per_camera:
            filtered_group = group
            print(f"{len(filtered_group)} rows (keeping all)")
        else:
            try:
                filtered_group = select_diverse_rotations(group, max_rows_per_object_per_camera)
                print(f"{len(filtered_group)} rows (filtered for diversity)")
            except Exception as e:
                print(f"Error processing object {object_id}, camera {camera_type}: {e}")
                # Fallback: just take first max_rows_per_object_per_camera rows
                filtered_group = group.head(max_rows_per_object_per_camera)
                print(f"{len(filtered_group)} rows (fallback: first {max_rows_per_object_per_camera})")
        
        filtered_groups.append(filtered_group)
    
    # Combine all filtered groups
    result_df = pd.concat(filtered_groups, ignore_index=True)
    
    print(f"\nFiltered dataset shape: {result_df.shape}")
    print(f"Reduction: {len(df)} -> {len(result_df)} rows ({len(result_df)/len(df)*100:.1f}%)")
    
    # Print summary statistics
    summary = result_df.groupby(['object_id', 'camera_type']).size().reset_index(name='count')
    print(f"\nSummary per object-camera combination:")
    print(f"Min images per object-camera: {summary['count'].min()}")
    print(f"Max images per object-camera: {summary['count'].max()}")
    print(f"Mean images per object-camera: {summary['count'].mean():.1f}")
    
    # Save the result
    try:
        result_df.to_csv(output_csv_path, index=False)
        print(f"Saved filtered dataset to: {output_csv_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Filter dataset to keep max 5 rows per object per camera with most diverse rotations")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument("--output", "-o", default="simplified_dataset.csv", 
                       help="Output CSV file path (default: simplified_dataset.csv)")
    parser.add_argument("--max-rows", "-m", type=int, default=5,
                       help="Maximum rows per object per camera (default: 5)")
    
    args = parser.parse_args()
    
    filter_dataset(args.input_csv, args.output, args.max_rows)

if __name__ == "__main__":
    main()

    # python script.py input_dataset.csv --output filtered_dataset.csv --max-rows 5