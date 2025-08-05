import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class SimplePoseComparisonAnalyzer:
    def __init__(self, output_dir='pose_analysis_results'):
        self.your_results = None
        self.reference_results = None
        self.reference_results_filtered = None
        self.detection_stats = None
        self.pose_errors = None
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, your_csv_path, reference_csv_path):
        """Load and preprocess the CSV files"""
        print("Loading CSV files...")
        
        # Load data
        self.your_results = pd.read_csv(your_csv_path)
        self.reference_results = pd.read_csv(reference_csv_path)
        
        # Parse rotation matrices and translation vectors
        self.your_results = self._parse_pose_data(self.your_results)
        self.reference_results = self._parse_pose_data(self.reference_results)
        
        print(f"Your results: {len(self.your_results)} detections")
        print(f"Reference results: {len(self.reference_results)} detections")
        
    def _parse_pose_data(self, df):
        """Parse R and t columns into proper arrays"""
        df = df.copy()
        
        # Parse rotation matrix R
        if isinstance(df['R'].iloc[0], str):
            df['R_matrix'] = df['R'].apply(lambda x: np.array(list(map(float, x.split()))).reshape(3, 3))
        else:
            df['R_matrix'] = df['R'].apply(lambda x: np.array(x).reshape(3, 3))
            
        # Parse translation vector t
        if isinstance(df['t'].iloc[0], str):
            df['t_vector'] = df['t'].apply(lambda x: np.array(list(map(float, x.split()))))
        else:
            df['t_vector'] = df['t'].apply(lambda x: np.array(x))
            
        return df
    
    def filter_reference_duplicates(self, score_threshold_percentile=25):
        """Filter reference results to keep only high-scoring detections"""
        print("Filtering reference duplicates...")
        
        # Calculate score threshold
        score_threshold = np.percentile(self.reference_results['score'], score_threshold_percentile)
        
        # Filter low scores
        filtered_ref = self.reference_results[self.reference_results['score'] > score_threshold].copy()
        
        # For each scene+image+object combination, keep only the highest scoring detection
        filtered_ref = (filtered_ref.groupby(['scene_id', 'im_id', 'obj_id'], group_keys=False)
                       .apply(lambda x: x.loc[x['score'].idxmax()]))
        
        self.reference_results_filtered = filtered_ref.reset_index(drop=True)
        print(f"Filtered reference results: {len(self.reference_results_filtered)} detections")
        
        return self.reference_results_filtered
    
    def compare_scene_image_stats(self):
        """Compare object detection statistics per scene+image"""
        print("Analyzing detection statistics...")
        
        # Get all unique scene+image combinations
        all_scenes = set(self.your_results[['scene_id', 'im_id']].apply(tuple, axis=1)) | \
                    set(self.reference_results_filtered[['scene_id', 'im_id']].apply(tuple, axis=1))
        
        comparison_data = []
        
        for scene_id, im_id in all_scenes:
            your_detections = self.your_results[
                (self.your_results['scene_id'] == scene_id) & 
                (self.your_results['im_id'] == im_id)
            ]
            ref_detections = self.reference_results_filtered[
                (self.reference_results_filtered['scene_id'] == scene_id) & 
                (self.reference_results_filtered['im_id'] == im_id)
            ]
            
            # Count objects
            your_obj_counts = your_detections['obj_id'].value_counts().to_dict()
            ref_obj_counts = ref_detections['obj_id'].value_counts().to_dict()
            
            comparison_data.append({
                'scene_id': scene_id,
                'im_id': im_id,
                'your_total_objects': len(your_detections),
                'ref_total_objects': len(ref_detections),
                'common_objects': len(set(your_obj_counts.keys()) & set(ref_obj_counts.keys())),
                'your_only_objects': len(set(your_obj_counts.keys()) - set(ref_obj_counts.keys())),
                'ref_only_objects': len(set(ref_obj_counts.keys()) - set(your_obj_counts.keys())),
            })
        
        self.detection_stats = pd.DataFrame(comparison_data)
        return self.detection_stats
    
    def calculate_pose_errors(self, position_clustering_eps=0.5):  # Increased threshold
        """Calculate pose errors for matching detections"""
        print("Calculating pose errors...")
        
        pose_errors = []
        
        for _, stats in self.detection_stats.iterrows():
            scene_id, im_id = stats['scene_id'], stats['im_id']
            
            your_detections = self.your_results[
                (self.your_results['scene_id'] == scene_id) & 
                (self.your_results['im_id'] == im_id)
            ]
            ref_detections = self.reference_results_filtered[
                (self.reference_results_filtered['scene_id'] == scene_id) & 
                (self.reference_results_filtered['im_id'] == im_id)
            ]
            
            # For each object type that appears in both
            common_objects = set(your_detections['obj_id']) & set(ref_detections['obj_id'])
            
            for obj_id in common_objects:
                your_obj = your_detections[your_detections['obj_id'] == obj_id]
                ref_obj = ref_detections[ref_detections['obj_id'] == obj_id]
                
                # Match poses based on 3D position proximity
                matches = self._match_poses_by_position(your_obj, ref_obj, position_clustering_eps)
                
                for your_idx, ref_idx in matches:
                    your_pose = your_obj.iloc[your_idx]
                    ref_pose = ref_obj.iloc[ref_idx]
                    
                    # Calculate translation error
                    translation_error = np.linalg.norm(your_pose['t_vector'] - ref_pose['t_vector'])
                    
                    # Calculate rotation error (angular distance)
                    rotation_error = self._rotation_error(your_pose['R_matrix'], ref_pose['R_matrix'])
                    
                    pose_errors.append({
                        'scene_id': scene_id,
                        'im_id': im_id,
                        'obj_id': obj_id,
                        'translation_error': translation_error,
                        'rotation_error': rotation_error,
                        'your_score': your_pose.get('score', 1.0),
                        'ref_score': ref_pose['score']
                    })
        
        self.pose_errors = pd.DataFrame(pose_errors)
        print(f"Found {len(self.pose_errors)} matching poses for error calculation")
        return self.pose_errors
    
    def _match_poses_by_position(self, your_poses, ref_poses, eps=0.5):
        """Match poses based on 3D position proximity using Hungarian algorithm"""
        if len(your_poses) == 0 or len(ref_poses) == 0:
            return []
        
        # Extract 3D positions
        your_positions = np.array([pose['t_vector'] for _, pose in your_poses.iterrows()])
        ref_positions = np.array([pose['t_vector'] for _, pose in ref_poses.iterrows()])
        
        # Calculate distance matrix
        distance_matrix = cdist(your_positions, ref_positions)
        
        # Use Hungarian algorithm for optimal assignment
        your_indices, ref_indices = linear_sum_assignment(distance_matrix)
        
        # Filter matches that are too far apart
        matches = []
        for i, j in zip(your_indices, ref_indices):
            if distance_matrix[i, j] <= eps:
                matches.append((i, j))
        
        return matches
    
    def _rotation_error(self, R1, R2):
        """Calculate rotation error in degrees"""
        R_diff = R1.T @ R2
        trace = np.trace(R_diff)
        # Clamp to avoid numerical issues
        trace = np.clip(trace, -1, 3)
        angle = np.arccos((trace - 1) / 2)
        return np.degrees(angle)
    
    def save_detection_comparison_plots(self):
        """Create and save detection comparison plots"""
        print("Creating detection comparison plots...")
        
        # Calculate precision and recall
        self.detection_stats['precision'] = self.detection_stats['common_objects'] / (
            self.detection_stats['your_total_objects'] + 1e-8)
        self.detection_stats['recall'] = self.detection_stats['common_objects'] / (
            self.detection_stats['ref_total_objects'] + 1e-8)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Total objects per scene
        axes[0,0].scatter(self.detection_stats['ref_total_objects'], 
                         self.detection_stats['your_total_objects'], alpha=0.6)
        max_objects = max(self.detection_stats['ref_total_objects'].max(), 
                         self.detection_stats['your_total_objects'].max())
        axes[0,0].plot([0, max_objects], [0, max_objects], 'r--')
        axes[0,0].set_xlabel('Reference Total Objects')
        axes[0,0].set_ylabel('Your Total Objects')
        axes[0,0].set_title('Total Objects per Scene+Image')
        
        # Common vs missed objects
        axes[0,1].hist(self.detection_stats['common_objects'], bins=20, alpha=0.7, label='Common')
        axes[0,1].hist(self.detection_stats['your_only_objects'], bins=20, alpha=0.7, label='Your Only')
        axes[0,1].hist(self.detection_stats['ref_only_objects'], bins=20, alpha=0.7, label='Ref Only')
        axes[0,1].set_xlabel('Number of Objects')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Object Detection Overlap')
        axes[0,1].legend()
        
        # Precision and recall
        axes[1,0].hist(self.detection_stats['precision'], bins=20, alpha=0.7)
        axes[1,0].set_xlabel('Precision')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Detection Precision Distribution')
        
        axes[1,1].hist(self.detection_stats['recall'], bins=20, alpha=0.7)
        axes[1,1].set_xlabel('Recall')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Detection Recall Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'detection_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved detection comparison plot to {self.output_dir}/detection_comparison.png")
    
    def save_pose_error_plots(self):
        """Create and save pose error plots"""
        
        if self.pose_errors.empty:
            print("No pose errors to visualize. Check if there are matching detections.")
            return
        
        print("Creating pose error plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Translation error distribution
        axes[0,0].hist(self.pose_errors['translation_error'], bins=30, alpha=0.7)
        axes[0,0].set_xlabel('Translation Error (units)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Translation Error Distribution')
        
        # Rotation error distribution
        axes[0,1].hist(self.pose_errors['rotation_error'], bins=30, alpha=0.7)
        axes[0,1].set_xlabel('Rotation Error (degrees)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Rotation Error Distribution')
        
        # Translation vs rotation error correlation
        axes[1,0].scatter(self.pose_errors['translation_error'], 
                         self.pose_errors['rotation_error'], alpha=0.6)
        axes[1,0].set_xlabel('Translation Error')
        axes[1,0].set_ylabel('Rotation Error')
        axes[1,0].set_title('Translation vs Rotation Error')
        
        # Cumulative error distribution
        sorted_trans_errors = np.sort(self.pose_errors['translation_error'])
        cumulative_pct = np.arange(1, len(sorted_trans_errors) + 1) / len(sorted_trans_errors)
        axes[1,1].plot(sorted_trans_errors, cumulative_pct)
        axes[1,1].set_xlabel('Translation Error')
        axes[1,1].set_ylabel('Cumulative Percentage')
        axes[1,1].set_title('Cumulative Translation Error')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pose_errors.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved pose error plots to {self.output_dir}/pose_errors.png")
    
    def create_simple_3d_visualization(self):
        """Create simple 3D visualization - individual HTML files for each scene/image"""
        print("Creating 3D visualizations...")
        
        # Get all unique scene and image combinations
        all_combinations = set()
        for df in [self.your_results, self.reference_results_filtered]:
            combinations = df[['scene_id', 'im_id']].drop_duplicates()
            for _, row in combinations.iterrows():
                all_combinations.add((row['scene_id'], row['im_id']))
        
        all_combinations = sorted(list(all_combinations))
        
        if not all_combinations:
            print("No scene/image combinations found!")
            return
        
        # Create individual HTML files for each combination
        created_files = []
        
        for scene_id, im_id in all_combinations[:10]:  # Limit to first 10 to avoid too many files
            filename = f'3d_plot_scene_{scene_id}_image_{im_id}'
            self._create_static_3d_plot(scene_id, im_id, save_name=filename)
            created_files.append(f'{filename}.html')
        
        # Create a simple index HTML file listing all visualizations
        self._create_index_html(all_combinations[:10])
        
        print(f"Created {len(created_files)} 3D visualization files")
        print(f"Open 'index.html' in your browser to see all visualizations")
    
    def _create_index_html(self, combinations):
        """Create a simple index page linking to all visualizations"""
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>6D Pose Comparison - All Visualizations</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 40px; 
            background-color: #f5f5f5; 
        }
        h1 { 
            color: #333; 
            text-align: center; 
            margin-bottom: 30px; 
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); 
            gap: 20px; 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .card { 
            background: white; 
            border-radius: 8px; 
            padding: 20px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            text-align: center; 
        }
        .card h3 { 
            margin-top: 0; 
            color: #555; 
        }
        .card a { 
            display: inline-block; 
            background: #007bff; 
            color: white; 
            padding: 10px 20px; 
            text-decoration: none; 
            border-radius: 5px; 
            margin: 5px; 
        }
        .card a:hover { 
            background: #0056b3; 
        }
        .legend { 
            background: white; 
            border-radius: 8px; 
            padding: 20px; 
            margin-bottom: 30px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            max-width: 600px; 
            margin-left: auto; 
            margin-right: auto; 
        }
        .legend h3 { 
            margin-top: 0; 
            color: #333; 
        }
        .legend-item { 
            display: flex; 
            align-items: center; 
            margin: 10px 0; 
        }
        .legend-symbol { 
            width: 20px; 
            height: 20px; 
            border-radius: 50%; 
            margin-right: 10px; 
        }
        .blue-circle { 
            background-color: #1f77b4; 
        }
        .red-diamond { 
            background-color: #d62728; 
            transform: rotate(45deg); 
            border-radius: 0; 
        }
    </style>
</head>
<body>
    <h1>6D Pose Estimation Results Comparison</h1>
    
    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-item">
            <div class="legend-symbol blue-circle"></div>
            <span><strong>Blue Circles:</strong> Your pose detections</span>
        </div>
        <div class="legend-item">
            <div class="legend-symbol red-diamond"></div>
            <span><strong>Red Diamonds:</strong> Reference pose detections</span>
        </div>
        <p><strong>Numbers on markers:</strong> Object IDs</p>
        <p><strong>Axes range:</strong> 0 to 2.5 meters (X, Y, Z)</p>
    </div>
    
    <div class="grid">
"""
        
        # Add cards for each scene/image combination
        for scene_id, im_id in combinations:
            filename = f'3d_plot_scene_{scene_id}_image_{im_id}.html'
            
            # Count detections for this combination
            your_count = len(self.your_results[
                (self.your_results['scene_id'] == scene_id) & 
                (self.your_results['im_id'] == im_id)
            ])
            ref_count = len(self.reference_results_filtered[
                (self.reference_results_filtered['scene_id'] == scene_id) & 
                (self.reference_results_filtered['im_id'] == im_id)
            ])
            
            html_content += f"""
        <div class="card">
            <h3>Scene {scene_id}, Image {im_id}</h3>
            <p>Your detections: <strong>{your_count}</strong></p>
            <p>Reference detections: <strong>{ref_count}</strong></p>
            <a href="{filename}" target="_blank">View 3D Visualization</a>
        </div>
"""
        
        html_content += """
    </div>
    
    <div style="text-align: center; margin-top: 40px; color: #666;">
        <p>Click on any "View 3D Visualization" button to open the interactive 3D plot</p>
        <p>Each plot opens in a new tab/window</p>
    </div>
</body>
</html>
"""
        
        # Save index file
        index_path = os.path.join(self.output_dir, 'index.html')
        with open(index_path, 'w') as f:
            f.write(html_content)
        
        print(f"Created index page: {index_path}")
    
    def _create_static_3d_plot(self, scene_id, im_id, save_name):
        """Create a static 3D plot for given scene and image - show ALL poses without grouping"""
        
        your_detections = self.your_results[
            (self.your_results['scene_id'] == scene_id) & 
            (self.your_results['im_id'] == im_id)
        ]
        ref_detections = self.reference_results_filtered[
            (self.reference_results_filtered['scene_id'] == scene_id) & 
            (self.reference_results_filtered['im_id'] == im_id)
        ]
        
        fig = go.Figure()
        
        # Collect all positions to determine axis limits
        all_positions = []
        
        # Plot ALL your detections (ignore scores completely)
        if len(your_detections) > 0:
            your_positions = np.array([t for t in your_detections['t_vector']])
            your_obj_ids = your_detections['obj_id'].values
            all_positions.append(your_positions)
            
            fig.add_trace(go.Scatter3d(
                x=your_positions[:, 0],
                y=your_positions[:, 1], 
                z=your_positions[:, 2],
                mode='markers+text',
                marker=dict(size=12, color='blue', symbol='circle'),
                text=[f'{obj_id}' for obj_id in your_obj_ids],
                textposition='top center',
                name='Your Detections',
                textfont=dict(size=12, color='blue'),
                hovertemplate="<b>Your Detection</b><br>" +
                            "Object ID: %{text}<br>" +
                            "X: %{x:.1f}<br>" +
                            "Y: %{y:.1f}<br>" +
                            "Z: %{z:.1f}<extra></extra>"
            ))
        
        # Plot ALL reference detections (already filtered by score thresholds)
        if len(ref_detections) > 0:
            ref_positions = np.array([t for t in ref_detections['t_vector']])
            ref_obj_ids = ref_detections['obj_id'].values
            ref_scores = ref_detections['score'].values
            all_positions.append(ref_positions)
            
            fig.add_trace(go.Scatter3d(
                x=ref_positions[:, 0],
                y=ref_positions[:, 1],
                z=ref_positions[:, 2],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='diamond'),
                text=[f'{obj_id}' for obj_id in ref_obj_ids],
                textposition='bottom center',
                name='Reference Detections',
                textfont=dict(size=12, color='red'),
                hovertemplate="<b>Reference Detection</b><br>" +
                            "Object ID: %{text}<br>" +
                            "Score: %{customdata:.3f}<br>" +
                            "X: %{x:.1f}<br>" +
                            "Y: %{y:.1f}<br>" +
                            "Z: %{z:.1f}<extra></extra>",
                customdata=ref_scores
            ))
        
        # Determine axis limits from actual data
        if all_positions:
            all_coords = np.vstack(all_positions)
            x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
            y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
            z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
            
            # Add some padding (10% of range)
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            x_padding = max(x_range * 0.1, 100)  # At least 100 units padding
            y_padding = max(y_range * 0.1, 100)
            z_padding = max(z_range * 0.1, 100)
            
            x_limits = [x_min - x_padding, x_max + x_padding]
            y_limits = [y_min - y_padding, y_max + y_padding]
            z_limits = [z_min - z_padding, z_max + z_padding]
            
            # Detect if data is likely in mm (values > 1000) or m (values < 10)
            max_coord = max(abs(x_max), abs(y_max), abs(z_max))
            if max_coord > 1000:
                units = "mm"
                axis_title_suffix = " (mm)"
            elif max_coord < 10:
                units = "m"
                axis_title_suffix = " (m)"
            else:
                units = "units"
                axis_title_suffix = ""
            
            print(f"Data range: X: {x_min:.1f} to {x_max:.1f}, Y: {y_min:.1f} to {y_max:.1f}, Z: {z_min:.1f} to {z_max:.1f} {units}")
            
        else:
            # Fallback to default range if no data
            x_limits = y_limits = z_limits = [0, 2500]  # Default to mm range
            axis_title_suffix = " (mm)"
            print("No pose data found for this scene/image combination")
        
        # Update layout with dynamic axis limits
        fig.update_layout(
            title=f'3D Pose Visualization - Scene {scene_id}, Image {im_id}<br>' +
                  f'Your: {len(your_detections)} detections, Reference: {len(ref_detections)} detections',
            scene=dict(
                xaxis_title=f'X{axis_title_suffix}',
                yaxis_title=f'Y{axis_title_suffix}',
                zaxis_title=f'Z{axis_title_suffix}',
                xaxis=dict(range=x_limits),
                yaxis=dict(range=y_limits),
                zaxis=dict(range=z_limits),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900,
            height=700
        )
        
        # Save plot
        plot_path = os.path.join(self.output_dir, f'{save_name}.html')
        fig.write_html(plot_path)
        print(f"Created: {save_name}.html")
    
def run_simple_analysis(your_csv_path, reference_csv_path, output_dir='pose_analysis_results'):
    """Run the complete analysis pipeline - simple version"""
    
    analyzer = SimplePoseComparisonAnalyzer(output_dir=output_dir)
    
    # Load data
    analyzer.load_data(your_csv_path, reference_csv_path)
    
    # Filter reference duplicates
    analyzer.filter_reference_duplicates(score_threshold_percentile=45)
    
    # Compare detection statistics
    analyzer.compare_scene_image_stats()
    
    # Calculate pose errors (with relaxed threshold)
    analyzer.calculate_pose_errors(position_clustering_eps=0.5)  # 50cm threshold
    
    # Generate and save all visualizations
    analyzer.save_detection_comparison_plots()
    analyzer.save_pose_error_plots()
    analyzer.create_simple_3d_visualization()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print(f"\nAll results saved to '{output_dir}' directory!")
    print("Files created:")
    print("- detection_comparison.png")
    print("- pose_errors.png (if matches found)")
    print("- index.html (main page with links to all 3D plots)")
    print("- 3d_plot_scene_X_image_Y.html (individual 3D visualizations)")
    print("- summary_report.txt")
    print(f"\n🌐 Open '{output_dir}/index.html' in your browser to see all visualizations!")
    
    return analyzer

# Run the analysis
# analyzer = run_simple_analysis('results.csv', 'ref.csv')
    """Run the complete analysis pipeline - simple version"""
    
    analyzer = SimplePoseComparisonAnalyzer(output_dir=output_dir)
    
    # Load data
    analyzer.load_data(your_csv_path, reference_csv_path)
    
    # Filter reference duplicates
    analyzer.filter_reference_duplicates(score_threshold_percentile=25)
    
    # Compare detection statistics
    analyzer.compare_scene_image_stats()
    
    # Calculate pose errors (with relaxed threshold)
    analyzer.calculate_pose_errors(position_clustering_eps=0.5)  # 50cm threshold
    
    # Generate and save all visualizations
    analyzer.save_detection_comparison_plots()
    analyzer.save_pose_error_plots()
    analyzer.create_simple_3d_visualization()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print(f"\nAll results saved to '{output_dir}' directory!")
    print("Files created:")
    print("- detection_comparison.png")
    print("- pose_errors.png (if matches found)")
    print("- 3d_plot_scene_X_image_Y.html (first 5 combinations)")
    print("- 3d_visualization.html (interactive template)")
    print("- summary_report.txt")
    
    return analyzer

# Run the analysis
# analyzer = run_simple_analysis('results.csv', 'ref.csv')
analyzer = run_simple_analysis(f'results.csv', f'ref.csv')