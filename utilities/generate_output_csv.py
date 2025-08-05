import csv
import numpy as np

class CSVWriter:
    def __init__(self, filename='results.csv'):
        self.filename = filename
        
        # Create CSV file with headers
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
    
    def add_row(self, scene_id, im_id, obj_id, score, R, t, time):
        # Process score: if it's an array, take the average
        if isinstance(score, (list, np.ndarray)):
            processed_score = np.mean(score)
        else:
            processed_score = score
        
        # Process R: flatten and join with spaces
        if isinstance(R, (list, np.ndarray)):
            R_flat = np.array(R).flatten()
            processed_R = ' '.join(map(str, R_flat))
        else:
            processed_R = str(R)
        
        # Process t: join with spaces
        if isinstance(t, (list, np.ndarray)):
            processed_t = ' '.join(map(str, t))
        else:
            processed_t = str(t)
        
        # Append new row and save immediately
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([scene_id, im_id, obj_id, processed_score, processed_R, processed_t, time])

# Usage example:
if __name__ == "__main__":
    # Create the CSV writer
    csv_writer = CSVWriter('my_results.csv')
    
    # Example for loop
    for i in range(5):  # Replace with your actual loop
        # Your processing logic that generates these values:
        scene_id = f"scene_{i}"
        im_id = f"image_{i}"
        obj_id = f"object_{i}"
        
        # Example with array for score (will be averaged)
        score = [0.70628375, 0.4116815, 0.5196455, 0.8063665]
        
        # Example rotation matrix (3x3)
        R = [[0.9422484, -0.3299687, 0.05734513],
             [0.33486202, 0.92515296, -0.1787722],
             [0.00593624, 0.18765044, 0.9822179]]
        
        # Example translation vector
        t = [-555.1432, 946.1903, 223.90796]
        
        time = f"2024-01-{i+1:02d}"
        
        # Add row to CSV and save
        csv_writer.add_row(scene_id, im_id, obj_id, score, R, t, time)