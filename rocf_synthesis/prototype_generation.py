import os
import csv
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- CONFIGURATION ---

# File and Directory Settings
INPUT_CSV_PATH = 'sample_data.csv' # Path to the source data
OUTPUT_FOLDER = "R30_generated_images" # Folder to save generated images
OUTPUT_CSV_FILE = "R30_image_scores.csv" # CSV to log image names and scores
TARGET_COLUMN_PREFIX = 'R30' # Prefix for the columns to be read from the input CSV

# Generation Parameters
NUM_DRAWINGS = 1500 # Total number of images to generate

# Distortion Parameters (controls the "human-like" errors)
# A higher score results in a smaller shift (less error)
MIN_SHIFT = 2   # Max error for a perfect score (slight hand tremor)
MID_SHIFT = 4   # Error for a score of 1.0
MAX_SHIFT = 12  # Max error for a score of 0.0 (significant distortion)


# --- DATA: FIGURE COMPONENTS AND CONNECTIONS ---

# This dictionary defines the geometric structure of the 18 components of the Rey Figure.
# Each component has a set of points and connections that form its shape.
ANNOTATIONS_CONNECTIONS = {
    1: {"label": "Outer cross", "points": [(31, 82), (112, 82), (70, 46), (70, 230), (70, 395), (118, 230)], "connections": [((31, 82), (112, 82)), ((70, 46), (70, 82)), ((70, 82), (70, 230)), ((70, 230), (70, 395)), ((118, 230), (70, 230))]},
    2: {"label": "Large rectangle", "points": [(118, 186), (118, 230), (118, 275), (118, 297), (118, 408), (118, 519), (118, 630), (285, 630), (449, 630), (780, 630), (780, 408), (532, 186)], "connections": [((118, 186), (118, 630)), ((118, 630), (780, 630)), ((780, 630), (780, 186)), ((780, 186), (118, 186))]},
    3: {"label": "Saint Andrew's Cross", "points": [(449, 408), (118, 186), (780, 186), (780, 630), (118, 630)], "connections": [((118, 186), (780, 630)), ((780, 186), (118, 630))]},
    4: {"label": "Horizontal median", "points": [(118, 408), (449, 408), (780, 408)], "connections": [((118, 408), (780, 408))]},
    5: {"label": "Vertical median", "points": [(449, 186), (449, 408), (449, 630)], "connections": [((449, 186), (449, 630))]},
    6: {"label": "Small inner rectangle", "points": [(118, 519), (284, 519), (284, 408), (284, 297), (118, 297), (118, 408)], "connections": [((118, 297), (284, 297)), ((284, 297), (284, 519)), ((284, 519), (118, 519)), ((118, 519), (118, 297)), ((118, 297), (284, 519)), ((118, 519), (284, 297))]},
    7: {"label": "Small segment", "points": [(118, 286), (267, 286)], "connections": [((118, 286), (267, 286))]},
    8: {"label": "Four parallel lines", "points": [(449, 230), (449, 275), (449, 319), (449, 364), (184.2, 230.4), (250.4, 274.8), (316.6, 319.2), (382.8, 363.6)], "connections": [((449, 230), (184.2, 230.4)), ((449, 275), (250.4, 274.8)), ((449, 319), (316.6, 319.2)), ((449, 364), (382.8, 363.6))]},
    9: {"label": "Right triangle", "points": [(449, 186), (532, 186), (780, 186), (449, 15)], "connections": [((449, 186), (780, 186)), ((780, 186), (449, 15)), ((449, 15), (449, 186))]},
    10: {"label": "Small perpendicular line", "points": [(532, 186), (532, 351)], "connections": [((532, 186), (532, 351))]},
    11: {"label": "Circle", "points": [(620, 334), (623, 314), (634, 298), (650, 287), (670, 284), (689, 287), (705, 298), (716, 314), (720, 334), (716, 353), (705, 369), (689, 380), (670, 384), (650, 380), (634, 369), (623, 353)], "connections": [((620, 334), (623, 314)), ((623, 314), (634, 298)), ((634, 298), (650, 287)), ((650, 287), (670, 284)), ((670, 284), (689, 287)), ((689, 287), (705, 298)), ((705, 298), (716, 314)), ((716, 314), (720, 334)), ((720, 334), (716, 353)), ((716, 353), (705, 369)), ((705, 369), (689, 380)), ((689, 380), (670, 384)), ((670, 384), (650, 380)), ((650, 380), (634, 369)), ((634, 369), (623, 353)), ((623, 353), (620, 334))]},
    12: {"label": "Five parallel lines (diagonal)", "points": [(566, 448), (530, 502), (599, 470), (563, 524), (633, 492), (597, 546), (666, 514), (630, 568), (699, 536), (663, 590)], "connections": [((566, 448), (530, 502)), ((599, 470), (563, 524)), ((633, 492), (597, 546)), ((666, 514), (630, 568)), ((699, 536), (663, 590))]},
    13: {"label": "Isosceles triangle", "points": [(780, 630), (780, 408), (780, 186), (1002, 408)], "connections": [((780, 186), (1002, 408)), ((1002, 408), (780, 630))]},
    14: {"label": "Small diamond", "points": [(1002, 408), (972, 484), (1032, 484), (1002, 560)], "connections": [((1002, 408), (972, 484)), ((972, 484), (1002, 560)), ((1002, 560), (1032, 484)), ((1032, 484), (1002, 408))]},
    15: {"label": "Segment inside triangle", "points": [(891, 408), (891, 298), (891, 518)], "connections": [((891, 298), (891, 518))]},
    16: {"label": "Extension of median", "points": [(780, 408), (891, 408), (1002, 408)], "connections": [((780, 408), (1002, 408))]},
    17: {"label": "Lower cross", "points": [(285, 714), (714, 714), (640, 664), (640, 764), (449, 630), (449, 714)], "connections": [((285, 714), (714, 714)), ((640, 664), (640, 764)), ((449, 630), (449, 714))]},
    18: {"label": "Square at lower left", "points": [(118, 630), (118, 797), (285, 797), (285, 630)], "connections": [((118, 630), (118, 797)), ((118, 797), (285, 797)), ((285, 797), (285, 630)), ((285, 630), (118, 630)), ((118, 630), (285, 797))]},
}


# --- HELPER FUNCTIONS ---

def randomize_points(points, max_shift_value):
    """Applies a random directional shift to a list of points."""
    if max_shift_value <= 0:
        return points
    
    randomized_points = []
    for x, y in points:
        shift_x = random.uniform(max_shift_value - 2, max_shift_value) * random.choice([-1, 1])
        shift_y = random.uniform(max_shift_value - 2, max_shift_value) * random.choice([-1, 1])
        randomized_points.append((x + shift_x, y + shift_y))
    return randomized_points

def calculate_max_shift(score):
    """
    Calculates the maximum point shift based on a score from 0 to 2.
    A lower score results in a larger shift (more distortion).
    """
    if score >= 1:
        # High scores (1 to 2): scale between MIN_SHIFT and MID_SHIFT
        return MIN_SHIFT + (MID_SHIFT - MIN_SHIFT) * (2 - score)
    else:
        # Low scores (0 to 1): scale between MID_SHIFT and MAX_SHIFT
        return MID_SHIFT + (MAX_SHIFT - MID_SHIFT) * (1 - score)


# --- MAIN EXECUTION ---

def main():
    """Main function to generate drawings and log scores."""
    
    # Load the source data
    try:
        data = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: The input file was not found at '{INPUT_CSV_PATH}'")
        return

    # Prepare output directory and CSV file
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    target_columns = [f'{TARGET_COLUMN_PREFIX}/{i}' for i in range(1, 19)]
    
    # Initialize the CSV file with headers
    with open(OUTPUT_CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ['ImageName'] + [f'Score_{i}' for i in range(1, 19)] + ['TotalScore']
        writer.writerow(headers)

    # --- Generation Loop ---
    for i in range(NUM_DRAWINGS):
        # Use a unique seed for each drawing to ensure variability
        seed = int(time.time()) + i
        random.seed(seed)
        np.random.seed(seed)

        # 1. Sample scores for this drawing from the real data distribution
        sampled_scores_map = {}
        score_bins = np.arange(-0.25, 2.75, 0.5)
        possible_scores = np.arange(0, 2.25, 0.5)

        for col in target_columns:
            if col in data:
                col_data = data[col].dropna()
                counts, _ = np.histogram(col_data, bins=score_bins, density=True)
                probabilities = counts / np.sum(counts) if np.sum(counts) > 0 else None
                
                if probabilities is not None:
                    sampled_value = np.random.choice(possible_scores, p=probabilities)
                    sampled_scores_map[col] = sampled_value
                else:
                    sampled_scores_map[col] = 0 # Default if no data
            else:
                print(f"Warning: Column '{col}' not found in the input CSV.")
                sampled_scores_map[col] = 0

        # 2. Build the graph based on sampled scores
        G = nx.Graph()
        individual_scores = []
        
        for shape_id in range(1, 19):
            col_name = f'{TARGET_COLUMN_PREFIX}/{shape_id}'
            sampled_score = sampled_scores_map.get(col_name, 0)
            individual_scores.append(sampled_score)
            
            # Decide whether to draw the shape. A score of 2 is always drawn, 0 is never drawn.
            if random.random() < (sampled_score / 2.0):
                shape_data = ANNOTATIONS_CONNECTIONS[shape_id]
                
                # Apply distortion based on score
                f_shift = calculate_max_shift(sampled_score)
                original_points = shape_data["points"]
                randomized_points = randomize_points(original_points, max_shift_value=f_shift)
                
                point_mapping = dict(zip(original_points, randomized_points))
                
                # Add nodes and edges to the graph
                for point in randomized_points:
                    G.add_node(point, pos=point)
                
                for p1, p2 in shape_data["connections"]:
                    if p1 in point_mapping and p2 in point_mapping:
                        G.add_edge(point_mapping[p1], point_mapping[p2])

        # 3. Plot and save the graph as an image
        plt.figure(figsize=(12, 10))
        pos = nx.get_node_attributes(G, 'pos')
        
        line_thickness = random.uniform(1.5, 4.0)
        opacity = random.uniform(0.7, 1.0)
        
        nx.draw(
            G, pos, 
            with_labels=False, 
            node_size=0, 
            edge_color='black', 
            width=line_thickness, 
            alpha=opacity
        )

        plt.gca().invert_yaxis() # Match image coordinate system
        plt.axis('off') # Remove axes for a cleaner image
        plt.tight_layout(pad=0)

        # Save the figure
        graph_filename = f"drawing_{i+1}.png"
        graph_filepath = os.path.join(OUTPUT_FOLDER, graph_filename)
        plt.savefig(graph_filepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 4. Log the results to the CSV
        total_score = sum(individual_scores)
        with open(OUTPUT_CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([graph_filename] + individual_scores + [total_score])

        print(f"Generated {graph_filename} ({i+1}/{NUM_DRAWINGS}) with total score: {total_score:.2f}")

    print("\nGeneration complete.")
    print(f"Images saved in: '{OUTPUT_FOLDER}'")
    print(f"Scores logged in: '{OUTPUT_CSV_FILE}'")

if __name__ == "__main__":
    main()