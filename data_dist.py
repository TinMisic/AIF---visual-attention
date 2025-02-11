import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse

def read_csv(file_path):
    # Read the CSV file into a NumPy array using fromtxt
    data = np.loadtxt(file_path, delimiter=',')
    return data

def process_coordinates(data, i):
    # Extract and round the coordinates based on the given index i
    rounded_coords = np.round(data[:, i:i+2]).astype(int)
    return list(map(tuple, rounded_coords))

def plot_histogram(coords):
    # Count occurrences of each coordinate
    coord_counts = Counter(coords)
    
    # Define the image size and limits
    image_size = (53, 53)
    x_min, x_max = -10, 42
    y_min, y_max = -10, 42
    
    # Create an empty image matrix
    image = np.zeros(image_size, dtype=int)
    
    # Populate the image with frequency counts
    for (x, y), count in coord_counts.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            image[x - x_min, y - y_min] = count
    
    plt.figure(figsize=(10, 6))
    plt.imshow(image.T, origin='lower', cmap='viridis', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label='Frequency')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Histogram of Image Coordinates')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process image coordinates from CSV and plot histogram.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing coordinates')
    parser.add_argument('i', type=int, help='Column index for the x coordinate')
    args = parser.parse_args()
    
    data = read_csv(args.file_path)
    coords = process_coordinates(data, args.i)
    plot_histogram(coords)

if __name__ == '__main__':
    main()
