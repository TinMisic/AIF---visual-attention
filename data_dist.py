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

def generate_histogram(coords):
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
    
    return image

def plot_histograms(original_image):
    # Create rotated versions
    rotated_90 = np.rot90(original_image)
    rotated_180 = np.rot90(rotated_90)
    rotated_270 = np.rot90(rotated_180)
    summed_image = original_image + rotated_90 + rotated_180 + rotated_270
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    images = [original_image, rotated_90, rotated_180, rotated_270, summed_image]
    titles = ['Original', 'Rotated 90°', 'Rotated 180°', 'Rotated 270°', 'Summed']
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.T, origin='lower', cmap='viridis', extent=[-10, 42, -10, 42])
        ax.set_title(title)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
    
    plt.colorbar(axes[0].imshow(original_image.T, origin='lower', cmap='viridis', extent=[-10, 42, -10, 42]), ax=axes, orientation='horizontal', fraction=0.02, pad=0.04, label='Frequency')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process image coordinates from CSV and plot histogram.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file containing coordinates')
    parser.add_argument('i', type=int, help='Column index for the x coordinate')
    args = parser.parse_args()
    
    data = read_csv(args.file_path)
    coords = process_coordinates(data, args.i)
    original_image = generate_histogram(coords)
    plot_histograms(original_image)

if __name__ == '__main__':
    main()

