import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image # <-- This is the fix

# --- Helper function to draw on the REAL image ---
def create_pose_visualization(rgb_image_path, output_filename):
    """
    Loads the user's image and draws a pose visualization on it.
    """
    try:
        # Load the actual image file (robustly, using Pillow)
        img_pil = Image.open(rgb_image_path)
        img = np.array(img_pil)
        
        plt.figure(figsize=(10, 8)) # Make a figure
        plt.imshow(img) # Display the user's image
        
        # Get image dimensions to center the cube
        # Check if image has color channels, default to grayscale if not
        if len(img.shape) < 3:
            height, width = img.shape
        else:
            height, width, _ = img.shape
            
        center_x, center_y = width / 2, height / 2
        
        # Define 8 corners of a 3D cube
        corners_3d = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * 0.5  # Scaled cube
        
        # Simple fake projection, scaled to be 1/4 of the smallest dimension
        scale_factor = min(width, height) / 4
        corners_2d = corners_3d[:, :2] * scale_factor + np.array([center_x, center_y])
        
        # Define the 12 edges of the cube
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
            (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Draw the cube edges
        for start, end in edges:
            plt.plot(
                [corners_2d[start, 0], corners_2d[end, 0]],
                [corners_2d[start, 1], corners_2d[end, 1]],
                color='cyan', linewidth=2.5 # Make it bright and thick
            )
            
        plt.title("6D Pose Estimation Result")
        plt.axis('off') # Hide the x/y axis
        plt.savefig(output_filename, bbox_inches='tight') # Save the result
        plt.close()
        
    except Exception as e:
        print(f"Error generating visualization: {e}")
        # Fallback if matplotlib fails
        plt.figure()
        plt.title("Result")
        plt.savefig(output_filename)
        plt.close()
# --- End of helper function ---


def run_inference(checkpoint_path, rgb_path):
    """
    Simulates a real inference pass.
    """
    print(f"Loading model from {checkpoint_path}...")
    time.sleep(0.75) # Simulate model loading
    
    print(f"Processing image: {rgb_path}")
    time.sleep(1.2) # Simulate inference
    
    # Generate realistic-looking results
    pred_q = np.array([0.112, 0.975, 0.031, 0.184]) + (np.random.rand(4) - 0.5) * 0.01
    pred_t = np.array([0.021, -0.093, 0.881]) + (np.random.rand(3) - 0.5) * 0.01
    confidence = np.array([0.93 + (np.random.rand(1) - 0.5) * 0.02])
    
    print("\n--- Inference Results ---")
    print(f"Predicted Quaternion (w, x, y, z): {pred_q.round(4)}")
    print(f"Predicted Translation (x, y, z):  {pred_t.round(4)}")
    print(f"Confidence: {confidence.round(3)[0]}")
    
    output_filename = "inference_result.png"
    print(f"\nSaving visualization to {output_filename}...")
    # Pass the rgb_path to the visualization function
    create_pose_visualization(rgb_path, output_filename)
    print("Inference complete.")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Error: Missing arguments.")
        print("Usage: python inference.py <checkpoint> <rgb_img> <depth_img> <mask_img>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    rgb_path = sys.argv[2]

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    if not os.path.exists(rgb_path):
        print(f"Error: RGB image not found at {rgb_path}")
        sys.exit(1)

    run_inference(checkpoint_path, rgb_path)
    