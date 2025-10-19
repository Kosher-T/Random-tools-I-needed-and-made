# --- Frame Interpolation Generator using Dense Optical Flow (POC) ---

# GOAL: Take two input images, calculate the perceived motion (Dense Optical Flow), 
# and generate a plausible intermediate frame by warping and blending.

import cv2
import numpy as np
import sys
import os

# --- Constants ---
OUTPUT_FILENAME = "generated_intermediate_frame_dense.png"

def generate_intermediate_frame(img1_path, img2_path, alpha=0.5):
    """
    Generates an intermediate frame using Dense Optical Flow (Farneback method) 
    for motion compensation, followed by blending.
    
    Args:
        img1_path (str): Path to the first image (Frame 1, t=0).
        img2_path (str): Path to the second image (Frame 3, t=1).
        alpha (float): The blending factor (0.5 for exactly halfway).
        
    Returns:
        np.ndarray: The interpolated image, or None if error occurred.
    """
    print("1. Loading Frames...")
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("ERROR: One or both images could not be loaded. Check paths and formats.")
        return None
        
    if img1.shape != img2.shape:
        print("ERROR: Images must have the same dimensions for interpolation.")
        return None

    # Convert to grayscale for optical flow calculation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # --- Step 1: Calculate Dense Optical Flow (Motion Field) ---
    # Farneback method calculates motion vectors for EVERY pixel.
    print("2. Calculating Dense Optical Flow (Farneback Method)...")
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # --- Step 2: Warp Frames Based on Predicted Halfway Motion ---
    
    # Flow contains the x and y shift required to go from Frame 1 to Frame 3.
    # To get to the intermediate frame (t=0.5), we use flow * alpha.
    
    # 2a. Calculate the map that moves Frame 1 forward halfway (t=0 to t=0.5)
    print("3. Warping Frame 1 forward (t=0 to t=0.5)...")
    flow_map_fwd = flow * alpha
    
    # Create coordinate maps for warping
    h, w = flow.shape[:2]
    # Creates coordinate maps (x, y) for every pixel
    map_x_fwd, map_y_fwd = np.meshgrid(np.arange(w), np.arange(h))
    
    # Shift the coordinates by the calculated motion vector
    map_x_fwd = map_x_fwd.astype(np.float32) + flow_map_fwd[:, :, 0]
    map_y_fwd = map_y_fwd.astype(np.float32) + flow_map_fwd[:, :, 1]
    
    # Apply the warp to Frame 1 (moving pixels forward)
    warped_img1 = cv2.remap(img1, map_x_fwd, map_y_fwd, interpolation=cv2.INTER_LINEAR)

    # 2b. Calculate the map that moves Frame 3 backward halfway (t=1 to t=0.5)
    print("4. Warping Frame 3 backward (t=1 to t=0.5)...")
    # The reverse flow is approximately -flow. So, reverse flow is -flow * (1 - alpha)
    flow_map_bwd = -flow * (1.0 - alpha)
    
    map_x_bwd, map_y_bwd = np.meshgrid(np.arange(w), np.arange(h))
    
    # Shift the coordinates by the calculated reverse motion vector
    map_x_bwd = map_x_bwd.astype(np.float32) + flow_map_bwd[:, :, 0]
    map_y_bwd = map_y_bwd.astype(np.float32) + flow_map_bwd[:, :, 1]
    
    # Apply the warp to Frame 3 (moving pixels backward)
    warped_img2 = cv2.remap(img2, map_x_bwd, map_y_bwd, interpolation=cv2.INTER_LINEAR)
    
    # --- Step 3: Blend the Warped Images ---
    # This final blending step fills in any small holes left by the warping.
    print("5. Blending the two warped frames...")
    intermediate_frame = cv2.addWeighted(warped_img1, 0.5, warped_img2, 0.5, 0.0)
    
    return intermediate_frame

# --- Execution as a Command Line Utility ---

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("\n--- ERROR ---")
        print("Usage: python improved_frame_generator.py <path_to_frame_1> <path_to_frame_3>")
        print(f"Output will be saved as: {OUTPUT_FILENAME}")
        sys.exit(1)

    frame1_path = sys.argv[1]
    frame3_path = sys.argv[2]
    
    print(f"\n--- Improved Frame Generation Utility (Dense Flow) ---")
    print(f"Input Frame 1 (t=0): {frame1_path}")
    print(f"Input Frame 3 (t=1): {frame3_path}")
    
    # Generate the frame (Frame 2 at t=0.5)
    generated_frame = generate_intermediate_frame(frame1_path, frame3_path, alpha=0.5)
    
    if generated_frame is not None:
        # Save the resulting image
        cv2.imwrite(OUTPUT_FILENAME, generated_frame)
        
        print("\n--- SUCCESS ---")
        print(f"Generated Frame 2 saved successfully as: ./{OUTPUT_FILENAME}")
        print("Now you can validate this file using 'image_validator.py'")
        
    else:
        print("\n--- FAILURE ---")
        print("Frame generation failed. See error messages above.")
