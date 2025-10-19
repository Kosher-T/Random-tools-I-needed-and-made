# --- Image Similarity Validator (SSIM) - Command Line Utility ---

# GOAL: Accept two image file paths from the command line, load them, 
# and calculate the Structural Similarity Index Measure (SSIM) between them.
# The first image (target) serves as the standard, and the second (candidate) 
# is the one being evaluated.

import cv2
import numpy as np
import sys # Required to read command line arguments
from skimage.metrics import structural_similarity as ssim

# --- Core Similarity Function (Unchanged) ---

def calculate_ssim(imgA, imgB):
    """
    Calculates the Structural Similarity Index Measure (SSIM) between two images.
    
    Args:
        imgA (np.ndarray): The first image (Target/Standard).
        imgB (np.ndarray): The second image (Candidate/Comparison).
        
    Returns:
        float: The SSIM score (1.0 = identical, 0.0 = completely different).
    """
    # 1. Convert to Grayscale
    # SSIM often works best on grayscale images.
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    
    # 2. Check Dimensions
    # Critical: SSIM requires both images to have the exact same size.
    if grayA.shape != grayB.shape:
        print("Error: Images must have the same dimensions for comparison.")
        return 0.0
    
    # 3. Calculate SSIM
    (score, diff_map) = ssim(grayA, grayB, full=True)
    
    return score

# --- Execution as a Command Line Utility ---

if __name__ == "__main__":
    
    # Check if the user provided exactly two file paths
    if len(sys.argv) != 3:
        print("\n--- ERROR ---")
        print("Usage: python image_validator.py <path_to_target_image> <path_to_candidate_image>")
        print("Example: python image_validator.py frames/real_f2.png frames/generated_f2.png")
        sys.exit(1)

    # File paths are stored in sys.argv[1] and sys.argv[2]
    target_path = sys.argv[1]
    candidate_path = sys.argv[2]
    
    print(f"\n--- SSIM Validation Utility ---")
    print(f"Target/Standard Image: {target_path}")
    print(f"Candidate/Comparison Image: {candidate_path}")

    # Load images using OpenCV
    img_target = cv2.imread(target_path)
    img_candidate = cv2.imread(candidate_path)

    # Check if images were loaded successfully
    if img_target is None:
        print(f"\nERROR: Could not load Target image at '{target_path}'. Check path and file format.")
        sys.exit(1)
        
    if img_candidate is None:
        print(f"\nERROR: Could not load Candidate image at '{candidate_path}'. Check path and file format.")
        sys.exit(1)
    
    # Calculate the similarity score
    try:
        ssim_score = calculate_ssim(img_target, img_candidate)
        
        # Output the final result
        print("\n--- RESULT ---")
        print(f"Structural Similarity Index Measure (SSIM) Score: {ssim_score:.6f}")
        
        # Provide interpretation
        if ssim_score >= 0.95:
            print("Interpretation: Excellent similarity. The candidate frame is nearly identical to the target.")
        elif ssim_score >= 0.70:
            print("Interpretation: Good similarity. The structure and contrast are well preserved.")
        else:
            print("Interpretation: Low similarity. The frames differ significantly.")

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        sys.exit(1)
