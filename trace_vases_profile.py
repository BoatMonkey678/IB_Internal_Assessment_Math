"""
Vase Profile Detection

This script processes vase profile images to detect point locations and save
the results in both CSV and PDF formats.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import logging
import sys
import traceback

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def validate_image_path(image_path: str) -> bool:
    """
    Validate whether the image path exists and is accessible.
    
    Args:
        image_path (str): Path to the image file to validate
        
    Returns:
        bool: True if file exists and is a regular file; False otherwise.
    """
    logger.debug(f"Validating image path: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image file does not exist: {image_path}")
        return False
    
    if not os.path.isfile(image_path):
        logger.error(f"Path is not a valid file: {image_path}")
        return False
    
    return True


def load_and_threshold_image(image_path: str, threshold: int = 200) -> np.ndarray:
    """
    Load and threshold an image to create a binary mask. Values above the
    threshold become white (255) while values below become black (0).

    Args:
        image_path (str): Path to the input image
        threshold (int): Threshold value for binarization (0-255)

    Returns:
        np.ndarray: Binary image (uint8) with values {0, 255}, rotated 90°
        clockwise.        
    """
    logger.info(f"Attempting to load image from {image_path}")
    
    # Validate image path
    if not validate_image_path(image_path):
        error_msg = f"Invalid image path: {image_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        # Load the image in grayscale mode
        logger.debug("Reading image in grayscale mode")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image was loaded successfully
        if image is None:
            error_msg = f"Failed to load image: {image_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Rotate image
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Log image properties
        height, width = image.shape
        logger.info(f"Image loaded successfully: {width}x{height} pixels")
            
        # Apply thresholding to create binary image
        logger.debug(f"Applying threshold {threshold} to image")
        _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        
        # Log thresholding results
        white_pixels = np.sum(binary == 255)
        white_percentage = (white_pixels / (width * height)) * 100
        logger.debug(f"Thresholding result: {white_percentage:.2f}% white pixels")
        
        logger.info(f"Successfully loaded and thresholded image: {image_path}")
        return binary
        
    except Exception as e:
        error_msg = f"Error processing image {image_path}: {str(e)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        exit(1)


def detect_points(binary_image: np.ndarray) -> List[Tuple[float, float]]:
    """
    Detect points in binary image using SimpleBlobDetector.
    
    Args:
        binary_image (np.ndarray): Binary input image with 0 and 255 values

    Returns:
                List[Tuple[float, float]]: Detected point coordinates as (x, y) tuples
                in pixel units.
    """
    logger.info("Starting point detection with SimpleBlobDetector")
    
    # Configure blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by area - adjust based on expected point size in the image
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1000
    logger.debug(f"Blob area filter: min={params.minArea}, max={params.maxArea}")
    
    # Filter by circularity - ensures detected blobs are approximately circular
    params.filterByCircularity = True
    params.minCircularity = 0.7  # 1.0 is a perfect circle
    logger.debug(f"Blob circularity filter: min={params.minCircularity}")

    # Create detector with parameters and detect keypoints
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary_image)
    
    # Extract point coordinates from keypoints
    detected_points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
    
    logger.info(f"Detected {len(detected_points)} points in the image")
    
    if detected_points:
        # Log some statistics about detected points
        x_coords = [pt[0] for pt in detected_points]
        y_coords = [pt[1] for pt in detected_points]
        
        logger.debug(f"X-coordinate range: {min(x_coords):.1f} to {max(x_coords):.1f}")
        logger.debug(f"Y-coordinate range: {min(y_coords):.1f} to {max(y_coords):.1f}")
    else:
        logger.warning("No points were detected in the image")

    return detected_points


def scale_points(points: np.ndarray, width_range: float, 
                height_range: float) -> np.ndarray:
    """
    Scale point coordinates to match physical dimensions.
    
    Args:
        points (np.ndarray): Array of point coordinates
                width_range (float): Target width range in cm (final Y extent)
                height_range (float): Target height range in cm (final X extent)
        
    Returns:
                np.ndarray: Scaled point coordinates in cm
    """
    # Extract original pixel dimensions
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    pixel_width = x_max - x_min
    pixel_height = y_max - y_min
    
    logger.debug(f"Original pixel dimensions: width={pixel_width:.1f}, height={pixel_height:.1f}")
    
    # Calculate scaling factors (cm per pixel)
    scale_x = height_range / pixel_width  # Note the swap of width/height due to
    scale_y = width_range / pixel_height  # coordinate transformation later
    
    logger.info(f"Scaling factors: x={scale_x:.6f} cm/px, y={scale_y:.6f} cm/px")
    
    # Make a copy to avoid modifying the original array
    scaled_points = points.copy()
    
    # Apply scaling
    scaled_points[:, 0] *= scale_x
    scaled_points[:, 1] *= scale_y
    
    # Log the results of scaling
    logger.debug(f"After scaling: x range={scaled_points[:, 0].min():.2f} to {scaled_points[:, 0].max():.2f} cm")
    logger.debug(f"After scaling: y range={scaled_points[:, 1].min():.2f} to {scaled_points[:, 1].max():.2f} cm")
    
    return scaled_points


def apply_thickness_adjustments(points: np.ndarray, wall_thickness: float, bottom_thickness: float) -> np.ndarray:
    """
    Apply wall and bottom thicknesses adjustments to point coordinates.
    
    Args:
        points (np.ndarray): Array of point coordinates (cm)
        wall_thickness (float): Wall thickness offset in cm 
        bottom_thickness (float): Base thickness in cm
        
    Returns:
        np.ndarray: Points with thickness adjustments applied (cm)
    """
    logger.debug("Applying wall thickness offsets")
    
    # Make a copy of points to avoid modifying the input
    adjusted_points = points.copy()
    
    # Normalize X coordinates (shift to start at 0)
    x_min = adjusted_points[:, 0].min()
    adjusted_points[:, 0] -= x_min
    logger.debug(f"Normalized X coordinates by subtracting {x_min:.3f}")
    
    # Apply vase bottom thickness adjustment
    adjusted_points[:, 0] -= bottom_thickness
    logger.debug(f"Applied bottom thickness offset: -{bottom_thickness} cm")
    
    # Filter out points with negative X (below the base)
    original_count = len(adjusted_points)
    adjusted_points = adjusted_points[adjusted_points[:, 0] >= 0]
    filtered_count = len(adjusted_points)
    
    if filtered_count < original_count:
        logger.debug(f"Filtered out {original_count - filtered_count} points below base")
    
    # Center points vertically
    adjusted_points[:, 1] -= adjusted_points[:, 1].min()
    adjusted_points[:, 1] -= adjusted_points[:, 1].max() / 2
    logger.debug("Centered points vertically")
    
    # Apply wall thickness adjustments based on Y coordinate sign
    positive_mask = adjusted_points[:, 1] >= 0
    negative_mask = adjusted_points[:, 1] < 0
    
    # Count points in each section for logging
    positive_count = np.sum(positive_mask)
    negative_count = np.sum(negative_mask)
    
    logger.debug(f"Adjusting wall thickness for {positive_count} positive Y points")
    logger.debug(f"Adjusting wall thickness for {negative_count} negative Y points")
    
    # Apply thickness adjustments
    adjusted_points[positive_mask, 1] -= wall_thickness  # Positive Y points
    adjusted_points[negative_mask, 1] += wall_thickness  # Negative Y points
    
    return adjusted_points


def balance_sycmetry(points: np.ndarray) -> np.ndarray:
    """
    Balance the symmetry of points around the center axis.
    
    Args:
        points (np.ndarray): Array of point coordinates (cm)
        
    Returns:
        np.ndarray: Balanced point coordinates (cm)
     """
    # Calculate average positions for positive and negative sides
    positive_mask = points[:, 1] >= 0
    negative_mask = points[:, 1] < 0
    
    if not np.any(positive_mask) or not np.any(negative_mask):
        logger.warning("Cannot balance sycmetry - missing points on one side")
        return points
    
    avg_max = points[positive_mask, 1].mean()
    avg_min = points[negative_mask, 1].mean()
    
    # Calculate adjustment to balance the profile
    move = (avg_max + avg_min) / 2 / 2
    logger.info(f"Sycmetry adjustment: avg_max={avg_max:.3f}, avg_min={avg_min:.3f}, move={move:.3f}")
    
    # Apply adjustment to both sides
    balanced_points = points.copy()
    balanced_points[:, 1] -= move
    
    return balanced_points


def process_points(
    points: np.ndarray, width_range: float, height_range: float, wall_thickness: float, bottom_thickness: float
) -> np.ndarray:
    """
    Scale and transform detected points to match physical dimensions (cm).

    Args:
        points (np.ndarray): Array of point coordinates
        width_range (float): Target width range in cm
        height_range (float): Target height range in cm
        wall_thickness (float): Wall thickness offset in cm
        bottom_thickness (float): Bottom/base thickness in cm

    Returns:
        np.ndarray: Processed point coordinates in physical dimensions in cm
    """
    logger.info(f"Processing {len(points)} points with dimensions: " 
              f"width={width_range} cm, height={height_range} cm, wall_thickness={wall_thickness} cm, bottom_thickness={bottom_thickness} cm")
    
    # Step 1: Scale points to physical dimensions
    points = scale_points(points, width_range, height_range)
    
    # Step 2: Apply thickness adjustments and filtering
    points = apply_thickness_adjustments(points, wall_thickness, bottom_thickness)
    
    # Step 3: Balance profile sycmetry
    points = balance_sycmetry(points)
    
    # Log final statistics
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    
    logger.info(f"Processing complete. Final dimensions: " 
              f"height={x_range:.2f} cm, width={y_range:.2f} cm")
    logger.info(f"Final point count: {len(points)}")
    
    return points


def setup_plot(
    plot_width_cm: float, plot_height_cm: float, margins_inch: float = 0.5
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Set up matplotlib figure.

    Args:
        plot_width_cm (float): Plot width in centimeters (excluding margins)
        plot_height_cm (float): Plot height in centimeters (excluding margins)
        margins_inch (float): Margins in inches around the plot area

    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes configured for plotting
    """
    logger.debug(f"Setting up plot with dimensions: {plot_width_cm} × {plot_height_cm} cm")
    
    # Convert millimeters to inches (matplotlib uses inches)
    cm_to_inch = 1 / 2.54
    plot_width_inch = plot_width_cm * cm_to_inch
    plot_height_inch = plot_height_cm * cm_to_inch
    
    # Calculate figure dimensions including margins
    fig_width_inch = plot_width_inch + 2 * margins_inch
    fig_height_inch = plot_height_inch + 2 * margins_inch
    
    logger.debug(f"Figure size: {fig_width_inch:.2f} × {fig_height_inch:.2f} inches")
    
    # Create figure with high resolution
    fig = plt.figure(dpi=300)
    fig.set_size_inches(fig_width_inch, fig_height_inch)
    
    # Position the axes precisely within the figure
    ax = fig.add_axes(
        [
            margins_inch / fig_width_inch,         # Left margin (normalized)
            margins_inch / fig_height_inch,        # Bottom margin (normalized)
            plot_width_inch / fig_width_inch,      # Width (normalized)
            plot_height_inch / fig_height_inch,    # Height (normalized)
        ]
    )
    
    logger.debug("Plot setup complete")
    return fig, ax


def create_output_directory(dir_path: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        dir_path (str): Directory path to create
    """
    if not os.path.exists(dir_path):
        logger.info(f"Creating output directory: {dir_path}")
        os.makedirs(dir_path)
    else:
        logger.debug(f"Output directory already exists: {dir_path}")


def save_plot_and_data(points: np.ndarray, output_dir: str) -> None:
    """
    Save processed points and plot to files.
    
    Args:
        points (np.ndarray): Processed point coordinates
        output_dir (str): Output directory path

    Outputs:
        - CSV: "_detected_points.csv" with semicolon delimiter and header "x;y"
        - PDF: "_detected_points.pdf" showing the plotted profile
    """
    # Save points to CSV file
    csv_path = f"{output_dir}/_detected_points.csv"
    logger.info(f"Saving {len(points)} points to CSV: {csv_path}")
    np.savetxt(csv_path, points, delimiter=";", header="x;y", comments="#")
    
    # Save plot to PDF
    pdf_path = f"{output_dir}/_detected_points.pdf"
    logger.info(f"Saving plot to PDF: {pdf_path}")
    plt.savefig(pdf_path, dpi=300)


def process_vase(name: str, dimensions: Dict[str, float], 
                plot_width: float, plot_height: float) -> None:
    """
    Process a single vase image.
    
    Args:
        name (str): Vase name/identifier (also the folder containing inputs/outputs)
        dimensions (Dict[str, float]): Physical dimensions in cm with keys:
            - width, height, wall_thickness, bottom_thickness
        plot_width (float): Plot width in cm
        plot_height (float): Plot height in cm
    """
    logger.info("=" * 50)
    logger.info(f"Processing vase: {name}")
    logger.info(f"Dimensions: {dimensions}")
    
    # Setup output directory
    output_dir = name
    create_output_directory(output_dir)
    
    # Construct image path
    image_path = f"{output_dir}/scan_filtered.png"
    
    try:
        # Load and process image
        binary = load_and_threshold_image(image_path)
        
        # Detect points in the image
        detected_points = detect_points(binary)
        if not detected_points:
            logger.error(f"No points detected in {image_path}. Skipping.")
            return
            
        # Convert detected points to numpy array
        points = np.array(detected_points).reshape(-1, 2)
        logger.info(f"Converting {len(detected_points)} points to numpy array")
        
        # Process points to physical coordinates
        points = process_points(
            points, 
            dimensions["width"], 
            dimensions["height"], 
            dimensions["wall_thickness"],
            dimensions["bottom_thickness"],
        )
        
        # Create plot with precise dimensions
        fig, ax = setup_plot(plot_width, plot_height)
        
        # Configure plot
        logger.debug("Configuring plot appearance and data")
        
        # Set axis limits with 20% padding
        ax.set_xlim(1.2 * points[:, 0].min(), 1.2 * points[:, 0].max())
        ax.set_ylim(0, 1.2 * points[:, 1].max())
        
        # Plot positive Y points
        positive_points = points[points[:, 1] >= 0]
        if len(positive_points) > 0:
            ax.scatter(positive_points[:, 0], positive_points[:, 1], 
                      s=2, color='blue', label='Positive side')
            logger.debug(f"Plotted {len(positive_points)} points on positive side")
        
        # Plot absolute values of negative Y points
        negative_points = points[points[:, 1] < 0]
        if len(negative_points) > 0:
            ax.scatter(negative_points[:, 0], np.abs(negative_points[:, 1]), 
                      s=2, color='red', label='Negative side')
            logger.debug(f"Plotted {len(negative_points)} points on negative side")
        
        # Add grid, labels and styling
        ax.grid(True, which="major", linewidth=0.5)
        ax.grid(True, which="minor", linewidth=0.2)
        ax.set_xlabel("cm")
        ax.set_ylabel("cm")
        ax.set_title(f"Vase Profile: {name}")
        
        # Save results
        save_plot_and_data(points, output_dir)
        
        # Show the plot
        # logger.info(f"Displaying plot for {name}")
        # plt.show()
        
        logger.info(f"Completed processing for {name}")
        
    except Exception as e:
        logger.error(f"Error processing vase {name}: {str(e)}")
        logger.debug(traceback.format_exc())


def main():
    """
    Main function to process multiple vase images.
    """
    logger.info("Starting vase profile detection and processing")
    
    # Constants for plot dimensions
    PLOT_WIDTH_CM = 27 / 3.5  # Approximately 7.7 cm
    PLOT_HEIGHT_CM = 21 / 3.5  # Approximately 6.0 cm
    logger.info(f"Plot dimensions: {PLOT_WIDTH_CM:.1f} × {PLOT_HEIGHT_CM:.1f} cm")

    # Vase specifications with physical dimensions
    vases = {
        "Vase_01": {"width": 18.1, "height": 20.0, "wall_thickness": .45, "bottom_thickness" : .5},
        "Vase_02": {"width": 15.6, "height": 23.7, "wall_thickness": .55, "bottom_thickness" : .6},
        "Vase_03": {"width": 11.1, "height": 29.4, "wall_thickness": .42, "bottom_thickness" : .42},
    }
    
    logger.info(f"Processing {len(vases)} vase designs")

    # Process each vase
    for vase_name, dimensions in vases.items():
        try:
            process_vase(vase_name, dimensions, PLOT_WIDTH_CM, PLOT_HEIGHT_CM)
        except Exception as e:
            logger.error(f"Failed to process {vase_name}: {str(e)}")
            continue
    
    logger.info("Vase processing completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)