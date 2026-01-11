"""
Vase profile curve fitting and visualization.

This module loads detected profile points for multiple vase designs, splits the
profile into ranges, fits polynomials per range (with simple uncertainty
estimates), finds intersections between consecutive fits, and produces a PDF
plot and saves the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import pickle

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_points(filepath: str) -> np.ndarray:
    """
    Load profile points from a semicolon-separated CSV and normalize Y.

    Args:
        filepath (str): Path to a CSV file with header "x;y" and subsequent
            rows of numeric values.

    Returns:
    np.ndarray: N×2 array of points [x, y]
    """
    logger.info(f"Loading points from {filepath}")
    points = pd.read_csv(
        filepath,
        delimiter=";",
        comment="#",
        names=["x", "y"],
        skiprows=1
    ).to_numpy()
    logger.info(f"Loaded {len(points)} points")
    points[:, 1] = np.abs(points[:, 1])
    return points


def get_min_max_x(points: np.ndarray):
    """
    Return points with smallest and largest x.

    Args:
    points (np.ndarray): N×2 array of points [x, y].

    Returns:
    Tuple[np.ndarray, np.ndarray]: The row with minimum x and the row
    with maximum x.
    """
    min_x_point = min(points, key=lambda point: point[0])
    max_x_point = max(points, key=lambda point: point[0])
    logger.info(f"Point with smallest x: {min_x_point}")
    logger.info(f"Point with largest x: {max_x_point}")
    return min_x_point, max_x_point


def get_ranges(points: np.ndarray, split_points_indexes) -> list:
    """
    Return list of point ranges for fitting.

    The function splits sorted points into contiguous ranges according to the
    provided split indexes.

    Args:
        points (np.ndarray): Sorted array of points (increasing x).
        split_points_indexes (List[int]): Indexes in the sorted
            array indicating where a new segment should begin.

    Returns:
        List[np.ndarray]: List of point subarrays, each N_i×2.
    """
    indices = [0] + split_points_indexes + [len(points)]
    # Add +1 to the end index of each range to include the split point itself

    ranges = [points[indices[i]:indices[i+1]+1] for i in range(len(indices)-1)]

    logger.info(f"No of point ranges: {len(ranges)}")
    return ranges


def find_exact_intersections(polys, x_ranges):
    """
    Find an intersection of each consecutive polynomial pair within overlap.

    Args:
        polys (List[np.poly1d]): Fitted polynomials (in descending powers).
        x_ranges (List[Tuple[float, float]]): (min_x, max_x) fit domains per
            polynomial.

    Returns:
        List[Tuple[float, float]]: One intersection (x, p_i(x)) per adjacent
        pair, when found.
    """
    intersections = []
    for i in range(len(polys) - 1):
        p1 = polys[i]
        p2 = polys[i + 1]
        # Difference polynomial
        p_diff = p1 - p2
        # Find real roots
        roots = np.roots(p_diff)
        # Use only real roots within the overlapping x-range
        x_min = max(x_ranges[i][0], x_ranges[i + 1][0])
        x_max = min(x_ranges[i][1], x_ranges[i + 1][1])

        # Add some margin to the range
        margin_min = x_min * 0.2
        margin_max = x_max * 0.2
        x_min -= margin_min
        x_max += margin_max
        
        # Collect valid roots
        valid_roots = []
        for r in roots:
            if np.isreal(r):
                x_real = np.real(r)
                if x_min <= x_real <= x_max:
                    valid_roots.append(x_real)
        
        # If we found any valid roots
        if valid_roots:
            # Find the shared boundary between ranges
            boundary = (x_ranges[i][1] + x_ranges[i+1][0]) / 2
            
            # Find the root closest to the shared boundary
            best_root = min(valid_roots, key=lambda x: abs(x - boundary))
            y_real = p1(best_root)
            intersections.append((best_root, y_real))
        else:
            # If no intersection found, log a warning
            logger.warning(
                f"No intersection found between polynomials {i} and {i+1}"
            )
            
    return intersections


def estimate_polyfit_with_uncertainties(x, y, degree, sigma=1.0):
    """
    Calculate polynomial fit with uncertainties.

    Args:
        x (np.ndarray): X data (1D).
        y (np.ndarray): Y data (1D).
        degree (int): Polynomial degree.
        sigma (float): Per-point Y std-dev used to build equal weights
            (1/sigma).

    Returns:
        coeffs (np.ndarray): Polynomial coefficients.
        coeffs_std (np.ndarray): 1-sigma uncertainties.
    """
    # Fit polynomial and get covariance matrix
    coeffs, cov = np.polyfit(
        x, y, degree, cov=True, w=np.full_like(y, 1 / sigma)
    )
    coeffs_std = np.sqrt(np.diag(cov))
    return coeffs, coeffs_std


def plot_polynomial_fits(
    adjusted_points_array,
    ranges,
    degrees,
    vase_name,
    show_plots=False,
):
    """
    Plot data/fits/uncertainty bands, and mark intersections.

    Args:
        adjusted_points_array (np.ndarray): Sorted N×2 array of all points.
        ranges (List[np.ndarray]): List of subarrays (one per range).
        degrees (List[int]): Polynomial degree per range.
        vase_name (str): Vase name.
        show_plots (bool): If True, display the plot interactively.

    Returns:
        dict: Results with keys:
            - fit_coefficient: list of coeff arrays per range
            - fit_coefficient_up/down: coeff arrays from Y±0.1 cm
              perturbation
            - fit_coefficient_unc: 1-sigma coefficient uncertainties
            - intersections: list of (x, y) boundary points
            - poly_degree: degrees used per range
            - r_squared: R² per range
    """
    plt.figure(figsize=(12, 10))  # Made taller to accommodate legend at bottom

    # Replace manual color list with matplotlib's default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    polys = []
    x_ranges = []
    fit_coefficient = []
    fit_coefficient_up = []
    fit_coefficient_down = []
    fit_coefficient_unc = []
    poly_degree = []
    r_squared_values = []  # New list to store R-squared values

    # Fit and plot each polynomial
    for i, (r, degree) in enumerate(zip(ranges, degrees)):
        if len(r) > 0:
            x_range, y_range = r[:, 0], r[:, 1]

            # Estimate coefficients and their uncertainties
            coeffs, coeffs_std = estimate_polyfit_with_uncertainties(
                x_range, y_range, degree, sigma=1.0
            )
            coeffs_up, _ = estimate_polyfit_with_uncertainties(
                x_range, y_range + 0.1, degree, sigma=1.0
            )
            coeffs_down, _ = estimate_polyfit_with_uncertainties(
                x_range, y_range - 0.1, degree, sigma=1.0
            )

            poly = np.poly1d(coeffs)
            polys.append(poly)
            x_ranges.append((min(x_range), max(x_range)))

            # Calculate R-squared for this fit
            r_squared = calculate_r_squared(x_range, y_range, poly)
            r_squared_values.append(r_squared)

            logger.info(
                f"Coefficients for Range {i + 1} (degree {degree}): {coeffs}"
            )
            logger.info(f"Uncertainties for Range {i + 1}: {coeffs_std}")
            logger.info(f"R-squared for Range {i + 1}: {r_squared:.4f}")

            fit_coefficient.append(coeffs)
            fit_coefficient_unc.append(coeffs_std)
            fit_coefficient_up.append(coeffs_up)
            fit_coefficient_down.append(coeffs_down)

            poly_degree.append(degree)

            # Extend the fit for better intersection visualization
            x_fit = np.linspace(min(x_range) - 0.2, max(x_range) + 0.2, 100)
            y_fit = poly(x_fit)
            # Draw hollow markers: colored edge, transparent interior
            plt.scatter(
                x_range,
                y_range,
                label=f'{i + 1}: Points',
                edgecolors=colors[i],
                facecolors='none',
                marker='P',
                s=60,
                linewidths=0.8,
                alpha=1,
            )
            plt.plot(
                x_fit,
                y_fit,
                linestyle='-',
                label=f'{i + 1}: Fit',
                color=colors[i],
                zorder=4,
            )

            # Slope/intercept + uncertainty
            poly_plus = np.poly1d(coeffs_up)
            y_fit_plus = poly_plus(x_fit)
            # plt.plot(x_fit, y_fit_plus, linestyle='dotted',
            #          color=colors[i], alpha=0.7)

            # Slope/intercept - uncertainty
            poly_minus = np.poly1d(coeffs_down)
            y_fit_minus = poly_minus(x_fit)
            # plt.plot(x_fit, y_fit_minus, linestyle='dotted',
            #          color=colors[i], alpha=0.7)

            # Fill uncertainty band without drawing border lines
            # (edgecolor='none', linewidth=0)
            plt.fill_between(
                x_fit,
                y_fit_minus,
                y_fit_plus,
                color=colors[i],
                alpha=0.33,
                edgecolor='none',
                linewidth=0,
                label=f'{i + 1}: Uncertainty',
                zorder=3,
            )

    # Find and plot intersections between consecutive fitted polynomials
    intersections = find_exact_intersections(polys, x_ranges)

    # Add first and last point from fitted ranges to intersections
    if len(ranges) > 0:
        first_range = ranges[0]
        last_range = ranges[-1]
        if len(first_range) > 0:
            first_point = (0, polys[0](0))  # First point at x=0
            intersections = [first_point] + intersections
        if len(last_range) > 0:
            last_point = (last_range[-1][0], polys[-1](last_range[-1][0]))
            intersections = intersections + [last_point]

    if intersections:
        intersections_np = np.array(intersections)
        plt.scatter(
            intersections_np[:, 0],
            intersections_np[:, 1],
            color='black',
            marker='+',
            s=350,
            label='Integration boundaries',
            zorder=5,
        )
        for idx, (x_int, y_int) in enumerate(intersections):
            # plt.text(x_int - 2, y_int + 2,
            #          f"Inter B.: ({x_int:.0f}, {y_int:.0f})",
            #          fontsize=9, color='C9', ha='right', va='top')
            logger.info(
                f"Integration boundaries: ({x_int:.4f}, {y_int:.4f})"
            )

    plt.xlabel('X [cm]')
    plt.ylabel('Y [cm]')
    plt.axis('scaled')

    # Add a dummy plot for spacing in legend
    if len(ranges) > 1:
        plt.plot(np.nan, np.nan, '-', color='none', label=' ')

    # Position legend outside at top center of the plot
    plt.legend(
        loc='lower center',  # bbox_to_anchor=(0.5, 1.5),
        ncol=6 if len(ranges) > 4 else 5,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    # Extend y-axis a bit for better visibility
    plt.ylim(0, np.ceil(plt.ylim()[1]))
    plt.minorticks_on()
    # Ensure grids are drawn behind plot elements
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(True, which='major', linestyle='-', linewidth=1.0, zorder=0)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.7, zorder=0)

    # Adjust layout to make room for legend at top
    # plt.tight_layout(rect=[0, 0, 1, 0.9])  # [left, bottom, right, top]
    plt.tight_layout()  # [left, bottom, right, top]

    plt.savefig(f"{vase_name}/_fits_polynomial.pdf", bbox_inches='tight')

    if show_plots:
        plt.show()

    # Return map of results
    return {
        "fit_coefficient": fit_coefficient,
        "fit_coefficient_up": fit_coefficient_up,
        "fit_coefficient_down": fit_coefficient_down,
        "fit_coefficient_unc": fit_coefficient_unc,
        "intersections": intersections,
        "poly_degree": poly_degree,
        "r_squared": r_squared_values,  # Add R-squared values to results
    }


def save_results_to_pickle(results: dict, filename: str):
    """
    Save results pickle file.

    Args:
                results (dict): Output map from plot_polynomial_fits.
                filename (str): Target pickle path.
    """
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Results saved to {filename}")


def save_fit_results_to_text(results: dict, vase_name: str):
    """
    Save fit results to text file.
    
    Args:
        results (dict): Dictionary from plot_polynomial_fits
        vase_name (str): Vase directory name (output path prefix)
    """
    import os
    logger.info(f"Saving fit results to text file for {vase_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(vase_name, exist_ok=True)
    
    output_text = []
    output_text.append(f"Polynomial Fitting Results for {vase_name}")
    output_text.append("=" * 50)
    output_text.append("")
    
    # Add polynomial coefficients and uncertainties
    fit_coefficient = results["fit_coefficient"]
    fit_coefficient_unc = results["fit_coefficient_unc"]
    fit_coefficient_up = results["fit_coefficient_up"]
    fit_coefficient_down = results["fit_coefficient_down"]
    poly_degree = results["poly_degree"]
    r_squared = results.get("r_squared", [0] * len(fit_coefficient))
    
    output_text.append("Polynomial Coefficients:")
    output_text.append("-" * 50)
    
    for i, (
        coeffs,
        coeffs_unc,
        coeffs_up,
        coeffs_down,
        degree,
        r2,
    ) in enumerate(
        zip(
            fit_coefficient,
            fit_coefficient_unc,
            fit_coefficient_up,
            fit_coefficient_down,
            poly_degree,
            r_squared,
        )
    ):
        output_text.append(f"\nSection {i+1} - Polynomial Degree: {degree}")
        output_text.append(f"Coefficient of determination (R²): {r2:.6f}")
        
        # Format nominal polynomial equation with uncertainties
        terms = []
        for j, (c, unc) in enumerate(zip(coeffs, coeffs_unc)):
            power = degree - j
            if power > 1:
                terms.append(f"({c:.6f} ± {unc:.6f})x^{power}")
            elif power == 1:
                terms.append(f"({c:.6f} ± {unc:.6f})x")
            else:
                terms.append(f"({c:.6f} ± {unc:.6f})")
        
        poly_eq = " + ".join(terms)
        output_text.append(f"p(x) = {poly_eq}")
        
        # Format upper bound polynomial
        up_terms = []
        for j, c in enumerate(coeffs_up):
            power = degree - j
            if power > 1:
                up_terms.append(f"{c:.6f}x^{power}")
            elif power == 1:
                up_terms.append(f"{c:.6f}x")
            else:
                up_terms.append(f"{c:.6f}")
        
        up_poly_eq = " + ".join(up_terms)
        output_text.append(f"p_up(x) = {up_poly_eq}")
        
        # Format lower bound polynomial
        down_terms = []
        for j, c in enumerate(coeffs_down):
            power = degree - j
            if power > 1:
                down_terms.append(f"{c:.6f}x^{power}")
            elif power == 1:
                down_terms.append(f"{c:.6f}x")
            else:
                down_terms.append(f"{c:.6f}")
        
        down_poly_eq = " + ".join(down_terms)
        output_text.append(f"p_down(x) = {down_poly_eq}")
    
    # Add intersection points
    intersections = results["intersections"]
    output_text.append("\nIntegration Boundaries:")
    output_text.append("-" * 50)
    
    for i, (x, y) in enumerate(intersections):
        output_text.append(f"Boundary {i+1}: x = {x:.4f} cm, y = {y:.4f} cm")
    
    # Add statistics
    output_text.append("\nFit Statistics:")
    output_text.append("-" * 50)
    output_text.append(f"Number of sections: {len(poly_degree)}")
    output_text.append(f"Polynomial degrees: {poly_degree}")
    output_text.append(f"Average R²: {sum(r_squared)/len(r_squared):.6f}")
    
    # Write to file
    with open(f"{vase_name}/_fit_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_text))
    
    logger.info(f"Fit results saved to {vase_name}/_fit_results.txt")


def updateGlobalPlotSettings():
    """
    Update global Matplotlib settings for consistent fonts and sizing.
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 18
    })


def calculate_r_squared(x, y, poly):
    """
    Calculate the coefficient of determination for a polynomial fit.
    
    Args:
        x (np.ndarray): X data points
        y (np.ndarray): Y data points (actual values)
        poly (np.poly1d): Fitted polynomial
        
    Returns:
        float: coefficient of determination.
    """
    y_pred = poly(x)
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def main():
    """
    Load data, fit polynomials per range, export plots and results per vase.
    """
    updateGlobalPlotSettings()

    # Vase specifications with split points and polynomial degrees
    vases = {
        "Vase_01": {"Split points": [], "Degrees": [1]},
        "Vase_02": {
            "Split points": [12, 26, 33, 48, 60],
            "Degrees": [1, 1, 1, 1, 1],
        },
        "Vase_03": {"Split points": [49, 62, 97], "Degrees": [1, 2, 2, 1]}
    }

    # Process each vase
    for vase_name, vase_data_info in vases.items():
        logger.info(f"Processing {vase_name} vase design")

        # Directory and data file configuration
        DIR = vase_name
        DATA_FILE = f"{DIR}/_detected_points.csv"

        points_array = load_points(DATA_FILE)
        # get_min_max_x(points_array)

        # Sort by increasing x
        adjusted_points_array = points_array.copy()
        adjusted_points_array = adjusted_points_array[
            adjusted_points_array[:, 0].argsort()
        ]

        ranges = get_ranges(
            adjusted_points_array,
            vase_data_info["Split points"],
        )
        results = plot_polynomial_fits(
            adjusted_points_array,
            ranges,
            vase_data_info["Degrees"],
            vase_name,
            show_plots=False,
        )

        # Save results to pickle
        save_results_to_pickle(results, f"{DIR}/_fit_results.pkl")
        # Save results to text file
        save_fit_results_to_text(results, vase_name)


if __name__ == "__main__":
    main()


