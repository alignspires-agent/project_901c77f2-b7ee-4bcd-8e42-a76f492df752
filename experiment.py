
import numpy as np
import sys
import logging
from typing import Tuple, List, Optional
from scipy.stats import norm
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LevyProkhorovConformalPredictor:
    """
    Implementation of Conformal Prediction under Lévy-Prokhorov Distribution Shifts
    Based on the paper: "Conformal Prediction under Lévy–Prokhorov Distribution Shifts: 
    Robustness to Local and Global Perturbations"
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize the robust conformal predictor
        
        Args:
            alpha: Significance level (1-alpha is the target coverage)
        """
        self.alpha = alpha
        self.calibration_scores = None
        self.epsilon = None
        self.rho = None
        
    def compute_worst_case_quantile(self, beta: float, empirical_dist: np.ndarray, 
                                  epsilon: float, rho: float) -> float:
        """
        Compute worst-case quantile according to Proposition 3.4
        
        Args:
            beta: Quantile level
            empirical_dist: Empirical distribution of calibration scores
            epsilon: Local perturbation parameter
            rho: Global perturbation parameter
            
        Returns:
            Worst-case quantile value
        """
        if rho >= 1 - beta:
            warnings.warn(f"rho ({rho}) >= 1 - beta ({1-beta}), quantile becomes trivial")
            return np.max(empirical_dist)
        
        # Sort calibration scores
        sorted_scores = np.sort(empirical_dist)
        n = len(sorted_scores)
        
        # Compute empirical quantile at level (beta + rho)
        quantile_level = beta + rho
        if quantile_level > 1:
            quantile_level = 1.0
            
        index = int(np.ceil(quantile_level * n)) - 1
        index = max(0, min(index, n - 1))
        
        base_quantile = sorted_scores[index]
        worst_case_quantile = base_quantile + epsilon
        
        return worst_case_quantile
    
    def compute_worst_case_coverage(self, q: float, empirical_dist: np.ndarray,
                                  epsilon: float, rho: float) -> float:
        """
        Compute worst-case coverage according to Proposition 3.5
        
        Args:
            q: Quantile value
            empirical_dist: Empirical distribution of calibration scores
            epsilon: Local perturbation parameter
            rho: Global perturbation parameter
            
        Returns:
            Worst-case coverage probability
        """
        # Shift by epsilon and compute empirical CDF
        shifted_scores = empirical_dist - epsilon
        empirical_cdf = np.mean(shifted_scores <= q)
        
        worst_case_coverage = max(0.0, empirical_cdf - rho)
        return worst_case_coverage
    
    def fit(self, calibration_scores: np.ndarray, epsilon: float, rho: float):
        """
        Fit the conformal predictor with calibration data
        
        Args:
            calibration_scores: Non-conformity scores from calibration set
            epsilon: Local perturbation parameter
            rho: Global perturbation parameter
        """
        if len(calibration_scores) == 0:
            logger.error("Calibration scores cannot be empty")
            sys.exit(1)
            
        if epsilon < 0 or rho < 0 or rho >= 1:
            logger.error(f"Invalid parameters: epsilon={epsilon}, rho={rho}")
            sys.exit(1)
            
        self.calibration_scores = calibration_scores
        self.epsilon = epsilon
        self.rho = rho
        
        logger.info(f"Fitted LP conformal predictor with epsilon={epsilon:.3f}, rho={rho:.3f}")
        
    def predict_interval(self, test_score: float) -> Tuple[float, float]:
        """
        Compute prediction interval for a test score
        
        Args:
            test_score: Non-conformity score for test point
            
        Returns:
            Tuple of (lower_bound, upper_bound) for prediction interval
        """
        if self.calibration_scores is None:
            logger.error("Predictor not fitted. Call fit() first.")
            sys.exit(1)
            
        # Adjust alpha for finite-sample correction (Corollary 4.2)
        n = len(self.calibration_scores)
        beta = self.alpha + (self.alpha - self.rho - 2) / n
        
        # Compute worst-case quantile
        worst_case_quantile = self.compute_worst_case_quantile(
            1 - beta, self.calibration_scores, self.epsilon, self.rho
        )
        
        # For regression tasks, prediction interval is [y_hat - quantile, y_hat + quantile]
        # Here we return symmetric interval around the test score
        lower_bound = test_score - worst_case_quantile
        upper_bound = test_score + worst_case_quantile
        
        return lower_bound, upper_bound
    
    def evaluate_coverage(self, test_scores: np.ndarray, true_values: np.ndarray, 
                         predicted_values: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate empirical coverage and average interval width
        
        Args:
            test_scores: Non-conformity scores for test set
            true_values: True target values
            predicted_values: Predicted values
            
        Returns:
            Tuple of (coverage_rate, avg_interval_width)
        """
        if len(test_scores) != len(true_values) or len(test_scores) != len(predicted_values):
            logger.error("Input arrays must have same length")
            sys.exit(1)
            
        coverage_count = 0
        total_width = 0.0
        
        for i, (score, true_val, pred_val) in enumerate(zip(test_scores, true_values, predicted_values)):
            lower, upper = self.predict_interval(score)
            
            # Convert score-based interval to value-based interval
            value_lower = pred_val - (upper - lower) / 2
            value_upper = pred_val + (upper - lower) / 2
            
            if value_lower <= true_val <= value_upper:
                coverage_count += 1
                
            total_width += (value_upper - value_lower)
            
        coverage_rate = coverage_count / len(test_scores)
        avg_width = total_width / len(test_scores)
        
        return coverage_rate, avg_width


class TimeSeriesSimulator:
    """
    Simulate time series data with distribution shifts
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def generate_ar1_process(self, n_samples: int, phi: float = 0.8, sigma: float = 1.0) -> np.ndarray:
        """
        Generate AR(1) process: x_t = phi * x_{t-1} + epsilon_t
        
        Args:
            n_samples: Number of time steps
            phi: AR coefficient
            sigma: Noise standard deviation
            
        Returns:
            Generated time series
        """
        x = np.zeros(n_samples)
        x[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))
        
        for t in range(1, n_samples):
            x[t] = phi * x[t-1] + np.random.normal(0, sigma)
            
        return x
    
    def introduce_distribution_shift(self, time_series: np.ndarray, shift_point: int, 
                                   shift_magnitude: float, shift_type: str = "mean") -> np.ndarray:
        """
        Introduce distribution shift at specified point
        
        Args:
            time_series: Original time series
            shift_point: Index where shift occurs
            shift_magnitude: Magnitude of shift
            shift_type: Type of shift ("mean", "variance", "both")
            
        Returns:
            Time series with distribution shift
        """
        shifted_series = time_series.copy()
        
        if shift_type == "mean":
            shifted_series[shift_point:] += shift_magnitude
        elif shift_type == "variance":
            noise = np.random.normal(0, shift_magnitude, len(shifted_series) - shift_point)
            shifted_series[shift_point:] += noise
        elif shift_type == "both":
            shifted_series[shift_point:] += shift_magnitude
            noise = np.random.normal(0, shift_magnitude/2, len(shifted_series) - shift_point)
            shifted_series[shift_point:] += noise
            
        return shifted_series
    
    def compute_nonconformity_scores(self, predictions: np.ndarray, 
                                   true_values: np.ndarray) -> np.ndarray:
        """
        Compute non-conformity scores (absolute errors)
        
        Args:
            predictions: Predicted values
            true_values: True values
            
        Returns:
            Non-conformity scores
        """
        return np.abs(predictions - true_values)


def simple_ar_predictor(time_series: np.ndarray, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple AR model predictor for demonstration
    
    Args:
        time_series: Input time series
        order: AR order
        
    Returns:
        Tuple of (predictions, true_values)
    """
    n = len(time_series)
    predictions = np.zeros(n - order)
    true_values = time_series[order:]
    
    for i in range(order, n):
        # Simple prediction: average of previous 'order' values
        predictions[i - order] = np.mean(time_series[i-order:i])
        
    return predictions, true_values


def main():
    """
    Main experiment: Test LP-based conformal prediction on time series with distribution shifts
    """
    logger.info("Starting LP Conformal Prediction Experiment on Time Series Data")
    
    try:
        # Parameters
        n_samples = 1000
        calibration_size = 300
        test_size = 200
        alpha = 0.1  # 90% coverage target
        
        # LP parameters (simulating distribution shifts)
        epsilon_values = [0.1, 0.2, 0.5]  # Local perturbation
        rho_values = [0.05, 0.1, 0.15]    # Global perturbation
        
        # Generate synthetic time series data
        logger.info("Generating synthetic time series data...")
        simulator = TimeSeriesSimulator(seed=42)
        
        # Generate base AR(1) process
        base_series = simulator.generate_ar1_process(n_samples, phi=0.8, sigma=1.0)
        
        # Introduce distribution shift at midpoint
        shift_point = n_samples // 2
        shifted_series = simulator.introduce_distribution_shift(
            base_series, shift_point, shift_magnitude=2.0, shift_type="mean"
        )
        
        # Split data: pre-shift for calibration, post-shift for test
        calibration_data = base_series[:calibration_size]
        test_data = shifted_series[shift_point:shift_point + test_size]
        
        # Generate predictions and compute scores
        logger.info("Generating predictions and computing non-conformity scores...")
        cal_predictions, cal_true = simple_ar_predictor(calibration_data)
        test_predictions, test_true = simple_ar_predictor(test_data)
        
        cal_scores = simulator.compute_nonconformity_scores(cal_predictions, cal_true)
        test_scores = simulator.compute_nonconformity_scores(test_predictions, test_true)
        
        # Experiment results storage
        results = []
        
        # Test different LP parameter combinations
        logger.info("Testing different LP parameter combinations...")
        for epsilon in epsilon_values:
            for rho in rho_values:
                # Initialize and fit LP conformal predictor
                predictor = LevyProkhorovConformalPredictor(alpha=alpha)
                predictor.fit(cal_scores, epsilon=epsilon, rho=rho)
                
                # Evaluate on test set with distribution shift
                coverage, avg_width = predictor.evaluate_coverage(
                    test_scores, test_true, test_predictions
                )
                
                results.append({
                    'epsilon': epsilon,
                    'rho': rho,
                    'coverage': coverage,
                    'avg_width': avg_width
                })
                
                logger.info(f"ε={epsilon:.2f}, ρ={rho:.2f}: Coverage={coverage:.3f}, Width={avg_width:.3f}")
        
        # Compare with standard conformal prediction (epsilon=0, rho=0)
        logger.info("Comparing with standard conformal prediction...")
        standard_predictor = LevyProkhorovConformalPredictor(alpha=alpha)
        standard_predictor.fit(cal_scores, epsilon=0.0, rho=0.0)
        std_coverage, std_width = standard_predictor.evaluate_coverage(
            test_scores, test_true, test_predictions
        )
        
        # Print final results
        logger.info("\n" + "="*60)
        logger.info("FINAL EXPERIMENT RESULTS")
        logger.info("="*60)
        
        logger.info(f"Standard Conformal Prediction (ε=0, ρ=0):")
        logger.info(f"  Coverage: {std_coverage:.3f} (Target: {1-alpha:.3f})")
        logger.info(f"  Average Interval Width: {std_width:.3f}")
        
        logger.info("\nLP Robust Conformal Prediction Results:")
        logger.info("ε\tρ\tCoverage\tWidth\t\tImprovement")
        logger.info("-" * 50)
        
        best_coverage = std_coverage
        best_config = "Standard"
        
        for result in results:
            coverage_improvement = result['coverage'] - std_coverage
            width_ratio = result['avg_width'] / std_width
            
            logger.info(f"{result['epsilon']:.1f}\t{result['rho']:.2f}\t{result['coverage']:.3f}\t\t"
                       f"{result['avg_width']:.3f}\t\t{coverage_improvement:+.3f}")
            
            if result['coverage'] > best_coverage:
                best_coverage = result['coverage']
                best_config = f"ε={result['epsilon']}, ρ={result['rho']}"
        
        logger.info("\nSUMMARY:")
        logger.info(f"Best configuration: {best_config}")
        logger.info(f"Best coverage achieved: {best_coverage:.3f}")
        logger.info(f"Target coverage: {1-alpha:.3f}")
        
        # Check if LP method provides robustness
        if best_coverage > std_coverage:
            logger.info("✓ LP-based method provides improved robustness to distribution shifts")
        else:
            logger.info("✗ Standard method performed better in this experiment")
            
        # Generate concrete numbers for final assessment
        concrete_results = {
            'standard_coverage': round(std_coverage, 4),
            'standard_width': round(std_width, 4),
            'best_lp_coverage': round(best_coverage, 4),
            'best_lp_config': best_config,
            'coverage_improvement': round(best_coverage - std_coverage, 4),
            'target_coverage': 1 - alpha
        }
        
        logger.info("\nCONCRETE RESULTS:")
        for key, value in concrete_results.items():
            logger.info(f"{key}: {value}")
            
        return concrete_results
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    results = main()
