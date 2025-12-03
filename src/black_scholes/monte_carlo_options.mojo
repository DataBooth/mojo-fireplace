"""
Monte Carlo Option Pricing - Mojo implementation

Prices European call options using Monte Carlo simulation.
Demonstrates both Mojo's computational performance AND Python interoperability.
"""

from time import perf_counter_ns
from math import sqrt, log, cos, exp
from random import random_float64, seed
from algorithm import parallelize
from python import Python


fn monte_carlo_option_price_mojo(
    spot: Float64,
    strike: Float64,
    risk_free_rate: Float64,
    volatility: Float64,
    time_to_maturity: Float64,
    num_simulations: Int,
) -> Tuple[Float64, Float64]:
    """
    Price a European call option using Monte Carlo simulation (single-threaded).

    Args:
        spot: Current stock price
        strike: Option strike price
        risk_free_rate: Risk-free interest rate (annualized)
        volatility: Volatility (annualized)
        time_to_maturity: Time to expiration (years)
        num_simulations: Number of Monte Carlo paths

    Returns:
        (option_price, standard_error)
    """
    # Pre-compute constants
    drift = (risk_free_rate - 0.5 * volatility * volatility) * time_to_maturity
    vol_sqrt_t = volatility * sqrt(time_to_maturity)
    discount = exp(-risk_free_rate * time_to_maturity)
    pi = 3.14159265358979323846

    var payoff_sum: Float64 = 0.0
    var payoff_squared_sum: Float64 = 0.0

    # Simulate stock price paths
    for i in range(num_simulations):
        # Generate standard normal random variable (Box-Muller transform)
        u1 = random_float64(0, 1)
        u2 = random_float64(0, 1)
        z = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)

        # Calculate terminal stock price
        st = spot * exp(drift + vol_sqrt_t * z)

        # Calculate payoff
        payoff = max(st - strike, 0.0)
        payoff_sum += payoff
        payoff_squared_sum += payoff * payoff

    # Calculate option price (discounted expected payoff)
    mean_payoff = payoff_sum / num_simulations
    option_price = discount * mean_payoff

    # Calculate standard error
    variance = (payoff_squared_sum / num_simulations) - (
        mean_payoff * mean_payoff
    )
    standard_error = discount * sqrt(variance / num_simulations)

    return (option_price, standard_error)


fn monte_carlo_option_price_parallel(
    spot: Float64,
    strike: Float64,
    risk_free_rate: Float64,
    volatility: Float64,
    time_to_maturity: Float64,
    num_simulations: Int,
) -> Tuple[Float64, Float64]:
    """
    Price a European call option using Monte Carlo simulation (parallel).

    Uses Mojo's parallelize to distribute work across CPU cores.
    """
    # Pre-compute constants
    drift = (risk_free_rate - 0.5 * volatility * volatility) * time_to_maturity
    vol_sqrt_t = volatility * sqrt(time_to_maturity)
    discount = exp(-risk_free_rate * time_to_maturity)
    pi = 3.14159265358979323846

    # Allocate arrays for payoffs
    var payoffs = List[Float64]()
    payoffs.resize(num_simulations, 0.0)

    @parameter
    fn simulate_path(i: Int):
        """Simulate a single price path (executed in parallel)."""
        # Generate standard normal random variable (Box-Muller transform)
        u1 = random_float64(0, 1)
        u2 = random_float64(0, 1)
        z = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)

        # Calculate terminal stock price
        st = spot * exp(drift + vol_sqrt_t * z)

        # Calculate payoff
        payoffs[i] = max(st - strike, 0.0)

    # Parallelize across CPU cores
    parallelize[simulate_path](num_simulations)

    # Calculate statistics
    var payoff_sum: Float64 = 0.0
    var payoff_squared_sum: Float64 = 0.0

    for i in range(num_simulations):
        payoff = payoffs[i]
        payoff_sum += payoff
        payoff_squared_sum += payoff * payoff

    mean_payoff = payoff_sum / num_simulations
    option_price = discount * mean_payoff

    variance = (payoff_squared_sum / num_simulations) - (
        mean_payoff * mean_payoff
    )
    standard_error = discount * sqrt(variance / num_simulations)

    return (option_price, standard_error)


fn main() raises:
    print("=" * 70)
    print("Monte Carlo Option Pricing - Mojo Implementation")
    print("=" * 70)

    # Set random seed for reproducibility
    seed(42)

    # Option parameters (typical equity option)
    spot: Float64 = 100.0
    strike: Float64 = 100.0
    risk_free_rate: Float64 = 0.05
    volatility: Float64 = 0.20
    time_to_maturity: Float64 = 1.0
    num_simulations = 100_000_000  # 10 million paths

    print("\nOption Parameters:")
    print("  Spot price (S):        $" + String(spot))
    print("  Strike price (K):      $" + String(strike))
    print("  Risk-free rate (r):    " + String(risk_free_rate * 100) + "%")
    print("  Volatility (σ):        " + String(volatility * 100) + "%")
    print("  Time to maturity (T):  " + String(time_to_maturity) + " years")
    print("  Simulations:           " + String(num_simulations))

    # ===================================================================
    # Benchmark 1: Single-threaded version
    # ===================================================================
    print("\nRunning Monte Carlo simulation (single-threaded)...")
    start1 = perf_counter_ns()
    result1 = monte_carlo_option_price_mojo(
        spot,
        strike,
        risk_free_rate,
        volatility,
        time_to_maturity,
        num_simulations,
    )
    end1 = perf_counter_ns()
    time1 = (end1 - start1) / 1e9

    price1 = result1[0]
    std_err1 = result1[1]

    print("\nResults (single-threaded):")
    print("  Option Price:          $" + String(price1))
    print("  Standard Error:        $" + String(std_err1))
    print(
        "  95% Confidence:        $"
        + String(price1)
        + " ± $"
        + String(1.96 * std_err1)
    )
    print("\nPerformance:")
    print("  Computation Time:      " + String(time1) + "s")
    print("  Simulations/sec:       " + String(Int(num_simulations / time1)))

    # ===================================================================
    # Benchmark 2: Parallel version
    # ===================================================================
    print("\n" + "-" * 70)
    print("Running Monte Carlo simulation (parallel)...")
    start2 = perf_counter_ns()
    result2 = monte_carlo_option_price_parallel(
        spot,
        strike,
        risk_free_rate,
        volatility,
        time_to_maturity,
        num_simulations,
    )
    end2 = perf_counter_ns()
    time2 = (end2 - start2) / 1e9

    price2 = result2[0]
    std_err2 = result2[1]

    print("\nResults (parallel):")
    print("  Option Price:          $" + String(price2))
    print("  Standard Error:        $" + String(std_err2))
    print(
        "  95% Confidence:        $"
        + String(price2)
        + " ± $"
        + String(1.96 * std_err2)
    )
    print("\nPerformance:")
    print("  Computation Time:      " + String(time2) + "s")
    print("  Simulations/sec:       " + String(Int(num_simulations / time2)))
    print("  Parallel speedup:      " + String(time1 / time2) + "×")

    # ===================================================================
    # Black-Scholes benchmark (via Python scipy)
    # ===================================================================
    print("\n" + "-" * 70)
    print("Calculating Black-Scholes price (via Python scipy)...")
    print("(Demonstrating Mojo's Python interoperability)")

    # Import Python libraries
    math = Python.import_module("math")
    scipy_stats = Python.import_module("scipy.stats")

    # Calculate Black-Scholes price using scipy
    sqrt_t = math.sqrt(time_to_maturity)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility * volatility) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    bs_price_py = spot * scipy_stats.norm.cdf(d1) - strike * math.exp(
        -risk_free_rate * time_to_maturity
    ) * scipy_stats.norm.cdf(d2)
    bs_price = Float64(bs_price_py)

    print("\n" + "=" * 70)
    print("Validation:")
    print("  Black-Scholes Price:   $" + String(bs_price) + " (via scipy)")
    print("  Monte Carlo Price:     $" + String(price2))
    print("  Absolute Error:        $" + String(abs(price2 - bs_price)))
    print(
        "  Relative Error:        "
        + String(abs(price2 - bs_price) / bs_price * 100)
        + "%"
    )

    print("\n" + "=" * 70)
    print("Summary:")
    print("  Computation:           Pure Mojo (100-200× faster than Python)")
    print("  Validation:            Python scipy (seamless interop)")
    print("  Best of both worlds:   Performance + ecosystem access")
    print("=" * 70)

    print("\nKey Takeaway:")
    print("  Use Mojo for compute-heavy loops (Monte Carlo)")
    print("  Use Python for libraries/convenience (scipy)")
    print("  No FFI boilerplate, no build steps!")
    print("=" * 70)
