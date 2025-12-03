"""
Monte Carlo Option Pricing - Pure Python implementation

Prices European call options using Monte Carlo simulation.
Computation-heavy problem ideal for demonstrating Mojo's advantages.

Common use cases:
- Financial derivatives pricing
- Risk analysis (VaR, CVaR)
- Portfolio optimization under uncertainty
- Regulatory stress testing
"""

import time
import random
import math


def monte_carlo_option_price_python(
    spot: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_maturity: float,
    num_simulations: int
) -> tuple[float, float]:
    """
    Price a European call option using Monte Carlo simulation.
    
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
    random.seed(42)
    
    # Pre-compute constants
    drift = (risk_free_rate - 0.5 * volatility * volatility) * time_to_maturity
    vol_sqrt_t = volatility * math.sqrt(time_to_maturity)
    discount = math.exp(-risk_free_rate * time_to_maturity)
    
    payoff_sum = 0.0
    payoff_squared_sum = 0.0
    
    # Simulate stock price paths
    for _ in range(num_simulations):
        # Generate standard normal random variable (Box-Muller transform)
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        
        # Calculate terminal stock price
        st = spot * math.exp(drift + vol_sqrt_t * z)
        
        # Calculate payoff
        payoff = max(st - strike, 0.0)
        payoff_sum += payoff
        payoff_squared_sum += payoff * payoff
    
    # Calculate option price (discounted expected payoff)
    mean_payoff = payoff_sum / num_simulations
    option_price = discount * mean_payoff
    
    # Calculate standard error
    variance = (payoff_squared_sum / num_simulations) - (mean_payoff * mean_payoff)
    standard_error = discount * math.sqrt(variance / num_simulations)
    
    return option_price, standard_error


def main():
    print("=" * 70)
    print("Monte Carlo Option Pricing - Pure Python Implementation")
    print("=" * 70)
    
    # Option parameters (typical equity option)
    spot = 100.0            # Current stock price
    strike = 100.0          # At-the-money option
    risk_free_rate = 0.05   # 5% annual risk-free rate
    volatility = 0.20       # 20% annual volatility
    time_to_maturity = 1.0  # 1 year to expiration
    num_simulations = 10_000_000  # 10 million paths
    
    print("\nOption Parameters:")
    print(f"  Spot price (S):        ${spot:.2f}")
    print(f"  Strike price (K):      ${strike:.2f}")
    print(f"  Risk-free rate (r):    {risk_free_rate*100:.1f}%")
    print(f"  Volatility (σ):        {volatility*100:.1f}%")
    print(f"  Time to maturity (T):  {time_to_maturity:.1f} years")
    print(f"  Simulations:           {num_simulations:,}")
    
    print("\nRunning Monte Carlo simulation...")
    start = time.time()
    price, std_err = monte_carlo_option_price_python(
        spot, strike, risk_free_rate, volatility, time_to_maturity, num_simulations
    )
    end = time.time()
    
    elapsed = end - start
    
    print(f"\n{'Results:':<25}")
    print(f"{'  Option Price:':<25} ${price:.4f}")
    print(f"{'  Standard Error:':<25} ${std_err:.4f}")
    print(f"{'  95% Confidence:':<25} ${price:.4f} ± ${1.96*std_err:.4f}")
    print(f"\n{'Performance:':<25}")
    print(f"{'  Computation Time:':<25} {elapsed:.3f}s")
    print(f"{'  Simulations/sec:':<25} {num_simulations/elapsed:,.0f}")
    
    print("\n" + "=" * 70)
    print("Bottleneck: Python interpreter overhead on math-heavy loops")
    print("Each path requires: exp, sqrt, log, cos, max operations")
    print("Potential speedup with Mojo: 50-200×")
    print("=" * 70)
    
    # Black-Scholes analytical price for comparison
    from scipy.stats import norm
    
    d1 = (math.log(spot/strike) + (risk_free_rate + 0.5*volatility**2)*time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
    d2 = d1 - volatility * math.sqrt(time_to_maturity)
    bs_price = spot * norm.cdf(d1) - strike * math.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2)
    
    print(f"\nBlack-Scholes Price: ${bs_price:.4f}")
    print(f"Monte Carlo Error:   ${abs(price - bs_price):.4f} ({abs(price-bs_price)/bs_price*100:.2f}%)")


if __name__ == "__main__":
    main()
