import warrants

# Parameters for the warrant
s0 = 100.0
strike_discount = 0.9
buyback_price = 15.0
t = 1.0
r = 0.05
sigma = 0.2
n_paths = 10000
n_steps = 252
poly_degree = 4
seed = 42

# Price the warrant
try:
    price = warrants.price_exotic_warrant(
        s0,
        strike_discount,
        buyback_price,
        t,
        r,
        sigma,
        n_paths,
        n_steps,
        poly_degree,
        seed,
    )
    print(f"The price of the exotic warrant is: {price}")

    # Test with high buyback price (should be higher than low buyback)
    price_high_buyback = warrants.price_exotic_warrant(
        s0,
        strike_discount,
        1000.0, # High buyback, should not be exercised
        t,
        r,
        sigma,
        n_paths,
        n_steps,
        poly_degree,
        seed,
    )
    print(f"Price with high buyback price: {price_high_buyback}")
    assert price < price_high_buyback

    print("\nIntegration test successful!")

except Exception as e:
    print(f"An error occurred: {e}")