# DCC-GARCH Model Methodology

## 1. Univariate GARCH Model
We use a starndard GARCH(1,1) model. Intuitively, GARCH models volatility as a combination of last time periods volatility and shocks (volatility unmodelled by last periods equation), assuming residuals are normally distributed around zero. It is defined as follows.
Returns are described as:

r_{i,t} = μ_i + ε_{i,t}, ε_{i,t} = σ_{i,t} z_{i,t}

where:
- r_{i,t}: Return of asset i at time t
- μ_i: Mean return of asset i
- ε_{i,t}: Residual return
- σ_{i,t}: Conditional volatility
- z_{i,t} ~ N(0,1): Standardized residual

The GARCH(1,1) model for conditional volatility is:

σ_{i,t}^2 = ω_i + α_i ε_{i,t-1}^2 + β_i σ_{i,t-1}^2

where:
- ω_i, α_i, β_i: Parameters satisfying ω_i > 0, α_i ≥ 0, β_i ≥ 0, and α_i + β_i < 1

## 2. Dynamic Conditional Correlation
This model is an improves upon a standard garch, or simpler multivariate models by not making the assumption correlation within a portfolio is static over time. Instead, the model produces a time-varying matrix R_t, allowing for changing relationships between assets, which is particularly helpful in pikcing out how portfolios behave in periods of varying volatility.

The time-varying correlation matrix R_t is:

R_t = diag(Q_t)^{-1} Q_t diag(Q_t)^{-1}

where:
- Q_t: Time-varying covariance matrix of standardized residuals
- diag(Q_t): Diagonal matrix of the square roots of the diagonal elements of Q_t

The matrix Q_t is modelled as:

Q_t = (1 - α - β)Q + α z_{t-1} z_{t-1}^⊤ + β Q_{t-1}

where:
- Q: Time-invariant quasi-correlation matrix of the standardized residuals, calculated as the average of all Q_t 
- z_{t-1} = (ε_{1,t-1}/σ_{1,t-1}, ..., ε_{N,t-1}/σ_{N,t-1})^⊤: Vector of standardized residuals, both taken from our GARCH(1,1) model 
- α, β: Non-negative parameters to be found satisfying α + β < 1

## 3. Conditional Covariance Matrix
The full conditional covariance matrix of asset returns is:

H_t = D_t R_t D_t

where:
- D_t = diag(σ_{1,t}, σ_{2,t}, ..., σ_{N,t}): Diagonal matrix of individual asset volatilities

## 4. Solving for α and β
This is solved using scipy.optimise to find perameters that minimise the models log-liklehood function: 
l_t = Σ_{t=1}^T -0.5 * (log|R_t| + z_t^⊤ R_t^{-1} z_t)

## 5. Backtesting
Backtesting methodology was kept simple, and is something being worked on currently. Portfolio volatility at t+1 is computed using:
w_t * H_{t+1} * w_t' where H_{t+1} is the predicted variance - covariance matrix and w_t the weighting of the assets. The 5% VaR

