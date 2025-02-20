import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance import shared
# Import the new refactored classes.
# (Assuming your new refactored code is in a module called "dcc_garch")
from functions import MarketData, DCCModel, Portfolio, RiskMetrics

# Helper: clear session state
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

rate_limited = False
# Helper: simple ticker validity check using yfinance
def is_ticker_valid(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        
        return not data.empty
    
    except Exception as e:
        if "Too Many Requests" in str(e) or "Rate limited" in str(e):            
            rate_limited = True
            st.warning(f"Error fetching data for {ticker}: {e}")
            

        return False

# Example inputs
example_tickers = "AAPL,JPM,A,DIS,HSBC,MSFT" 
example_weights = "0.2,0.2,0.2,0.1,0.1,0.2"

st.title("DCC-GARCH Tool")

# Create tabs
tabs = st.tabs(["About", "Inputs", "Historical Analysis", "Present Day Analysis"])

with tabs[0]:
    st.write("##### About")
    st.write(
        """This tool implements the Dynamic Conditional Correlation GARCH (DCC-GARCH) model to analyze a portfolio.
        
Full code and methodology explanation can be found in the repository:
https://github.com/danielmw2001/DCC-GARCH/tree/main

DCC-GARCH improves on univariate and non-dynamic GARCH models by modeling the time-varying covariance within portfolios under different market conditions."""
    )
    st.write("##### Input Tab")
    st.write("Input tickers and respective weights (separated by commas). Then move to the Historical Analysis tab.")
    st.write("##### Historical Analysis Tab")
    st.write("Displays portfolio correlation matrices and volatility for periods of high, medium, and low volatility over the last 5 years.")
    st.write("##### Present-day Analysis Tab")
    st.write("Shows current day correlation, recommended portfolio split to minimize variance, as well as VaR and CVaR from Monte Carlo simulations.")

with tabs[1]:
    # Option to autofill with example inputs
    if st.button("Autofill with example"):
        tickers_input = example_tickers
        weights_input = example_weights
        st.success("Example autofilled. Stocks: AAPL,JPM,A,DIS,HSBC,MSFT. Weights: 0.2,0.2,0.2,0.1,0.1,0.2")
        # Allow the user to clear and enter their own values
        if st.button("Add my own"):
            clear_session_state()
            st.experimental_rerun()
    else:
        tickers_input = st.text_input("Enter tickers separated by commas (e.g., AAPL,GOOGL,MSFT)")
        weights_input = st.text_input("Enter corresponding weights separated by commas (e.g., 0.5,0.5; must sum to 1)")
    
    # Process inputs
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    try:
        weights = [float(w.strip()) for w in weights_input.split(',') if w.strip()]
    except ValueError:
        weights = []
    
    if tickers and weights:
        if len(tickers) != len(weights):
            st.warning("Number of tickers and weights must match!")
        else:
            valid_tickers = all(is_ticker_valid(t) for t in tickers)
            total_weight = sum(weights)
            if not valid_tickers:
                st.warning("One or more tickers are invalid.")
                if rate_limited:
                    st.warning('Too Many Requests. Rate limited. Try after a while.')
            elif abs(total_weight - 1.0) > 1e-10:
                st.warning(f"Total weight is {total_weight}. Weights must sum to 1.")
            else:
                st.success("Inputs are valid!")
                # --- RUN THE ANALYSIS USING THE NEW CLASSES ---
                # 1. Get historical price data and returns
                market_data = MarketData(tickers, period='5y')
                # 2. Run the DCC estimation
                dcc_model = DCCModel(tickers, market_data.returns)
                dcc_results = dcc_model.run_dcc()
                Hts = dcc_results["Hts"]
                Rts = dcc_results["Rts"]
                # 3. Compute predicted portfolio volatilities over time
                predicted_vols = [Portfolio.portfolio_volatility(H, np.array(weights)) for H in Hts]
                
                # 4. (Optional) Compute historical averages over selected windows.
                # Re-using the logic from your previous hml() function:
                def hml(all_vols, Hts, Rts):
                    rolling_avg = np.array(pd.Series(all_vols).rolling(window=7).mean().dropna())
                    highest_t = int(rolling_avg.argmax()) - 4
                    lowest_t = int(rolling_avg.argmin()) - 4
                    median_t = int((np.abs(rolling_avg - np.median(rolling_avg))).argmin()) - 4
                    avg_corrs = []
                    avg_covs = []
                    avg_vols = []
                    valid_indices = [highest_t, median_t, lowest_t]
                    for i in valid_indices:
                        if i < 6 or i + 1 > len(Hts):
                            avg_corrs.append(None)
                            avg_covs.append(None)
                            avg_vols.append(None)
                        else:
                            avg_cov = np.mean(Hts[i - 3 : i + 4], axis=0)
                            avg_corr = np.mean(Rts[i - 4 : i + 4], axis=0)
                            avg_vol = np.mean(all_vols[i - 3 : i + 4])
                            avg_corrs.append(avg_corr)
                            avg_covs.append(avg_cov)
                            avg_vols.append(avg_vol)
                    return avg_corrs, avg_covs, avg_vols
                
                avg_corrs, avg_covs, avg_vols = hml(predicted_vols, Hts, Rts)
                
                # 5. Compute variance-minimizing portfolio weights from the current covariance matrix
                w_min = Portfolio.optimize_weights(Hts[-1])
                
                # Store results in session state
                st.session_state.results = {
                    'Hts': Hts,
                    'predicted_vols': predicted_vols,
                    'Rts': Rts,
                    'avg_corrs': avg_corrs,
                    'avg_covs': avg_covs,
                    'avg_vols': avg_vols,
                    'w_min': w_min,
                    'tickers': tickers,
                    'weights': weights
                }

# Helper display functions
def display_single_asset_metrics(period_name, volatility):
    st.write(f"#### {period_name}")
    st.write(f"Daily Volatility: {volatility * 100:.3f}%")

def display_multi_asset_metrics(period_name, corr_matrix, cov_matrix, portfolio_vol, tickers):
    st.write(f"#### {period_name}")
    corr_df = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
    cov_df = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.write("##### Correlation Matrix")
        st.dataframe(corr_df.style.format("{:.2f}").background_gradient(cmap="coolwarm"), use_container_width=True)
    with col2:
        st.write("##### Variance-Covariance Matrix")
        st.dataframe(cov_df.style.format("{:.5f}").background_gradient(cmap="coolwarm"), use_container_width=True)
    with col3:
        st.write("##### Portfolio Volatility")
        st.write(f"Daily portfolio Volatility: {portfolio_vol * 100:.3f}%")
        st.write(f"Annual portfolio Volatility: {portfolio_vol * 100 * np.sqrt(252):.3f}%")

with tabs[2]:
    if 'results' in st.session_state:
        results = st.session_state.results
        periods = ["High Market Volatility", "Medium Market Volatility", "Low Market Volatility"]
        if len(results['tickers']) == 1:
            st.warning("Covariance and correlations unavailable when only one asset is selected.")
            for period in periods:
                # For single asset, display volatility only (taking square root of variance)
                display_single_asset_metrics(period, np.sqrt(results['avg_covs'][0]) if results['avg_covs'][0] is not None else 0)
        else:
            st.write("###### If matrices display badly, hover over and expand")
            for i, period in enumerate(periods):
                if results['avg_corrs'][i] is None:
                    st.warning(f"Not enough data to compute {period} metrics.")
                else:
                    display_multi_asset_metrics(
                        period,
                        results['avg_corrs'][i],
                        results['avg_covs'][i],
                        results['avg_vols'][i],
                        results['tickers']
                    )
    else:
        st.error("Please enter valid inputs in the Inputs tab.")

with tabs[3]:
    if 'results' in st.session_state:
        results = st.session_state.results
        st.write("### Present Day Analysis")
        if len(results['tickers']) == 1:
            st.warning("Covariance and correlations unavailable when only one asset is selected.")
            display_single_asset_metrics("Current Market Conditions", np.sqrt(np.mean(np.diag(results['Hts'][-1]))))
        else:
            current_corr_matrix = results['Rts'][-1]
            current_cov_matrix = results['Hts'][-1]
            current_volatility = results['predicted_vols'][-1]
            col5, col6 = st.columns(2, gap="large")
            with col5:
                st.write("#### Current Variance-Minimizing Portfolio Weights")
                weights_df = pd.DataFrame(results['w_min'], columns=['Optimal Weighting'], index=results['tickers'])
                st.dataframe(weights_df.style.format("{:.3f}"), use_container_width=True)
            with col6:
                st.write("#### VaR and CVaR")
                # Use RiskMetrics.both_vars from the refactored code
                for days, period in zip([1, 5, 252], ['Daily', 'Weekly', 'Annual']):
                    var, cvar = RiskMetrics.both_vars(results['predicted_vols'][-1], days)
                    st.write(f"{period} VaR: {var * 100:.3f}%")
                    st.write(f"{period} CVaR: {cvar * 100:.3f}%")
            st.write("#### Current Portfolio Metrics")
            display_multi_asset_metrics("Current Market Conditions", current_corr_matrix, current_cov_matrix, current_volatility, results['tickers'])
    else:
        st.error("Please enter valid inputs in the Inputs tab.")
