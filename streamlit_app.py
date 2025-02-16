from functions import *
import streamlit as st
import pandas as pd
import numpy as np

def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]
        
example_tickers = "AAPL,JPM,A,DIS,HSBC,MSFT" 
example_weights = "0.2,0.2,0.2,0.1,0.1,0.2"
st.title("DCC-GARCH Tool")
st.write('First run will take a minute as the server starts up')
predicted_vols = []
tabs = st.tabs(["About", "Inputs", "Historical Analysis", "Present Day Analysis"])

with tabs[0]:
    st.write("##### About")
    st.write('''This script implements the Dynamic Conditional Correlation GARCH (DCC-GARCH) to analyse an uploaded portfolio. Full code and methodology explanation found in the repo here: https://github.com/danielmw2001/DCC-GARCH/tree/main 
             \n DCC-GARCH improves on the univariate and non-dynamic GARCH models modelling the covariance within portfolios, and how that covariance dynamically changes depending on markey conditions.''')
    st.write("##### Input Tab")
    st.write('Input tickers and respective weights, seperated by commas, let it run then move to the historical analysis tab')
    st.write("##### Historical Analysis Tab")
    st.write('Displays the portfolios correlation matricies and volatility in the highest (typically covid), lowest and median periods of volatility in the last 5 years.')
    st.write("##### Present-day Analysis Tab")
    st.write('Shows the current day correlation, reccomended portfolio split to minimise variance, as well as VaR and CVaR from monte-carlo simulations')

with tabs[1]:
    if st.button("Autofill with example"):
        tickers_input = example_tickers
        weights_input = example_weights
        st.success("Example autofilled. Check the Output tab. Stocks: AAPL,JPM,A,DIS,HSBC,MSFT. Weights:0.2,0.2,0.2,0.1,0.1,0.2")
        if st.button("Add my own"):
            clear_session_state()
            st.rerun()
    else:
        tickers_input = st.text_input("Enter tickers separated by commas withoug spaces (e.g., AAPL,GOOGL,MSFT)")
        weights_input = st.text_input("Enter corresponding weights separated by commas (e.g 0.5,0.5. Must sum to 1)")
    
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
                st.warning('One or more tickers are invalid')
            elif abs(total_weight - 1.0) > 1e-10:
                st.warning(f"Total weight is {total_weight}. Weights must sum to 1.")
            else:
                st.success("Inputs are valid!")
                # Run analysis immediately when inputs are valid
                Hts, predicted_vols, Rts = whole_thing(tickers, np.array(weights))
                avg_corrs, avg_covs, avg_vols = hml(predicted_vols, Hts, Rts)
                w_min = optimize_portfolio_variance(Hts[-1])
                
                # Store results temporarily in session state
                st.session_state.results = {'Hts': Hts,'predicted_vols': predicted_vols, 'Rts': Rts,'avg_corrs': avg_corrs,'avg_covs': avg_covs,'avg_vols': avg_vols,'w_min': w_min,'tickers': tickers, 'weights': weights}

def display_single_asset_metrics(period_name, volatility):
    st.write(f"#### {period_name}")
    st.write(f"Daily Volatility: {float(volatility * 100):.3f}%")

def display_multi_asset_metrics(period_name, corr_matrix, cov_matrix, portfolio_vol, tickers):
    st.write(f"#### {period_name}")
    corr_df = pd.DataFrame(np.squeeze(corr_matrix),
        index=tickers,
        columns=tickers)
    
    cov_df = pd.DataFrame(
        np.squeeze(cov_matrix),
        index=tickers, columns=tickers)
    
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.write("##### Correlation Matrix")
        st.dataframe(corr_df.style.format("{:.2f}").background_gradient(cmap="coolwarm"),use_container_width=True)
    
    with col2:
        st.write("##### Variance Covariance Matrix") 
        st.dataframe(cov_df.style.format("{:.5f}").background_gradient(cmap="coolwarm"),use_container_width=True)
    with col3:
        st.write("##### Portfolio Volatility")
        st.write(f"Daily portfolio Volatility: {float(portfolio_vol) * 100:.3f}%")
        st.write(f"Annual portfolio Volatility: {float(portfolio_vol) * 100 * np.sqrt(252):.3f}%")
with tabs[2]:
    if 'results' in st.session_state:
        results = st.session_state.results
        periods = ["High Market Volatility", "Medium Market Volatility", "Low Market Volatility"]

        if len(results['tickers']) == 1:
            st.warning("Covariance and correlations unavailable, only 1 asset selected.")
            for i, period in enumerate(periods):
                display_single_asset_metrics(period, np.sqrt(results['avg_covs'][i]))
        else:
                
            st.write(f'###### If matricies display badly, hover over and expand')
            for i, period in enumerate(periods):
                display_multi_asset_metrics(period, results['avg_corrs'][i], results['avg_covs'][i], results['avg_vols'][i], 
                                            results['tickers'])
    else:
        st.error("Please enter valid inputs in the Inputs tab.")

with tabs[3]:
    if 'results' in st.session_state:
        results = st.session_state.results

        st.write("### Present Day Analysis")
        
        if len(results['tickers']) == 1:
            st.warning("Covariance and correlations unavailable, only 1 asset selected.")
            display_single_asset_metrics("Current Market Conditions", 
                np.sqrt(results['Hts'][-1].diagonal().mean()))
        else:
            # Extract current day's values
            current_corr_matrix = results['Rts'][-1]  # Current correlation matrix
            current_cov_matrix = results['Hts'][-1]  # Current covariance matrix
            current_volatility = results['predicted_vols'][-1]  # Current portfolio volatility

            # Display current variance-minimizing weights
            col5, col6 = st.columns(2, gap="large")
            
            with col5:
                st.write("#### Current Variance-Minimizing Portfolio Weights")
                st.dataframe(pd.DataFrame(
                        results['w_min'], 
                        columns=['Optimal weighting'], 
                        index=results['tickers']).style.format("{:.3f}"),
                    use_container_width=True)
            
            with col6:
                st.write(f'#### Vars and CVaRs')
                if predicted_vols:
                    days =[1,5,252]
                    periods = ['Daily','Weekly','Annual']
                    for days, period in zip(days,periods):
                        var, cvar = both_vars(predicted_vols[-1], days)
                        st.write(period,'VaR', f'{float(var) * 100:.3f}%')
                        st.write(period,'CVaR', f'{float(cvar) * 100:.3f}%')
                         
                    # st.write(f"##### Daily VaR: {float(var(predicted_vols[-1],1)) * 100:.3f}%")
                    # st.write(f"##### Weekly VaR: {float(var(predicted_vols[-1],5)) * 100:.3f}%")
                    # st.write(f"##### Annual  VaR: {float(var(predicted_vols[-1],252)) * 100:.3f}%")
                    # st.write(f"##### Daily CVaR: {float(cvar(predicted_vols[-1],1)) * 100:.3f}%")
                    # st.write(f"##### Weekly CVaR: {float(cvar(predicted_vols[-1],5)) * 100:.3f}%")
                    # st.write(f"##### Annual CVaR: {float(cvar(predicted_vols[-1],252)) * 100:.3f}%")


            # Display current portfolio metrics
            st.write("#### Current Portfolio Metrics")
            display_multi_asset_metrics("Current Market Conditions", current_corr_matrix, current_cov_matrix, current_volatility, results['tickers'])
    else:
        st.error("Please enter valid inputs in the Inputs tab.")
