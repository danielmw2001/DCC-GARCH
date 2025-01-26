from functions import *
import streamlit as st
import pandas as pd
import numpy as np

example_tickers = "AAPL,NVDA,GOOGL,AMZN" 
example_weights = "0.4,0.3,0.2,0.1"

st.title("DCC-GARCH Portfolio Input")

tabs = st.tabs(["About", "Inputs", "Historical Analysis","Present Day Analysis"])

with tabs[0]:
    st.header('About this app')
    st.write('I coded up the Dynamic-Conditional-Correlation GARCH model as a pet project and played around with what you could do with it. Full methodology and code available here: . To use, enter your tickers separated by commas and respective weights in your portfolio on the inputs page, and then switch to the outputs page to view your portfolio\'s correlation and variance-covariance matrices in times of high, low and medium market stress. Optimal portfolio weightings for the current market are also displayed, as well as VaR and CVaR found via Monte-Carlo simulations')

with tabs[1]:
    if st.button("Autofill with example"):
        tickers_input = example_tickers
        weights_input = example_weights
        st.success("Example autofilled. Check the Output tab. Stocks: AAPL,JPM,GOOGL,AMZN. Weights:0.4,0.3,0.2,0.1")
        if st.button("Add my own"):
            st.rerun()
    
    else:
        tickers_input = st.text_input("Enter tickers separated by commas (e.g., AAPL,GOOGL,MSFT)")
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
    
with tabs[2]:
    if tickers and weights and len(tickers) == len(weights) and abs(sum(weights) - 1.0) < 1e-10:
        Hts, predicted_vols, Rts = whole_thing(tickers, np.array(weights))
        avg_corrs, avg_covs, avg_vols = hml(predicted_vols, Hts, Rts) 
        w_min = optimize_portfolio_variance(Hts[-1])

        periods = ["High Market Volatility", "Medium Market Volatility", "Low Market Volatility"]

        if len(tickers) == 1:
            st.warning("Covariance and correlations unavailable, only 1 asset selected.")
            for i, period in enumerate(periods):
                st.write(f"#### {period} daily volatility: {float(np.sqrt(avg_covs[i])*100):.3f}%")

        else:
            for i, period in enumerate(periods):
                st.write(f"#### {period}")
                
                corr_df = pd.DataFrame(
                    np.squeeze(avg_corrs[i]),
                    index=tickers,
                    columns=tickers
                )
                
                cov_df = pd.DataFrame(
                    np.squeeze(avg_covs[i]),
                    index=tickers,
                    columns=tickers
                )
                
                col1, col2, col3 = st.columns(3, gap="large")
                
                with col1:
                    st.write(f"##### Correlation Matrix")
                    st.dataframe(
                        corr_df.style
                            .format("{:.2f}")
                            .background_gradient(cmap="coolwarm"),
                            use_container_width=True
                    )
                
                with col2:
                    st.write(f"##### Variance Covariance Matrix") 
                    st.dataframe(
                        cov_df.style
                            .format("{:.5f}") 
                            .background_gradient(cmap="coolwarm"),
                                use_container_width=True
                    )
                with col3:
                    st.write(f"##### Portfolio Volatility")
                    st.write(f"Portfolio Volatility: {float(avg_vols[i]) * 100:.3f}%")
    else:
                st.error("Weights must sum to 1, be non-zero for all assets, and all tickers must be valid before running the analysis.")
with tabs[3]:
    if tickers and weights and len(tickers) == len(weights) and abs(sum(weights) - 1.0) < 1e-10:
        st.write(f'##### Current Variance Minimising Portfolio Weights:') 
        st.write(pd.DataFrame(w_min,columns=['Weight'],index=tickers))
        st.write(f'##### Current Variance Minimising Portfolio Weights:')
        corr_df = pd.DataFrame(
                    np.squeeze(avg_corrs[-1]),
                    index=tickers,
                    columns=tickers
                )
                
        cov_df = pd.DataFrame(
                    np.squeeze(avg_covs[-1]),
                    index=tickers,
                    columns=tickers
                )
                
        col1, col2, col3 = st.columns(3, gap="large")
                
        with col1:
            st.write(f"##### Correlation Matrix")
            st.dataframe(
                corr_df.style
                    .format("{:.2f}")
                    .background_gradient(cmap="coolwarm"),
                    use_container_width=True
            )
        
        with col2:
            st.write(f"##### Variance Covariance Matrix") 
            st.dataframe(
                cov_df.style
                    .format("{:.5f}") 
                    .background_gradient(cmap="coolwarm"),
                        use_container_width=True
            )
        with col3:
            st.write(f"##### Portfolio Volatility")
            st.write(f"Portfolio Volatility: {float(avg_vols[-1]) * 100:.3f}%")
        
    else:
        st.error("Weights must sum to 1, be non-zero for all assets, and all tickers must be valid before running the analysis.")
