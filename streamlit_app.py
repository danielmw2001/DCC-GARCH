from functions import *  # Import your custom functions
import streamlit as st

# Set a title
st.title("DCC-GARCH Portfolio Input")

tabs = st.tabs(["Inputs","Outputs"])
# Enter the number of assets
with tabs[0]:
    n = st.number_input("Enter the number of assets in the portfolio", min_value=1, step=1, value=1)

    # Initialize session state for tickers and weights
    if "tickers" not in st.session_state:
        st.session_state.tickers = [""] * n
    if "weights" not in st.session_state:
        st.session_state.weights = [0.0] * n

    # Adjust the length of the tickers and weights lists if `n` changes
    if len(st.session_state.tickers) != n:
        st.session_state.tickers = st.session_state.tickers[:n] + [""] * max(0, n - len(st.session_state.tickers))

    if len(st.session_state.weights) != n:
        st.session_state.weights = st.session_state.weights[:n] + [0.0] * max(0, n - len(st.session_state.weights))

    # Input fields for tickers and weights
    st.write("Enter the ticker symbols and their weights:")
    for i in range(n):
        st.session_state.tickers[i] = st.text_input(f"Ticker {i + 1}", value=st.session_state.tickers[i]).upper()
        st.session_state.weights[i] = st.number_input(
            f"Weight for {st.session_state.tickers[i] or f'Ticker {i + 1}'}",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.weights[i],
            step=0.01
        )

    # Display the entered tickers and weights
    portfolio_data = [{"Ticker": st.session_state.tickers[i], "Weight": st.session_state.weights[i]} for i in range(n)]

    # Ensure weights sum to 1
    total_weight = sum(st.session_state.weights)
    valid = True
    for t in st.session_state.tickers:
        if not is_ticker_valid(t):
            valid = False
            st.warning('One or more tickers invalid')
            continue
        
    if total_weight != 1.0:
        st.warning(f"Total weight is {total_weight}. Make sure weights sum to 1.")
    elif not all(w > 0 for w in st.session_state.weights):
        st.warning("All asset weights must add to 1")


with tabs[1]:
    if total_weight == 1.0 and all(w > 0 for w in st.session_state.weights) and valid == True:
        # Perform analysis
        Hts, predicted_vols, Rts = whole_thing(st.session_state.tickers, np.array(st.session_state.weights))
        avg_corrs, avg_covs, avg_vols = hml(predicted_vols, Hts, Rts)
        w_min = optimize_portfolio_variance(Hts[-1])


        # Set column and row names for correlation and covariance matrices
        tickers = st.session_state.tickers
        periods = ["High Market Volatility", "Medium Market Volatility", "Low Market Volatility"]

        # Loop through the results and display them with proper labels
        if len(tickers) == 1:
            st.warning("Covariance and corrolations unavailable, only 1 asset selected.")
            for i, period in enumerate(periods):
                st.write(f"#### {period} daily volatility: {float(np.sqrt(avg_covs[i])*100):.3f}%")

        else:
            for i, period in enumerate(periods):
                st.write(f"#### {period}")

                # Prepare correlation DataFrame
                corr_df = pd.DataFrame(
                    np.squeeze(avg_corrs[i]),
                    index=tickers,
                    columns=tickers
                )

                # Prepare covariance DataFrame
                cov_df = pd.DataFrame(
                    np.squeeze(avg_covs[i]),
                    index=tickers,
                    columns=tickers
                )

                # Create side-by-side columns
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

        st.write(f'##### Current Variance Minimising Portfolio Weights:')
        st.write(pd.DataFrame(w_min,columns=['Weight'],index=tickers))
                

    else:
        st.error("Weights must sum to 1 and be non-zero for all assets, and all tickers must be valid before running the analysis.")
