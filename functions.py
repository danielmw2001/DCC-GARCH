#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 18:04:06 2025

@author: danmw
"""
import gc
from arch import arch_model
from scipy.optimize import minimize
import streamlit as slt
import numpy as np
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.linalg import sqrtm, inv
from datetime import datetime, timedelta
import pandas as pd
np.random.seed(42)

class Data:
    def __init__(self, tickers, period):
        data = yf.Tickers(tickers)
        prices = data.history(period=period)["Close"]
        
        returns = np.log(prices / prices.shift(1)).dropna()
        self.prices = prices
        self.returns = returns
        

class DCC:
    def __init__(self, tickers, returns):
        self.tickers = tickers
        self.returns = returns


    def standardising_residuals(self):
        all_sigma = []
        all_standardised_resid = []
        for t in self.tickers:
            garch_model = arch_model(self.returns[t], vol='Garch', p=1, q=1, mean='Zero')
            uni_var = garch_model.fit(disp="off")
            sigma_t = np.array(uni_var.conditional_volatility)
            standardise_resid = uni_var.resid / sigma_t
            all_standardised_resid.append(standardise_resid)
            all_sigma.append(sigma_t)
        
        all_var = np.array(np.column_stack(all_sigma))
        all_standardised_resids = np.array(np.column_stack(all_standardised_resid))
        return all_var, all_standardised_resids

    def find_Qbar(self, all_standardised_resids):
        Qt = sum(np.outer(resid, resid) for resid in all_standardised_resids)
        Qbar = Qt / len(all_standardised_resids)
        return Qbar

    def find_Qt(self, all_standardised_resids, Qbar, alpha, beta):
        Q_matrices = [Qbar]
        Qt = Qbar
        for t in range(1, len(all_standardised_resids)):
            eps_lag = all_standardised_resids[t-1]  
            Qt = Qbar \
                 + alpha * (np.outer(eps_lag, eps_lag) - Qbar) \
                 + beta * (Qt - Qbar)
            Q_matrices.append(Qt)
        return Q_matrices


    def log_likelihood(self, params, Qbar, standardized_residuals):
        alpha, beta = params
        T = len(standardized_residuals)
        Q_t = Qbar
        log_likelihood = 0
    
        for t in range(T):
            if t > 0:
                Q_t = Qbar + alpha * (np.outer(standardized_residuals[t-1], standardized_residuals[t-1]) - Qbar) + beta * (Q_t - Qbar)
    
            D_t = np.sqrt(np.diag(Q_t))
            D_inv = np.diag(1 / D_t)
            R_t = np.copy(D_inv @ Q_t @ D_inv)
            #print(R_t,'RT',D_t,'Dt')
            inv_Rt = np.linalg.inv(R_t)    
            
            det_Rt = np.linalg.det(R_t)
            resid_t = standardized_residuals[t]
            log_likelihood += -0.5 * (np.log(det_Rt) + resid_t @ inv_Rt @ resid_t)
    
        return -log_likelihood  

    def estimate_alpha_beta(self, Qbar, standardized_residuals):
        initial_params = [0.05, 0.9] 
        bounds = [(0, 1), (0, 1)]  
        constraints = [{'type': 'ineq', 'fun': lambda x: 0.99 - sum(x)},  
                    {'type': 'ineq', 'fun': lambda x: x[0]},       
                    {'type': 'ineq', 'fun': lambda x: x[1]}]       

        result = minimize(
            self.log_likelihood,
            initial_params,
            args=(Qbar, standardized_residuals),
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={'ftol': 1e-12}  

        )

        return result.x  
    
    def estimate_Ht(self, alpha, beta, all_var, Q_matrices):
        Dt = np.diag(all_var[-1])
        D_q = sqrtm(np.diag(np.diag(Q_matrices[-1])))
        D_q_inv = inv(D_q)
        Rt = D_q_inv  @ Q_matrices[-1] @ D_q_inv 
        Ht = Dt @ Rt @ Dt
        return Ht
    
    def estimate_Hts(self, alpha, beta, all_var, Q_matrices):
        Hts = []
        Rts = []
        for t in range(len(all_var)-1):
            Dt = np.diag(all_var[t])
            A = sqrtm(np.diag(np.diag(Q_matrices[t])))   # A is diagonal of sqrt of diag(Q)
            D_q = np.linalg.inv(A)                                 # So D_q is 1/sqrt
            Rt = D_q @ Q_matrices[t] @ D_q               # Now that is the correlation
            Ht = Dt @ Rt @ Dt
            Rts.append(Rt)
            Hts.append(Ht)
        return np.array(Hts), np.array(Rts)
    

def portfolio_vol(S,w):
    sd = np.sqrt(w.T @ S @ w)
    return sd

def true_vol(tickers, start,end):
    data = yf.Tickers(tickers)
    prices = data.history(start = start, end = end , interval='5m',progress=False)["Close"]
    returns = np.log(prices / prices.shift(1)).dropna()
    x = np.zeros((3,3))
    for i in range(len(returns)):
        x += np.outer(np.array(returns)[i],np.array(returns)[i])
        
    return(x)

def true_vol_new(tickers):
    data = yf.Tickers(tickers)
    prices = data.history(period='50d', interval='30m', progress=False)['Close']

    daily_groups = prices.groupby(prices.index.date)

    cov_series = []
    for day, df_day in daily_groups:
        returns = np.log(df_day / df_day.shift(1)).dropna()
        x = np.zeros((len(tickers), len(tickers)))
        for i in range(len(returns)):
            x += np.outer(returns.iloc[i], returns.iloc[i])
        cov_series.append(x)
    
    return cov_series
        
 


def true_daily_vol(tickers, w, days=70):
    vol = []
    
    start_date = datetime.now() - timedelta(days=days)
    end_date = start_date + timedelta(days=1)

    for _ in range(days):
        S = true_vol(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        vol_t = portfolio_vol(S, w)
        print(vol_t)
        
        start_date = start_date + timedelta(days=1)
        end_date = end_date + timedelta(days=1)
        
        if vol_t != 0:
            vol.append(vol_t)
    return np.array(vol)



def dcc_vol_over_time(Hts,w):
    vol = []
    for i in Hts:
        vol.append(portfolio_vol(i,w))
    return(vol)


def whole_thing(tickers = ["AAPL", "AMZN", "MSFT","NVDA"], w = np.array((0.2,0.2,0.3,0.3))):

    data = Data(tickers, '5y')  # Ensure Data provides returns for the tickers
    dcc = DCC(tickers, data.returns)
    
    # 1. Standardize residuals
    all_var, all_standardised_resids = dcc.standardising_residuals()
    
    # 2. Compute Qbar (unconditional covariance)
    Qbar = dcc.find_Qbar(all_standardised_resids)
    
    # 3. Estimate alpha and beta
    alpha, beta = dcc.estimate_alpha_beta(Qbar, all_standardised_resids)
    
    
    print(f"Optimized Alpha: {alpha}, Optimized Beta: {beta}")
    
    # 4. Compute Q_t matrices
    Q_matrices = dcc.find_Qt(all_standardised_resids, Qbar, alpha, beta)
    
    Hts,Rts = dcc.estimate_Hts(alpha, beta, all_var, Q_matrices)
    
    # Print final results    
    #tdv = true_daily_vol(tickers, w, days=30)
    
    cov_series = true_vol_new(tickers)
    tdv = []
    for x in cov_series:
        tdv.append(portfolio_vol(x,w))                  
    
    plt.plot(dcc_vol_over_time(Hts,w))
    #plt.plot(tdv)
    return(Hts,dcc_vol_over_time(Hts,w),Rts)


def portfolio_returns(w, returns_df):
    return returns_df @ w

def backtest(all_vols, returns_df, w=np.array([0.2, 0.2, 0.6])):
    port_ret = portfolio_returns(w, returns_df) 

    correct = 0
    incorrect = 0
    n = min(len(all_vols), len(port_ret))
    
    for i in range(n):
        var_threshold = 1.645 * all_vols[i]
        rp = port_ret[i]  

        if rp >= -var_threshold:
            correct += 1
        else:
            incorrect += 1
    
    if (correct + incorrect) == 0:
        return 0
    return correct / (correct + incorrect)

def hml(all_vols, Hts, Rts):
    rolling_avg = np.array(pd.Series(all_vols).rolling(window=7).mean().dropna())
    
    highest_t = rolling_avg.argmax() - 4
    lowest_t = rolling_avg.argmin() - 4
    median_t = (np.abs(rolling_avg - np.median(rolling_avg))).argmin() - 4
    
    avg_corrs = []
    avg_covs = []
    avg_vols = []
    valid_indices = [highest_t, median_t, lowest_t]
    
    for i in valid_indices:
        if i < 6:
            avg_corrs.append(None)
            avg_covs.append(None)
            avg_vols.append(None)
            continue
        

        if i + 1 > len(Hts):
            avg_corrs.append(None)
            avg_covs.append(None)
            avg_vols.append(None)
            continue
        
        avg_cov = np.mean(Hts[i - 3 : i + 4], axis=0)
        avg_corr = np.mean(Rts[i - 4 : i + 4], axis=0)
        avg_vol = np.mean(all_vols[i - 3 : i + 4])
        avg_corrs.append(avg_corr)
        avg_covs.append(avg_cov)
        avg_vols.append(avg_vol)
    
    return avg_corrs, avg_covs, avg_vols



def optimize_portfolio_variance(H):
    n = H.shape[0]
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    )

    bounds = [(0, 1) for _ in range(n)]

    w0 = np.ones(n) / n
    def portfolio_variance(w):
        return w.T @ H @ w
    result = minimize(
        portfolio_variance,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12}  

    )

    if not result.success:
        raise ValueError("Optimization failed:", result.message)

    return result.x

import yfinance as yf

def is_ticker_valid(ticker):
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return not data.empty  # Valid if data is not empty
    except Exception:
        return False

def both_vars(p_vol, days, simulations = 1000):
    x = np.random.normal(loc = 0, scale = p_vol*np.sqrt(days) ,size = simulations)
    var = - np.percentile(x, 5)
    tail_losses = x[x<=-var]
    cvar = - np.mean(tail_losses)
    return var, cvar

def cvar(p_vol, days, simulations = 1000):
    x = np.random.normal(loc = 0, scale = p_vol*np.sqrt(days) ,size = simulations)
    var = - np.percentile(x, 5)
    tail_losses = x[x<=var]
    cvar = - np.mean(tail_losses)
    return var, cvar
