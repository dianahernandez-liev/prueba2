import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew ,norm, shapiro
import scipy.stats as stats

st.set_page_config(
    page_title="Análisis de Rendimientos",
    page_icon=":chart_with_upwards_trend:",
    )
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #0f142e 100%);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00d4ff !important;
        font-family: 'Courier New', monospace !important;
        letter-spacing: 2px;
    }
    
    /* Metric cards */
    .stMetric {
        background: rgba(16, 20, 46, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .stMetric label {
        color: #8892b0 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Dataframes */
    .dataframe {
        background: rgba(16, 20, 46, 0.6);
        color: #ccd6f6;
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
    }
    
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #00d4ff 0%, #0066ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
    }
     /* Select Box General Styles */
    div[data-baseweb="select"] {
        background: rgba(16, 20, 46, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    
    div[data-baseweb="select"]:hover {
        border-color: rgba(0, 212, 255, 0.8) !important;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: rgba(16, 20, 46, 0.5);
        border-radius: 8px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #8892b0;
        font-family: monospace;
        font-size: 16px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00d4ff;
        border-bottom: 2px solid #00d4ff;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Visualización de Rendimientos de Acciones")
st.header("Media, Kurtosis y Sesgo")

@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Lista de acciones de ejemplo
stocks_lista = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
distribuciones = ['Normal', 't-Student']

with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)

# Selector de acción
stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

if stock_seleccionado:
    st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")
    
    rendimiento_medio = df_rendimientos[stock_seleccionado].mean()
    kurtosis_v = kurtosis(df_rendimientos[stock_seleccionado])
    sesgo = skew(df_rendimientos[stock_seleccionado])
    
    col1, col2, col3= st.columns(3)
    col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    col2.metric("Kurtosis", f"{kurtosis_v:.4}")
    col3.metric("Sesgo", f"{sesgo:.2}")

    # Gráfico de rendimientos diarios
    st.subheader(f"Gráfico de Rendimientos: {stock_seleccionado}")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado], label=stock_seleccionado)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax.set_title(f"Rendimientos de {stock_seleccionado}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Rendimiento Diario")
    ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
    ax.grid(True, alpha=0.2, color='white')
    st.pyplot(fig)
    # Histograma de rendimientos
    st.subheader("Distribución de Rendimientos")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_rendimientos[stock_seleccionado], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(rendimiento_medio, color='red', linestyle='dashed', linewidth=2, label=f"Promedio: {rendimiento_medio:.4%}")
    ax.set_title("Histograma de Rendimientos")
    ax.set_xlabel("Rendimiento Diario")
    ax.set_ylabel("Frecuencia")
    ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
    ax.grid(True, alpha=0.2, color='white')
    st.pyplot(fig)

    distribucion_seleccionada = st.selectbox("Selecciona una distribución", distribuciones)

    if distribucion_seleccionada:
        st.subheader(f"Métricas de Rendimiento: {stock_seleccionado}")
        if distribucion_seleccionada == 'Normal':
            #Metricas de riesgo
            # VaR Parametrico (distribución normal)
            mean_n = np.mean(df_rendimientos[stock_seleccionado])
            stdev_n = np.std(df_rendimientos[stock_seleccionado])
            VaR_95_n = (norm.ppf(1-0.95,mean_n,stdev_n))
            VaR_975_n = (norm.ppf(1-0.975,mean_n,stdev_n))
            VaR_99_n = (norm.ppf(1-0.99,mean_n,stdev_n))

            # Historical VaR
            hVaR_95 = (df_rendimientos[stock_seleccionado].quantile(0.05))
            hVaR_975 = (df_rendimientos[stock_seleccionado].quantile(0.025))
            hVaR_99 = (df_rendimientos[stock_seleccionado].quantile(0.01))

            # Monte Carlo

            # Number of simulations
            n_sims = 100000

            # Simulate returns and sort
            sim_returns = np.random.normal(mean_n, stdev_n, n_sims)

            MCVaR_95 = np.percentile(sim_returns, 5)
            MCVaR_975 = np.percentile(sim_returns, 2.5)
            MCVaR_99 = np.percentile(sim_returns, 1)

            CVaR_95 = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_95].mean())
            CVaR_975 = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_975].mean())
            CVaR_99 = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_99].mean())

            st.subheader("Metricas de riesgo")
            #95%
            col4, col5, col6, col7= st.columns(4)
            col4.metric("95% VaR", f"{VaR_95_n:.3%}")
            col5.metric("(Historical)", f"{hVaR_95:.3%}")
            col6.metric("(Monte Carlo)", f"{MCVaR_95:.3%}")
            col7.metric("95% CVaR", f"{CVaR_95:.3%}")
            st.subheader("Grafica metricas de riesgo")
            #Estilo general de las gráficas
            plt.style.use('dark_background')

            # Crear la figura y el eje
            fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0a0e27')
            ax.set_facecolor('#0f142e')

            # Generar histograma
            n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

            # Identificar y colorear de rojo las barras a la izquierda de hVaR_95
            for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
                if bin_left < hVaR_95:
                    patch.set_facecolor('red')

            # Marcar las líneas de VaR y CVaR
            ax.axvline(x=VaR_95_n, color='#00d4ff', linestyle='--', label='VaR 95% (Paramétrico)')
            ax.axvline(x=MCVaR_95, color='#9b59b6', linestyle='--', label='VaR 95% (Monte Carlo)')
            ax.axvline(x=hVaR_95, color='#00ff88', linestyle='--', label='VaR 95% (Histórico)')
            ax.axvline(x=CVaR_95, color='#ff6b35', linestyle='-.', label='CVaR 95%')

            # Configurar etiquetas y leyenda
            ax.set_title("Histograma de Rendimientos con VaR y CVaR", fontsize=14, fontweight='bold',color='#00d4ff',fontfamily='monospace',pad=20)
            ax.set_xlabel("Rendimiento Diario", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.set_ylabel("Frecuencia", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
            ax.grid(True, alpha=0.2, color='white')

            # Mostrar la figura en Streamlit
            st.pyplot(fig)
            

            #97.5%
            st.subheader("α = 0.975")
            col4, col5, col6, col7= st.columns(4)
            col4.metric("97.5% VaR", f"{VaR_975_n:.3%}")
            col5.metric("(Historical)", f"{hVaR_975:.3%}")
            col6.metric("(Monte Carlo)", f"{MCVaR_975:.3%}")
            col7.metric("97.5% CVaR", f"{CVaR_975:.3%}")
            # Crear la figura y el eje
            fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0a0e27')
            ax.set_facecolor('#0f142e')

            # Generar histograma
            n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

            # Identificar y colorear de rojo las barras a la izquierda de hVaR_95
            for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
                if bin_left < hVaR_975:
                    patch.set_facecolor('red')

            # Marcar las líneas de VaR y CVaR
            ax.axvline(x=VaR_975_n, color='#00d4ff', linestyle='--', label='VaR 97.5% (Paramétrico)')
            ax.axvline(x=MCVaR_975, color='#9b59b6', linestyle='--', label='VaR 97.5% (Monte Carlo)')
            ax.axvline(x=hVaR_975, color='#00ff88', linestyle='--', label='VaR 97.5% (Histórico)')
            ax.axvline(x=CVaR_975, color='#ff6b35', linestyle='-.', label='CVaR 97.5%')

            # Configurar etiquetas y leyenda
            ax.set_title("Histograma de Rendimientos con VaR y CVaR", fontsize=14, fontweight='bold',color='#00d4ff',fontfamily='monospace',pad=20)
            ax.set_xlabel("Rendimiento Diario", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.set_ylabel("Frecuencia", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
            ax.grid(True, alpha=0.2, color='white')

            # Mostrar la figura en Streamlit
            st.pyplot(fig)

            #99%
            st.subheader("α = 0.99")
            col4, col5, col6, col7= st.columns(4)
            col4.metric("99% VaR", f"{VaR_99_n:.3%}")
            col5.metric("(Historical)", f"{hVaR_99:.3%}")
            col6.metric("(Monte Carlo)", f"{MCVaR_99:.3%}")
            col7.metric("99% CVaR", f"{CVaR_99:.3%}")

            # Crear la figura y el eje
            fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0a0e27')
            ax.set_facecolor('#0f142e')

            # Generar histograma
            n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

            # Identificar y colorear de rojo las barras a la izquierda de hVaR_99
            for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
                if bin_left < hVaR_99:
                    patch.set_facecolor('red')

            # Marcar las líneas de VaR y CVaR
            ax.axvline(x=VaR_99_n, color='#00d4ff', linestyle='--', label='VaR 99% (Paramétrico)')
            ax.axvline(x=MCVaR_99, color='#9b59b6', linestyle='--', label='VaR 99% (Monte Carlo)')
            ax.axvline(x=hVaR_99, color='#00ff88', linestyle='--', label='VaR 99% (Histórico)')
            ax.axvline(x=CVaR_99, color='#ff6b35', linestyle='-.', label='CVaR 99%')

            # Configurar etiquetas y leyenda
            ax.set_title("Histograma de Rendimientos con VaR y CVaR", fontsize=14, fontweight='bold',color='#00d4ff',fontfamily='monospace',pad=20)
            ax.set_xlabel("Rendimiento Diario", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.set_ylabel("Frecuencia", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
            ax.grid(True, alpha=0.2, color='white')

            # Mostrar la figura en Streamlit
            st.pyplot(fig)

            
        elif distribucion_seleccionada == 't-Student':
            # VaR Parametrico (distribución t-student)
            mean_t = np.mean(df_rendimientos[stock_seleccionado])
            stdev_t = np.std(df_rendimientos[stock_seleccionado])
            VaR_95_t = (stats.t.ppf(1-0.95, df=len(df_rendimientos[stock_seleccionado])-1, loc=mean_t, scale=stdev_t))
            VaR_975_t = (stats.t.ppf(1-0.975, df=len(df_rendimientos[stock_seleccionado])-1, loc=mean_t, scale=stdev_t))
            VaR_99_t = (stats.t.ppf(1-0.99, df=len(df_rendimientos[stock_seleccionado])-1, loc=mean_t, scale=stdev_t))

            # Historical VaR
            hVaR_95_t = (df_rendimientos[stock_seleccionado].quantile(0.05))
            hVaR_975_t = (df_rendimientos[stock_seleccionado].quantile(0.025))
            hVaR_99_t = (df_rendimientos[stock_seleccionado].quantile(0.01))

            # Monte Carlo

            # Number of simulations
            n_sims = 100000

            # Simulate returns and sort
            sim_returns = stats.t.rvs(df=len(df_rendimientos[stock_seleccionado])-1, loc=mean_t, scale=stdev_t, size=n_sims)

            MCVaR_95_t = np.percentile(sim_returns, 5)
            MCVaR_975_t = np.percentile(sim_returns, 2.5)
            MCVaR_99_t = np.percentile(sim_returns, 1)

            CVaR_95_t = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_95_t].mean())
            CVaR_975_t = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_975_t].mean())
            CVaR_99_t = (df_rendimientos[stock_seleccionado][df_rendimientos[stock_seleccionado] <= hVaR_99_t].mean())

            st.subheader("Metricas de riesgo")
            #95%
            st.subheader("α = 0.95")
            col4, col5, col6, col7= st.columns(4)
            col4.metric("95% VaR", f"{VaR_95_t:.3%}")
            col5.metric("(Historical)", f"{hVaR_95_t:.3%}")
            col6.metric("(Monte Carlo)", f"{MCVaR_95_t:.3%}")
            col7.metric("95% CVaR", f"{CVaR_95_t:.3%}")
            st.subheader("Grafica metricas de riesgo")
            #Estilo general de las gráficas
            plt.style.use('dark_background')

            # Crear la figura y el eje
            fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0a0e27')
            ax.set_facecolor('#0f142e')

            # Generar histograma
            n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

            # Identificar y colorear de rojo las barras a la izquierda de hVaR_95
            for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
                if bin_left < hVaR_95_t:
                    patch.set_facecolor('red')

            # Marcar las líneas de VaR y CVaR
            ax.axvline(x=VaR_95_t, color='#00d4ff', linestyle='--', label='VaR 95% (Paramétrico)')
            ax.axvline(x=MCVaR_95_t, color='#9b59b6', linestyle='--', label='VaR 95% (Monte Carlo)')
            ax.axvline(x=hVaR_95_t, color='#00ff88', linestyle='--', label='VaR 95% (Histórico)')
            ax.axvline(x=CVaR_95_t, color='#ff6b35', linestyle='-.', label='CVaR 95%')

            # Configurar etiquetas y leyenda
            ax.set_title("Histograma de Rendimientos con VaR y CVaR", fontsize=14, fontweight='bold',color='#00d4ff',fontfamily='monospace',pad=20)
            ax.set_xlabel("Rendimiento Diario", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.set_ylabel("Frecuencia", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
            ax.grid(True, alpha=0.2, color='white')

            # Mostrar la figura en Streamlit
            st.pyplot(fig)
            

            #97.5%
            st.subheader("α = 0.975")
            col4, col5, col6, col7= st.columns(4)
            col4.metric("97.5% VaR", f"{VaR_975_t:.3%}")
            col5.metric("(Historical)", f"{hVaR_975_t:.3%}")
            col6.metric("(Monte Carlo)", f"{MCVaR_975_t:.3%}")
            col7.metric("97.5% CVaR", f"{CVaR_975_t:.3%}")
            # Crear la figura y el eje
            fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0a0e27')
            ax.set_facecolor('#0f142e')

            # Generar histograma
            n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

            # Identificar y colorear de rojo las barras a la izquierda de hVaR_95
            for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
                if bin_left < hVaR_975_t:
                    patch.set_facecolor('red')

            # Marcar las líneas de VaR y CVaR
            ax.axvline(x=VaR_975_t, color='#00d4ff', linestyle='--', label='VaR 97.5% (Paramétrico)')
            ax.axvline(x=MCVaR_975_t, color='#9b59b6', linestyle='--', label='VaR 97.5% (Monte Carlo)')
            ax.axvline(x=hVaR_975_t, color='#00ff88', linestyle='--', label='VaR 97.5% (Histórico)')
            ax.axvline(x=CVaR_975_t, color='#ff6b35', linestyle='-.', label='CVaR 97.5%')

            # Configurar etiquetas y leyenda
            ax.set_title("Histograma de Rendimientos con VaR y CVaR", fontsize=14, fontweight='bold',color='#00d4ff',fontfamily='monospace',pad=20)
            ax.set_xlabel("Rendimiento Diario", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.set_ylabel("Frecuencia", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
            ax.grid(True, alpha=0.2, color='white')

            # Mostrar la figura en Streamlit
            st.pyplot(fig)

            #99%
            st.subheader("α = 0.99")
            col4, col5, col6, col7= st.columns(4)
            col4.metric("99% VaR", f"{VaR_99_t:.3%}")
            col5.metric("(Historical)", f"{hVaR_99_t:.3%}")
            col6.metric("(Monte Carlo)", f"{MCVaR_99_t:.3%}")
            col7.metric("99% CVaR", f"{CVaR_99_t:.3%}")

            # Crear la figura y el eje
            fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0a0e27')
            ax.set_facecolor('#0f142e')

            # Generar histograma
            n, bins, patches = ax.hist(df_rendimientos[stock_seleccionado], bins=50, color='blue', alpha=0.7, label='Returns')

            # Identificar y colorear de rojo las barras a la izquierda de hVaR_99
            for bin_left, bin_right, patch in zip(bins, bins[1:], patches):
                if bin_left < hVaR_99_t:
                    patch.set_facecolor('red')

            # Marcar las líneas de VaR y CVaR
            ax.axvline(x=VaR_99_t, color='#00d4ff', linestyle='--', label='VaR 99% (Paramétrico)')
            ax.axvline(x=MCVaR_99_t, color='#9b59b6', linestyle='--', label='VaR 99% (Monte Carlo)')
            ax.axvline(x=hVaR_99_t, color='#00ff88', linestyle='--', label='VaR 99% (Histórico)')
            ax.axvline(x=CVaR_99_t, color='#ff6b35', linestyle='-.', label='CVaR 99%')

            # Configurar etiquetas y leyenda
            ax.set_title("Histograma de Rendimientos con VaR y CVaR", fontsize=14, fontweight='bold',color='#00d4ff',fontfamily='monospace',pad=20)
            ax.set_xlabel("Rendimiento Diario", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.set_ylabel("Frecuencia", fontsize=11, color='#8892b0',fontfamily='monospace',fontweight='bold')
            ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
            ax.grid(True, alpha=0.2, color='white')

            # Mostrar la figura en Streamlit
            st.pyplot(fig)


        
    ### ROLLING WINDOWS
    st.header("Rolling Windows VaR")
    # Calcular rolling windows SOLO para la acción seleccionada
    rolling_mean = df_rendimientos[stock_seleccionado].rolling(window=252).mean()
    rolling_std = df_rendimientos[stock_seleccionado].rolling(window=252).std()
    
    # VaR 95% Rolling Paramétrico
    VaR_95_rolling = norm.ppf(1-0.95, rolling_mean, rolling_std)
    VaR_95_rolling_percent = VaR_95_rolling * 100
    
    # VaR 95% Rolling Histórico
    hVaR_95_rolling = df_rendimientos[stock_seleccionado].rolling(window=252).quantile(0.05)
    hVaR_95_rolling_percent = hVaR_95_rolling * 100

    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(13, 5), facecolor='#0a0e27')
    ax.set_facecolor('#0f142e')
    ax.plot(df_rendimientos[stock_seleccionado].index, df_rendimientos[stock_seleccionado] * 100, label='Retornos diarios (%)', color='blue', alpha=0.5)

    
    ax.plot(df_rendimientos.index, hVaR_95_rolling_percent, label='95% Rolling VaR Histórico', color='#00ff88', linewidth=2)
    ax.plot(df_rendimientos.index, VaR_95_rolling_percent, label='95% Rolling VaR Paramétrico', color='#00d4ff', linewidth=2)
    #Configurar etiquetas y leyenda
    ax.set_title(f'95% Rolling VaR - {stock_seleccionado}', fontsize=14, fontweight='bold', color='#00d4ff', fontfamily='monospace', pad=20)
    ax.set_xlabel('Fecha', fontsize=11, color='#8892b0', fontfamily='monospace', fontweight='bold')
    ax.set_ylabel('VaR (%)', fontsize=11, color='#8892b0', fontfamily='monospace', fontweight='bold')
    ax.legend(loc='upper left', facecolor='#0f142e', edgecolor='#00d4ff')
    ax.grid(True, alpha=0.2, color='white')
    
    # Mostrar la figura
    st.pyplot(fig)

    st.subheader("Evaluación de Violaciones del VaR")
    contador = 0
    for i in range(len(df_rendimientos[stock_seleccionado])):
        if df_rendimientos[stock_seleccionado].iloc[i+1] < hVaR_95_rolling.iloc[i]:
            contador += 1
    # Calcular porcentaje de violaciones
    porcentaje_violaciones = (contador / len(df_rendimientos[stock_seleccionado])) * 100

    col1, col2, col3= st.columns(3)
    col1.metric("Violaciones", f"{contador}")
    col2.metric("Porcentaje de Violaciones", f"{porcentaje_violaciones:.2f}%")
    col3.metric("Total de Días", f"{len(df_rendimientos[stock_seleccionado])}")
