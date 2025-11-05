# -*- coding: utf-8 -*-
# streamlit_app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestRegressor

st.write("RandomForestRegressor disponible.")

st.set_page_config(page_title="Grupo Juyo", layout="wide")
sns.set(style="whitegrid")
plt.switch_backend("Agg")

# Carátula
st.image("streamlit_app/assets/portada_juyo.png", use_column_width=True)
st.title("Modelamiento Grupo Juyo")

# Menú lateral
menu = st.sidebar.radio("Menú principal", ["Mapa", "EDA", "Modelos"])

# Sección: Mapa
if menu == "Mapa":
    st.subheader("Distribución geográfica de gaseosas")
    try:
        df_mapa = pd.read_csv("data/raw/data6_corregido.csv")
        if "coordenadas_envio" in df_mapa.columns:
            coords = df_mapa["coordenadas_envio"].dropna().str.extract(r"\((.*),\s*(.*)\)")
            coords.columns = ["lat", "lon"]
            coords = coords.astype(float)
            st.map(coords)
        else:
            st.warning("No se encontraron coordenadas en la data corregida.")
    except Exception as e:
        st.error(f"No se pudo cargar el mapa: {e}")

# Sección: EDA
elif menu == "EDA":
    tipo_eda = st.radio("Tipo de EDA", ["Original", "Limpio", "Corregido", "Con features"])

    rutas = {
        "Original": "data/data6.csv",
        "Limpio": "data/data6_limpio.csv",
        "Corregido": "data/raw/data6_corregido.csv",
        "Con features": "data/features/data6_features.csv"
    }

    ruta = rutas[tipo_eda]
    try:
        df = pd.read_csv(ruta)
        st.subheader(f"EDA - Datos {tipo_eda.lower()}")
        st.write("Vista general:")
        st.dataframe(df.head())

        st.write("Estadísticas descriptivas:")
        st.write(df.describe())

        st.write("Valores nulos:")
        st.write(df.isnull().sum())

        st.write("Correlaciones:")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="viridis", ax=ax1)
        st.pyplot(fig1)

        st.write("Distribuciones:")
        df.select_dtypes(include="number").hist(figsize=(12, 8))
        st.pyplot(plt)

        if "frecuencia_mensual" in df.columns:
            st.write("Frecuencia mensual de compra:")
            fig3, ax3 = plt.subplots()
            sns.histplot(df["frecuencia_mensual"].dropna(), bins=30, kde=True, ax=ax3)
            st.pyplot(fig3)

        if "variabilidad_monto_cliente" in df.columns and "segmento_comportamiento" in df.columns:
            st.write("Variabilidad del monto por segmento:")
            fig4, ax4 = plt.subplots()
            sns.boxplot(x="segmento_comportamiento", y="variabilidad_monto_cliente", data=df, ax=ax4)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            st.pyplot(fig4)

    except Exception as e:
        st.error(f"No se pudo cargar el archivo: {e}")

# Sección: Modelos
elif menu == "Modelos":
    st.subheader("Predicción con modelos del Grupo Juyo")

    modelos_disponibles = {
        "Probabilidad de compra": "models/mejor_modelo_compro.pkl",
        "Día de compra": "models/modelo_dia_compra.pkl",
        "Producto comprado": "models/modelo_producto.pkl",
        "Cantidad comprada": "models/modelo_cantidad.pkl"
    }

    seleccion = st.selectbox("Selecciona un modelo", list(modelos_disponibles.keys()))
    modelo_path = modelos_disponibles[seleccion]

    try:
        with open(modelo_path, "rb") as f:
            modelo = pickle.load(f)

        if not hasattr(modelo, "predict"):
            st.error(f"El archivo '{modelo_path}' no contiene un modelo válido. Tipo: {type(modelo)}")
        else:
            df = pd.read_csv("data/features/data6_features.csv")

            st.write("Datos de entrada:")
            st.dataframe(df.head())

            pred = modelo.predict(df)

            st.write(f"Predicciones para: {seleccion}")
            st.write(pred)

            if seleccion == "Cantidad comprada":
                st.bar_chart(pred)
            elif seleccion == "Día de compra":
                st.bar_chart(pd.Series(pred).value_counts().sort_index())
            elif seleccion == "Producto comprado":
                st.write("Distribución de productos:")
                st.bar_chart(pd.Series(pred).value_counts())
            elif seleccion == "Probabilidad de compra":
                st.line_chart(pred)

    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {modelo_path}")
    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")
