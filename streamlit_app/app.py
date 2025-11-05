# -*- coding: utf-8 -*-
"""app.ipynb

Original file is located at
    https://colab.research.google.com/drive/1CzDZnFq4QJqJNt7Oyxw5qMmY683PDNAV
"""

# streamlit_app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="Grupo Juyo", layout="wide")
sns.set(style="whitegrid")
plt.switch_backend("Agg")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Carátula
st.image("https://copilot.microsoft.com/th/id/BCO.3345d9e9-667d-460a-91a0-b556a8c6e75d.png", use_column_width=True)
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
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="viridis")
        st.pyplot(plt)

        st.write("Distribuciones:")
        df.select_dtypes(include="number").hist(figsize=(12, 8))
        st.pyplot(plt)

        if "frecuencia_mensual" in df.columns:
            st.write("Frecuencia mensual de compra:")
            sns.histplot(df["frecuencia_mensual"].dropna(), bins=30, kde=True)
            st.pyplot()

        if "variabilidad_monto_cliente" in df.columns and "segmento_comportamiento" in df.columns:
            st.write("Variabilidad del monto por segmento:")
            sns.boxplot(x="segmento_comportamiento", y="variabilidad_monto_cliente", data=df)
            plt.xticks(rotation=45)
            st.pyplot()

    except Exception as e:
        st.error(f"No se pudo cargar el archivo: {e}")

# Sección: Modelos
elif menu == "Modelos":
    st.subheader("Predicción con modelos del Grupo Juyo")

    modelos_disponibles = {
        "Cantidad comprada": "modelo_cantidad.pkl",
        "Probabilidad de compra": "modelo_compra.pkl",
        "Día de compra": "modelo_dia_compra.pkl",
        "Producto comprado": "modelo_producto.pkl"
    }

    seleccion = st.selectbox("Selecciona un modelo", list(modelos_disponibles.keys()))
    modelo_path = f"models/{modelos_disponibles[seleccion]}"

    try:
        with open(modelo_path, "rb") as f:
            modelo = pickle.load(f)

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

    except Exception as e:
        st.error(f"No se pudo cargar el modelo: {e}")