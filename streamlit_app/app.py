# -*- coding: utf-8 -*-
# streamlit_app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor


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
    st.subheader("Distribución geográfica de ventas de gaseosas")
    
    try:
        df_mapa = pd.read_csv("data/raw/data6_corregido.csv")
        
        if "coordenadas_envio" in df_mapa.columns:
            # Extraer coordenadas
            coords = df_mapa["coordenadas_envio"].dropna().str.extract(r"\((.*),\s*(.*)\)")
            coords.columns = ["lat", "lon"]
            coords = coords.astype(float)
            
            # Filtrar coordenadas válidas
            coords = coords.dropna()
            coords = coords[(coords['lat'] >= -90) & (coords['lat'] <= 90) & 
                           (coords['lon'] >= -180) & (coords['lon'] <= 180)]
            
            if len(coords) > 0:
                # Mostrar estadísticas de ubicaciones
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de ubicaciones", len(coords))
                with col2:
                    st.metric("Ubicaciones únicas", len(coords.drop_duplicates()))
                with col3:
                    st.metric("Cobertura geográfica", f"{len(coords)/len(df_mapa)*100:.1f}%")
                
                # Selector de tipo de mapa
                tipo_mapa = st.radio(
                    "Tipo de visualización:",
                    ["Mapa de calor", "Puntos de distribución", "Análisis por densidad"],
                    horizontal=True
                )
                
                if tipo_mapa == "Mapa de calor":
                    # Mapa de calor
                    st.map(coords, use_container_width=True)
                    
                elif tipo_mapa == "Puntos de distribución":
                    # Puntos individuales con información adicional
                    if "cantidad" in df_mapa.columns and "producto" in df_mapa.columns:
                        # Crear DataFrame con información de ventas
                        mapa_data = coords.copy()
                        mapa_data['cantidad'] = df_mapa.loc[coords.index, 'cantidad'].values
                        mapa_data['producto'] = df_mapa.loc[coords.index, 'producto'].values
                        
                        # Mostrar puntos con tamaño según cantidad
                        st.map(mapa_data, size='cantidad', color='#FF4B4B')
                        
                        # Leyenda
                        st.caption("Tamaño de puntos representa la cantidad vendida")
                    else:
                        st.map(coords)
                        
                else:  # Análisis por densidad
                    # Agrupar por áreas geográficas
                    coords_rounded = coords.copy()
                    coords_rounded['lat'] = coords_rounded['lat'].round(1)
                    coords_rounded['lon'] = coords_rounded['lon'].round(1)
                    
                    densidad = coords_rounded.groupby(['lat', 'lon']).size().reset_index(name='count')
                    densidad = densidad[densidad['count'] > 1]  # Solo áreas con múltiples puntos
                    
                    if len(densidad) > 0:
                        st.map(densidad, size='count', color='#00CC96')
                        st.caption("Áreas con mayor densidad de ventas (puntos agrupados)")
                    else:
                        st.map(coords)
                        st.info("No se detectaron áreas de alta densidad")
                
                # Análisis adicional
                st.write("### Análisis Geográfico")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribución por regiones (simulada)
                    st.write("**Distribución aproximada por regiones:**")
                    if len(coords) > 0:
                        # Simular regiones basado en coordenadas
                        norte = len(coords[coords['lat'] > 4])
                        centro = len(coords[(coords['lat'] <= 4) & (coords['lat'] > 2)])
                        sur = len(coords[coords['lat'] <= 2])
                        
                        fig_regiones, ax_regiones = plt.subplots(figsize=(8, 4))
                        regiones = ['Norte', 'Centro', 'Sur']
                        ventas = [norte, centro, sur]
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                        
                        bars = ax_regiones.bar(regiones, ventas, color=colors, alpha=0.8)
                        ax_regiones.set_ylabel('Número de Ventas')
                        ax_regiones.set_title('Distribución de Ventas por Región')
                        
                        # Agregar valores en las barras
                        for bar, valor in zip(bars, ventas):
                            ax_regiones.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                           f'{valor}', ha='center', va='bottom')
                        
                        st.pyplot(fig_regiones)
                
                with col2:
                    # Top productos por ubicación
                    if "producto" in df_mapa.columns:
                        st.write("**Productos más vendidos:**")
                        top_productos = df_mapa['producto'].value_counts().head(5)
                        
                        fig_productos, ax_productos = plt.subplots(figsize=(8, 4))
                        top_productos.plot(kind='barh', ax=ax_productos, color='#A78BFA')
                        ax_productos.set_xlabel('Número de Ventas')
                        ax_productos.set_title('Top 5 Productos Más Vendidos')
                        plt.tight_layout()
                        st.pyplot(fig_productos)
                
                # Información detallada
                with st.expander("📊 Ver detalles de las ubicaciones"):
                    st.write("**Resumen de coordenadas:**")
                    st.write(f"- Latitud mínima: {coords['lat'].min():.4f}")
                    st.write(f"- Latitud máxima: {coords['lat'].max():.4f}")
                    st.write(f"- Longitud mínima: {coords['lon'].min():.4f}")
                    st.write(f"- Longitud máxima: {coords['lon'].max():.4f}")
                    st.write(f"- Centro geográfico: ({coords['lat'].mean():.4f}, {coords['lon'].mean():.4f})")
                    
                    # Mostrar tabla con algunas ubicaciones
                    st.write("**Muestra de ubicaciones:**")
                    muestra_ubicaciones = coords.head(10).copy()
                    if "producto" in df_mapa.columns:
                        muestra_ubicaciones['producto'] = df_mapa.loc[muestra_ubicaciones.index, 'producto'].values
                    st.dataframe(muestra_ubicaciones)
                    
            else:
                st.warning("No se encontraron coordenadas válidas en los datos.")
                
        else:
            st.warning("No se encontró la columna 'coordenadas_envio' en los datos.")
            st.info("""
            **Solución sugerida:**
            - Verifica que el archivo data6_corregido.csv contenga la columna 'coordenadas_envio'
            - Asegúrate de que las coordenadas estén en formato: (latitud, longitud)
            - Ejemplo: (-12.0464, -77.0428)
            """)
            
    except Exception as e:
        st.error(f"No se pudo cargar el mapa: {e}")
        st.info("""
        **Posibles soluciones:**
        - Verifica que el archivo data/raw/data6_corregido.csv exista
        - Asegúrate de que el archivo tenga el formato CSV correcto
        - Revisa que las coordenadas estén en el formato esperado
        """)

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
        
        # VISTA GENERAL PARA TODOS
        st.write("### Vista general:")
        st.dataframe(df.head())
        
        st.write("### Estadísticas descriptivas:")
        st.write(df.describe())
        
        st.write("### Valores nulos:")
        nulos = df.isnull().sum()
        st.write(nulos)
        
        # CONTENIDO ESPECÍFICO POR TIPO DE EDA
        if tipo_eda == "Original":
            st.info("""
            **Dataset Original** - Datos sin procesar directamente de la fuente.
            Contiene valores nulos, inconsistencias y datos sin transformar.
            """)
            
        elif tipo_eda == "Limpio":
            st.success("""
            **RESUMEN DE LIMPIEZA APLICADA:**
            - ✓ Valores nulos: ELIMINADOS (0 restantes)
            - ✓ Valores negativos: ELIMINADOS en cantidad, precio, variables climáticas
            - ✓ Rangos lógicos: ESTABLECIDOS (humedad 0-100%, viento ≥0, etc.)
            - ✓ Valores extremos: CORREGIDOS (gasto publicidad, inflación)
            - ✓ Fechas: CONVERTIDAS a datetime
            - ✓ Dataset listo para análisis avanzado
            """)
            
            # Gráficas específicas para datos limpios
            if "cantidad" in df.columns:
                st.write("### Distribución de Cantidad (Post-Limpieza)")
                fig, ax = plt.subplots(figsize=(10, 4))
                df["cantidad"].hist(bins=30, ax=ax, color='lightgreen')
                ax.set_title('Distribución de Cantidad - Datos Limpios')
                ax.set_xlabel('Cantidad')
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)
                
        elif tipo_eda == "Corregido":
            st.warning("""
            **MEJORAS INCORPORADAS:**
            - • Métodos de pago unificados y consistentes
            - • Tamaños de pedido categorizados  
            - • Variables climáticas normalizadas
            - • Fechas convertidas a datetime
            - • Valores extremos manejados
            """)
            
            # Análisis de variables numéricas corregidas
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                st.write("### Variables Numéricas (Corregidas)")
                # Seleccionar primeras 4 variables numéricas para mostrar
                cols_to_show = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes = axes.ravel()
                
                for i, col in enumerate(cols_to_show):
                    if i < len(axes):
                        df[col].hist(bins=20, ax=axes[i], color='lightblue')
                        axes[i].set_title(f'Distribución de {col}')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Frecuencia')
                
                # Ocultar ejes vacíos
                for i in range(len(cols_to_show), len(axes)):
                    axes[i].set_visible(False)
                    
                plt.tight_layout()
                st.pyplot(fig)
            
            # Análisis de variables categóricas corregidas
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            if len(categorical_cols) > 0:
                st.write("### Variables Categóricas (Corregidas)")
                # Seleccionar primeras 2 variables categóricas para mostrar
                cat_cols_to_show = categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols
                
                for col in cat_cols_to_show:
                    value_counts = df[col].value_counts().head(10)  # Top 10 categorías
                    if len(value_counts) > 0:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        value_counts.plot(kind='bar', ax=ax, color='orange')
                        ax.set_title(f'Distribución de {col}')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Frecuencia')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                        
        elif tipo_eda == "Con features":
            st.success("""
            **Dataset con Features de Ingeniería** - Contiene variables creadas para el modelamiento:
            - Comportamiento del cliente (frecuencia, antigüedad, valor promedio)
            - Segmentación (comportamiento, tipo cliente)
            - Variables temporales (estación, día de semana)
            - Métricas de consistencia
            """)
            
            # Mostrar features creadas específicamente
            engineered_features = [
                'frecuencia_mensual', 'antiguedad_meses', 'valor_promedio_cliente',
                'variabilidad_monto_cliente', 'consistencia_metodo_pago',
                'segmento_comportamiento', 'producto_favorito_cliente',
                'es_fin_de_semana', 'estacion', 'dia_semana_num'
            ]
            
            features_presentes = [f for f in engineered_features if f in df.columns]
            
            if features_presentes:
                st.write("### Features de Ingeniería Creadas")
                
                # Mostrar estadísticas de las features más importantes
                important_features = features_presentes[:6]  # Mostrar primeras 6
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.ravel()
                
                for i, feature in enumerate(important_features):
                    if i < len(axes):
                        df[feature].hist(bins=20, ax=axes[i], color='purple', alpha=0.7)
                        axes[i].set_title(feature)
                        axes[i].set_xlabel('Valor')
                        axes[i].set_ylabel('Frecuencia')
                
                # Ocultar ejes vacíos
                for i in range(len(important_features), len(axes)):
                    axes[i].set_visible(False)
                    
                plt.tight_layout()
                st.pyplot(fig)
                
                # Mostrar descripción de las features
                st.write("**Descripción de Features:**")
                feature_descriptions = {
                    'frecuencia_mensual': 'Número de compras mensuales del cliente',
                    'antiguedad_meses': 'Tiempo como cliente en meses',
                    'valor_promedio_cliente': 'Gasto promedio por transacción',
                    'variabilidad_monto_cliente': 'Consistencia en montos gastados',
                    'consistencia_metodo_pago': 'Estabilidad en métodos de pago',
                    'segmento_comportamiento': 'Segmento según comportamiento de compra',
                    'producto_favorito_cliente': 'Producto más comprado por el cliente',
                    'es_fin_de_semana': 'Indica si la compra fue en fin de semana',
                    'estacion': 'Estación del año de la compra',
                    'dia_semana_num': 'Día de la semana (numérico)'
                }
                
                for feature in features_presentes:
                    if feature in feature_descriptions:
                        st.write(f"• **{feature}**: {feature_descriptions[feature]}")

    except Exception as e:
        st.error(f"No se pudo cargar el archivo: {e}")
        st.write("Asegúrate de que los archivos de datos existan en las rutas especificadas.")

# Sección: Modelos
elif menu == "Modelos":
    st.subheader("Predicción con modelos del Grupo Juyo")
    
    # Cachear datos y modelos para mejor rendimiento
    @st.cache_data
    def cargar_datos():
        df = pd.read_csv("data/features/data6_features.csv")
        
        # PREPARAR DATOS ESPECÍFICAMENTE PARA MODELO DE CANTIDAD
        # Crear tamaño_pedido_num si no existe (igual que en el entrenamiento)
        if 'tamaño_pedido_num' not in df.columns and 'tamaño_pedido' in df.columns:
            tamaño_mapping = {'Pequeño': 1, 'Mediano': 2, 'Grande': 3, 'Muy Grande': 4}
            df['tamaño_pedido_num'] = df['tamaño_pedido'].map(tamaño_mapping)
            # Rellenar valores nulos con 2 (Mediano)
            df['tamaño_pedido_num'] = df['tamaño_pedido_num'].fillna(2)
        
        # Features para CADA modelo por separado:
        features_por_modelo = {
            'compro': ['frecuencia_mensual', 'antiguedad_meses', 'valor_promedio_cliente', 
                      'es_fin_de_semana', 'variabilidad_monto_cliente', 'consistencia_metodo_pago'],
            'dia_compra': ['frecuencia_mensual', 'antiguedad_meses', 'valor_promedio_cliente', 
                          'es_fin_de_semana', 'variabilidad_monto_cliente', 'consistencia_metodo_pago'],
            'producto': [
                'tipo_cliente', 'frecuencia_mensual', 'valor_promedio_cliente',
                'producto_favorito_cliente', 'metodo_pago_habitual', 'segmento_comportamiento',
                'antiguedad_meses', 'dia_semana_num', 'estacion'
            ],
            'cantidad': [
                'frecuencia_mensual', 'antiguedad_meses', 'valor_promedio_cliente',
                'es_fin_de_semana', 'variabilidad_monto_cliente', 'consistencia_metodo_pago',
                'precio_unitario', 'descuento', 'tamaño_pedido_num'  # INCLUIR todas las features del entrenamiento
            ]
        }
        
        return df, features_por_modelo
    
    @st.cache_resource
    def cargar_modelo(modelo_path):
        with open(modelo_path, "rb") as f:
            return pickle.load(f)
    
    datos_completos, features_por_modelo = cargar_datos()
    
    modelos_disponibles = {
        "Probabilidad de compra": "models/mejor_modelo_compro.pkl",
        "Día de compra": "models/modelo_dia_compra.pkl",
        "Producto comprado": "models/modelo_producto.pkl", 
        "Cantidad comprada": "models/modelo_cantidad.pkl"
    }

    seleccion = st.selectbox("Selecciona un modelo", list(modelos_disponibles.keys()))
    
    if st.button("Ejecutar Predicción"):
        modelo_path = modelos_disponibles[seleccion]

        try:
            # CARGAR DATOS ESPECÍFICOS PARA CADA MODELO
            if seleccion == "Probabilidad de compra":
                datos_muestra = datos_completos[features_por_modelo['compro']].head(100)
                modelo_data = cargar_modelo(modelo_path)
                
                # Verificar si es diccionario o modelo directo
                if isinstance(modelo_data, dict) and 'model' in modelo_data:
                    modelo = modelo_data['model']
                    features_esperadas = modelo_data.get('features', features_por_modelo['compro'])
                    datos_muestra = datos_completos[features_esperadas].head(100)
                else:
                    modelo = modelo_data
                
                with st.spinner('Calculando predicciones...'):
                    pred = modelo.predict(datos_muestra)
                    if hasattr(modelo, 'predict_proba'):
                        pred_proba = modelo.predict_proba(datos_muestra)[:, 1]
                    else:
                        pred_proba = pred

            elif seleccion == "Día de compra":
                datos_muestra = datos_completos[features_por_modelo['dia_compra']].head(100)
                modelo_data = cargar_modelo(modelo_path)
                
                if isinstance(modelo_data, dict) and 'model' in modelo_data:
                    modelo = modelo_data['model']
                    features_esperadas = modelo_data.get('features', features_por_modelo['dia_compra'])
                    datos_muestra = datos_completos[features_esperadas].head(100)
                else:
                    modelo = modelo_data
                
                with st.spinner('Calculando predicciones...'):
                    pred = modelo.predict(datos_muestra)
                    if hasattr(modelo, 'predict_proba'):
                        pred_proba = modelo.predict_proba(datos_muestra)[:, 1]
                    else:
                        pred_proba = pred

            elif seleccion == "Producto comprado":
                modelo_data = cargar_modelo(modelo_path)
                
                # Extraer modelo y features del diccionario
                if isinstance(modelo_data, dict) and 'model' in modelo_data:
                    modelo = modelo_data['model']
                    features_esperadas = modelo_data.get('features', features_por_modelo['producto'])
                else:
                    modelo = modelo_data
                    features_esperadas = features_por_modelo['producto']
                
                datos_muestra = datos_completos[features_esperadas].head(100).copy()
                
                # Procesar tipos de datos para CatBoost
                categorical_cols = ['tipo_cliente', 'producto_favorito_cliente', 
                                  'metodo_pago_habitual', 'segmento_comportamiento', 'estacion']
                
                for col in categorical_cols:
                    if col in datos_muestra.columns:
                        datos_muestra[col] = datos_muestra[col].astype('category')
                
                # Predecir
                with st.spinner('Calculando predicciones de producto...'):
                    pred = modelo.predict(datos_muestra)
                    if hasattr(modelo, 'predict_proba'):
                        pred_proba = modelo.predict_proba(datos_muestra)
                        pred_proba = np.max(pred_proba, axis=1)
                    else:
                        pred_proba = pred

            elif seleccion == "Cantidad comprada":
                modelo_data = cargar_modelo(modelo_path)
                
                # Extraer modelo y features del diccionario
                if isinstance(modelo_data, dict) and 'model' in modelo_data:
                    modelo = modelo_data['model']
                    features_esperadas = modelo_data.get('features', features_por_modelo['cantidad'])
                    
                    # VERIFICAR Y PREPARAR FEATURES EXACTAMENTE COMO EN ENTRENAMIENTO
                    features_faltantes = [f for f in features_esperadas if f not in datos_completos.columns]
                    if features_faltantes:
                        st.warning(f"Features faltantes: {features_faltantes}. Usando todas las disponibles.")
                        # Usar solo las features disponibles
                        features_esperadas = [f for f in features_esperadas if f in datos_completos.columns]
                else:
                    modelo = modelo_data
                    features_esperadas = features_por_modelo['cantidad']
                
                # Asegurar que tenemos todas las features necesarias
                if not features_esperadas:
                    st.error("No hay features disponibles para el modelo de cantidad")
                    st.stop()
                
                datos_muestra = datos_completos[features_esperadas].head(100)
                
                # VERIFICAR VALORES NULOS Y PREPROCESAR
                if datos_muestra.isnull().any().any():
                    st.warning("Hay valores nulos en los datos. Se rellenarán con 0.")
                    datos_muestra = datos_muestra.fillna(0)
                
                with st.spinner('Calculando predicciones de cantidad...'):
                    pred = modelo.predict(datos_muestra)
                    
                    # Para modelo de cantidad, usar las predicciones directamente como probabilidades
                    # Normalizar entre 0 y 1 para la visualización
                    pred_min = pred.min()
                    pred_max = pred.max()
                    if pred_max > pred_min:
                        pred_proba = (pred - pred_min) / (pred_max - pred_min)
                    else:
                        pred_proba = pred

            st.success("Predicciones completadas")
            
            # Mostrar información del modelo
            st.write(f"**Modelo:** {seleccion}")
            st.write(f"**Features utilizadas:** {len(datos_muestra.columns)}")
            st.write(f"**Muestras analizadas:** {len(datos_muestra)}")
            
            # Mostrar datos de entrada
            st.write("### Datos de Entrada (primeras 10 filas)")
            st.dataframe(datos_muestra.head(10))

            # Mostrar estadísticas básicas
            st.write("### Estadísticas de Predicciones")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mínimo", f"{pred_proba.min():.3f}")
            with col2:
                st.metric("Máximo", f"{pred_proba.max():.3f}")
            with col3:
                st.metric("Promedio", f"{pred_proba.mean():.3f}")
            with col4:
                st.metric("Desviación", f"{pred_proba.std():.3f}")

            # DISTRIBUCIÓN DE LAS PREDICCIONES
            st.write("### Distribución de Probabilidades")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(pred_proba, bins=20, alpha=0.7, color='blue', edgecolor='black')
            
            if seleccion == "Día de compra":
                ax.set_xlabel('Probabilidad de Compra el Martes')
                ax.set_title('Distribución de Probabilidades de Compra el Martes')
            elif seleccion == "Probabilidad de compra":
                ax.set_xlabel('Probabilidad de Compra')
                ax.set_title('Distribución de Probabilidades de Compra')
            elif seleccion == "Producto comprado":
                ax.set_xlabel('Confianza del Producto Predicho')
                ax.set_title('Distribución de Confianza en Predicciones de Producto')
            elif seleccion == "Cantidad comprada":
                ax.set_xlabel('Cantidad Normalizada')
                ax.set_title('Distribución de Cantidades Predichas (Normalizadas)')
            else:
                ax.set_xlabel('Probabilidad')
                ax.set_title('Distribución de las Predicciones')
                
            ax.set_ylabel('Frecuencia')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # ANÁLISIS DE UMBRALES
            st.write("### Análisis de Umbrales de Decisión")
            
            if seleccion == "Día de compra":
                umbrales = [0.3, 0.5, 0.7]
                resultados_umbral = []
                for umbral in umbrales:
                    martes_predichos = (pred_proba > umbral).sum()
                    porcentaje = (martes_predichos / len(pred_proba)) * 100
                    resultados_umbral.append({
                        'Umbral': umbral,
                        'Clientes Martes': martes_predichos,
                        'Porcentaje': f'{porcentaje:.1f}%'
                    })
            elif seleccion == "Producto comprado":
                umbrales = [0.1, 0.2, 0.3]
                resultados_umbral = []
                for umbral in umbrales:
                    alta_confianza = (pred_proba > umbral).sum()
                    porcentaje = (alta_confianza / len(pred_proba)) * 100
                    resultados_umbral.append({
                        'Umbral': umbral,
                        'Clientes Alta Confianza': alta_confianza,
                        'Porcentaje': f'{porcentaje:.1f}%'
                    })
            elif seleccion == "Cantidad comprada":
                # Umbrales específicos para cantidad
                umbrales = [0.25, 0.5, 0.75]
                resultados_umbral = []
                for umbral in umbrales:
                    alta_cantidad = (pred_proba > umbral).sum()
                    porcentaje = (alta_cantidad / len(pred_proba)) * 100
                    resultados_umbral.append({
                        'Umbral': umbral,
                        'Clientes Alta Cantidad': alta_cantidad,
                        'Porcentaje': f'{porcentaje:.1f}%'
                    })
            else:
                umbrales = [0.3, 0.5, 0.7, 0.9]
                resultados_umbral = []
                for umbral in umbrales:
                    compras_predichas = (pred_proba > umbral).sum()
                    porcentaje = (compras_predichas / len(pred_proba)) * 100
                    resultados_umbral.append({
                        'Umbral': umbral,
                        'Clientes que Comprarían': compras_predichas,
                        'Porcentaje': f'{porcentaje:.1f}%'
                    })

            df_umbrales = pd.DataFrame(resultados_umbral)
            st.dataframe(df_umbrales)

            # SEGMENTACIÓN DE CLIENTES
            st.write("### Segmentación de Clientes")
            
            if seleccion == "Día de compra":
                segmentos = [
                    (0.0, 0.2, 'Muy Probable Lunes'),
                    (0.2, 0.4, 'Probable Lunes'), 
                    (0.4, 0.6, 'Indeciso'),
                    (0.6, 0.8, 'Probable Martes'),
                    (0.8, 1.01, 'Muy Probable Martes')
                ]
            elif seleccion == "Producto comprado":
                segmentos = [
                    (0.0, 0.1, 'Muy Baja Confianza'),
                    (0.1, 0.2, 'Baja Confianza'), 
                    (0.2, 0.4, 'Confianza Media'),
                    (0.4, 0.7, 'Alta Confianza'),
                    (0.7, 1.01, 'Muy Alta Confianza')
                ]
            elif seleccion == "Cantidad comprada":
                segmentos = [
                    (0.0, 0.2, 'Baja Cantidad'),
                    (0.2, 0.4, 'Cantidad Media-Baja'), 
                    (0.4, 0.6, 'Cantidad Media'),
                    (0.6, 0.8, 'Cantidad Media-Alta'),
                    (0.8, 1.01, 'Alta Cantidad')
                ]
            else:
                segmentos = [
                    (0.0, 0.3, 'Baja Probabilidad'),
                    (0.3, 0.7, 'Media Probabilidad'), 
                    (0.7, 0.9, 'Alta Probabilidad'),
                    (0.9, 1.01, 'Muy Alta Probabilidad')
                ]

            datos_segmentos = []
            for min_p, max_p, nombre in segmentos:
                count = ((pred_proba >= min_p) & (pred_proba < max_p)).sum()
                porcentaje = (count / len(pred_proba)) * 100
                datos_segmentos.append({
                    'Segmento': nombre,
                    'Clientes': count,
                    'Porcentaje': f'{porcentaje:.1f}%'
                })

            df_segmentos = pd.DataFrame(datos_segmentos)
            st.dataframe(df_segmentos)

            # GRÁFICO DE SEGMENTOS
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.bar(df_segmentos['Segmento'], df_segmentos['Clientes'], 
                   color=['#ff6b6b', '#ffd166', '#06d6a0', '#118ab2', '#073b4c'][:len(segmentos)])
            ax2.set_ylabel('Número de Clientes')
            ax2.set_title('Distribución de Clientes por Segmento')
            ax2.tick_params(axis='x', rotation=45)
            
            for i, v in enumerate(df_segmentos['Clientes']):
                ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')
                
            plt.tight_layout()
            st.pyplot(fig2)

            # IMPORTANCIA DE CARACTERÍSTICAS (si está disponible)
            if hasattr(modelo, 'feature_importances_'):
                st.write("### Importancia de Características")
                
                importancia = modelo.feature_importances_
                caracteristicas = datos_muestra.columns
                
                df_importancia = pd.DataFrame({
                    'Característica': caracteristicas,
                    'Importancia': importancia
                }).sort_values('Importancia', ascending=False)
                
                st.dataframe(df_importancia)
                
                # Gráfico de importancia
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.barh(df_importancia['Característica'], df_importancia['Importancia'])
                ax3.set_xlabel('Importancia')
                ax3.set_title('Importancia de Características en el Modelo')
                plt.tight_layout()
                st.pyplot(fig3)

            # RESUMEN EJECUTIVO
            st.write("### Resumen Ejecutivo")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Métricas Clave:**")
                st.write(f"- Total clientes analizados: {len(pred_proba)}")
                
                if seleccion == "Día de compra":
                    martes_predichos = (pred_proba > 0.5).sum()
                    st.write(f"- Clientes que comprarán Martes: {martes_predichos} ({martes_predichos/len(pred_proba)*100:.1f}%)")
                    st.write(f"- Probabilidad promedio Martes: {pred_proba.mean():.1%}")
                elif seleccion == "Probabilidad de compra":
                    alta_prob = (pred_proba > 0.7).sum()
                    st.write(f"- Clientes con alta probabilidad (>70%): {alta_prob} ({alta_prob/len(pred_proba)*100:.1f}%)")
                    st.write(f"- Confianza promedio: {pred_proba.mean():.1%}")
                elif seleccion == "Producto comprado":
                    alta_conf = (pred_proba > 0.4).sum()
                    st.write(f"- Clientes con alta confianza (>40%): {alta_conf} ({alta_conf/len(pred_proba)*100:.1f}%)")
                    st.write(f"- Confianza promedio: {pred_proba.mean():.1%}")
                elif seleccion == "Cantidad comprada":
                    alta_cant = (pred_proba > 0.5).sum()
                    st.write(f"- Clientes con alta cantidad (>50%): {alta_cant} ({alta_cant/len(pred_proba)*100:.1f}%)")
                    st.write(f"- Cantidad promedio: {pred_proba.mean():.1%}")
                else:
                    st.write(f"- Clientes objetivo: {len(pred_proba)}")
                    st.write(f"- Probabilidad promedio: {pred_proba.mean():.1%}")
                
            with col2:
                st.write("**Evaluación del Modelo:**")
                st.write(f"- Variabilidad: {'Alta' if pred_proba.std() > 0.1 else 'Moderada' if pred_proba.std() > 0.05 else 'Baja'}")
                st.write(f"- Rango de predicciones: {pred_proba.max() - pred_proba.min():.3f}")
                
                if seleccion == "Día de compra":
                    muy_martes = (pred_proba > 0.8).sum()
                    st.write(f"- Clientes muy seguros de Martes (>80%): {muy_martes}")
                elif seleccion == "Producto comprado":
                    muy_seguros = (pred_proba > 0.7).sum()
                    st.write(f"- Clientes muy seguros (>70%): {muy_seguros}")
                elif seleccion == "Cantidad comprada":
                    muy_alta_cant = (pred_proba > 0.8).sum()
                    st.write(f"- Clientes con muy alta cantidad (>80%): {muy_alta_cant}")
                else:
                    muy_seguros = (pred_proba > 0.9).sum()
                    st.write(f"- Clientes muy seguros (>90%): {muy_seguros}")

        except FileNotFoundError:
            st.error(f"No se encontró el archivo: {modelo_path}")
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
            st.write("Detalles del error:", str(e))
