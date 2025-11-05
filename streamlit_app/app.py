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
    
    # Cachear datos y modelos para mejor rendimiento
    @st.cache_data
    def cargar_datos():
        df = pd.read_csv("data/features/data6_features.csv")
        
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
            'cantidad': ['frecuencia_mensual', 'antiguedad_meses', 'valor_promedio_cliente', 
                        'es_fin_de_semana', 'variabilidad_monto_cliente', 'consistencia_metodo_pago']
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
                modelo = cargar_modelo(modelo_path)
                with st.spinner('Calculando predicciones...'):
                    pred = modelo.predict(datos_muestra)
                    if hasattr(modelo, 'predict_proba'):
                        pred_proba = modelo.predict_proba(datos_muestra)[:, 1]
                    else:
                        pred_proba = pred

            elif seleccion == "Día de compra":
                datos_muestra = datos_completos[features_por_modelo['dia_compra']].head(100)
                modelo = cargar_modelo(modelo_path)
                with st.spinner('Calculando predicciones...'):
                    pred = modelo.predict(datos_muestra)
                    if hasattr(modelo, 'predict_proba'):
                        pred_proba = modelo.predict_proba(datos_muestra)[:, 1]
                    else:
                        pred_proba = pred

            elif seleccion == "Producto comprado":
                # Cargar modelo COMPLETO con cat_features
                modelo_data = cargar_modelo(modelo_path)
                
                # Verificar estructura
                if not isinstance(modelo_data, dict) or 'model' not in modelo_data:
                    st.error("Error: El modelo no tiene la estructura esperada")
                    return
                    
                modelo = modelo_data['model']
                cat_features = modelo_data['cat_features']
                features_esperadas = modelo_data['features']
                
                # Usar las features ESPECÍFICAS del modelo guardado
                datos_muestra = datos_completos[features_esperadas].head(100)
                
                # Asegurar tipos de datos correctos
                datos_muestra = datos_muestra.astype({
                    'tipo_cliente': 'int32',
                    'frecuencia_mensual': 'float32',
                    'valor_promedio_cliente': 'float32', 
                    'producto_favorito_cliente': 'int32',
                    'metodo_pago_habitual': 'int32',
                    'segmento_comportamiento': 'int32',
                    'antiguedad_meses': 'int32',
                    'dia_semana_num': 'int32',
                    'estacion': 'int32'
                })
                
                # Predecir CON cat_features
                with st.spinner('Calculando predicciones de producto...'):
                    pred = modelo.predict(datos_muestra, cat_features=cat_features)
                    pred_proba = modelo.predict_proba(datos_muestra, cat_features=cat_features)
                    # Para multiclase, usar la probabilidad máxima como score
                    pred_proba = np.max(pred_proba, axis=1)

            elif seleccion == "Cantidad comprada":
                datos_muestra = datos_completos[features_por_modelo['cantidad']].head(100)
                modelo = cargar_modelo(modelo_path)
                with st.spinner('Calculando predicciones...'):
                    pred = modelo.predict(datos_muestra)
                    if hasattr(modelo, 'predict_proba'):
                        pred_proba = modelo.predict_proba(datos_muestra)[:, 1]
                    else:
                        pred_proba = pred

            st.success("Predicciones completadas")
            
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
                else:
                    muy_seguros = (pred_proba > 0.9).sum()
                    st.write(f"- Clientes muy seguros (>90%): {muy_seguros}")

            # PREDICCIONES INDIVIDUALES
            st.write("### Predicciones Individuales (primeras 10)")
            
            if seleccion == "Día de compra":
                df_predicciones = pd.DataFrame({
                    'Probabilidad_Martes': pred_proba[:10],
                    'Día_Predicho': ['MARTES' if p > 0.5 else 'LUNES' for p in pred_proba[:10]]
                })
            elif seleccion == "Probabilidad de compra":
                df_predicciones = pd.DataFrame({
                    'Probabilidad_Compra': pred_proba[:10],
                    'Decisión': ['COMPRA' if p > 0.5 else 'NO COMPRA' for p in pred_proba[:10]]
                })
            elif seleccion == "Producto comprado":
                df_predicciones = pd.DataFrame({
                    'Confianza_Predicción': pred_proba[:10],
                    'Producto_Predicho': pred[:10]
                })
                st.info("Nota: Los productos aparecen codificados (0-9) como durante el entrenamiento")
            else:
                df_predicciones = pd.DataFrame({
                    'Probabilidad': pred_proba[:10],
                    'Predicción': pred[:10]
                })
            
            st.dataframe(df_predicciones)

        except FileNotFoundError:
            st.error(f"No se encontró el archivo: {modelo_path}")
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")
