# Sección: Modelos
elif menu == "Modelos":
    st.subheader("Predicción con modelos del Grupo Juyo")
    
    # Cachear datos y modelos para mejor rendimiento
    @st.cache_data
    def cargar_datos():
        df = pd.read_csv("data/features/data6_features.csv")
        # Features para diferentes modelos
        features_compro = ['frecuencia_mensual', 'antiguedad_meses', 'valor_promedio_cliente', 
                          'es_fin_de_semana', 'variabilidad_monto_cliente', 'consistencia_metodo_pago']
        return df, features_compro
    
    @st.cache_resource
    def cargar_modelo(modelo_path):
        with open(modelo_path, "rb") as f:
            return pickle.load(f)
    
    datos_completos, features_compro = cargar_datos()
    
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
            modelo = cargar_modelo(modelo_path)
            
            # Usar solo 100 filas para mejor rendimiento
            datos_muestra = datos_completos[features_compro].head(100)
            
            with st.spinner('Calculando predicciones...'):
                # PREDICCIÓN ESPECÍFICA PARA CADA MODELO
                if seleccion == "Producto comprado":
                    # Para CatBoost: convertir datos al tipo correcto y NO usar cat_features
                    datos_muestra = datos_muestra.astype({
                        'frecuencia_mensual': 'float32',
                        'antiguedad_meses': 'int32', 
                        'valor_promedio_cliente': 'float32',
                        'es_fin_de_semana': 'int32',
                        'variabilidad_monto_cliente': 'float32',
                        'consistencia_metodo_pago': 'int32'
                    })
                    pred = modelo.predict(datos_muestra)
                    pred_proba = modelo.predict_proba(datos_muestra)
                    # Para multiclase, usar la probabilidad máxima como score
                    pred_proba = np.max(pred_proba, axis=1)
                else:
                    # Para modelos binarios
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
                ax.set_xlabel('Probabilidad del Producto Predicho')
                ax.set_title('Distribución de Probabilidades del Producto')
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
                    'Producto_Predicho': pred[:10]  # Los productos ya vienen codificados
                })
                st.info("Nota: Los productos aparecen codificados (0-9) como durante el entrenamiento")
            else:
                df_predicciones = pd.DataFrame({
                    'Probabilidad': pred_proba[:10],
                    'Predicción': pred_proba[:10]
                })
            
            st.dataframe(df_predicciones)

        except FileNotFoundError:
            st.error(f"No se encontró el archivo: {modelo_path}")
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")

