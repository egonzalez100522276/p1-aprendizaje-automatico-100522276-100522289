import streamlit as st
import pandas as pd
import joblib
import random

# ==========================================
# 0. CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Despliegue del Modelo Bancario", page_icon="🏦", layout="centered")
st.title("Predicción de Subscripción Bancaria")
st.write("Aplicación interactiva para predecir si un cliente suscribirá un depósito a plazo.")

# ==========================================
# 1. CARGA DEL MODELO Y METADATOS
# ==========================================
@st.cache_resource
def load_metadata():
    return joblib.load("modelo_final.joblib")

try:
    data = load_metadata()
    pipeline = data['pipeline']
    feature_names = data['feature_names']
    target_names = data['target_names']
    cat_features = data['categorical_features']
    num_features = data['numerical_features']
    
    opciones_categoricas = data.get('category_values', {}) 
    
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# ==========================================
# 2. LÓGICA DE ALEATORIZACIÓN REALISTA
# ==========================================
baremos_numericos = {
    'age': (18, 90),
    'balance': (-500, 15000),
    'duration': (0, 1200),
    'campaign': (1, 15),
    'pdays': (-1, 30), 
    'previous': (0, 7)
}

def aleatorizar_datos():
    for feat in num_features:
        if feat == 'duration':
            st.session_state[feat] = float(random.randint(10, 180) if random.random() < 0.8 else random.randint(180, 1000))
        elif feat == 'pdays':
            st.session_state[feat] = -1.0 if random.random() < 0.95 else float(random.randint(1, 15))
        elif feat == 'previous':
            st.session_state[feat] = 0.0 if random.random() < 0.85 else float(random.randint(1, 4))
        elif feat == 'campaign':
            st.session_state[feat] = float(random.randint(1, 3) if random.random() < 0.8 else random.randint(4, 10))
        else:
            mini, maxi = baremos_numericos.get(feat, (0, 100))
            st.session_state[feat] = float(random.randint(mini, maxi))
            
    for feat in cat_features:
        opciones = opciones_categoricas.get(feat, ["unknown"])
        if feat == 'poutcome' and "nonexistent" in opciones:
            pesos = [10 if op == 'failure' else 85 if op == 'nonexistent' else 5 if op == 'success' else 1 for op in opciones]
            st.session_state[feat] = random.choices(opciones, weights=pesos)[0]
        else:
            if opciones:
                st.session_state[feat] = random.choice(opciones)

# ==========================================
# 3. INTERFAZ DE USUARIO (FORMULARIO)
# ==========================================
st.button("🎲 Aleatorizar Datos de Cliente", on_click=aleatorizar_datos)

with st.form("prediction_form"):
    st.subheader("Variables Numéricas")
    num_cols_ui = st.columns(2)
    for i, feat in enumerate(num_features):
        with num_cols_ui[i % 2]:
            st.number_input(label=feat, step=1.0, key=feat)
            
    st.subheader("Variables Categóricas")
    cat_cols_ui = st.columns(2)
    for i, feat in enumerate(cat_features):
        with cat_cols_ui[i % 2]:
            opciones = opciones_categoricas.get(feat, ["unknown"])
            st.selectbox(label=feat, options=opciones, key=feat)
            
    st.markdown("---")
    submitted = st.form_submit_button("Predecir Subscripción", use_container_width=True)

# ==========================================
# 4. PREDICCIÓN Y EXPORTACIÓN
# ==========================================
if submitted:
    # 1. Preparar DataFrame
    X_new = pd.DataFrame([{f: st.session_state.get(f, 0) for f in feature_names}])
    
    for col in num_features:
        X_new[col] = pd.to_numeric(X_new[col])
    for col in cat_features:
        X_new[col] = X_new[col].astype(str)

    try:
        # 2. Predicción
        y_pred = pipeline.predict(X_new)[0]
        
        is_yes = (y_pred == 1 or str(y_pred).lower() == 'yes')
        resultado = "YES (Suscribirá)" if is_yes else "NO (No suscribirá)"
        color = "green" if is_yes else "red"
        
        st.markdown(f"### Resultado de la predicción: :{color}[**{resultado}**]")
        
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_new)[0]
            proba_yes = proba[1] if len(proba) > 1 else proba[0]
            st.metric(label="Probabilidad de Subscripción", value=f"{proba_yes * 100:.2f}%")
            
        st.markdown("---")
        st.subheader("🛠️ Exportar para validación en Notebook")
        st.write("Copia y pega este código en tu Jupyter Notebook para validar que las predicciones coinciden:")
        
        # 3. Generar código de exportación para Notebook
        dict_str = "{\n"
        for col in feature_names:
            val = X_new[col].iloc[0]
            if isinstance(val, str):
                dict_str += f"    '{col}': ['{val}'],\n"
            else:
                dict_str += f"    '{col}': [{val}],\n"
        dict_str += "}"
        
        codigo_export = f"""import pandas as pd

# 1. Crear el DataFrame con la instancia exacta
X_prueba = pd.DataFrame({dict_str})

# 2. Hacer la predicción (asumiendo que tu pipeline se llama pipeline_modelo_final)
prediccion = pipeline_modelo_final.predict(X_prueba)
print("Predicción para esta instancia:", prediccion)
"""
        st.code(codigo_export, language="python")
        
        # 4. Botón de descarga CSV
        csv_data = X_new.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 O descargar instancia como CSV",
            data=csv_data,
            file_name="instancia_prueba.csv",
            mime="text/csv"
        )
            
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")