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
    # Asegúrate de que este nombre coincide con tu archivo generado en el Notebook
    return joblib.load("modelo_final.joblib")

try:
    data = load_metadata()
    pipeline = data['pipeline']
    feature_names = data['feature_names']
    target_names = data['target_names']
    cat_features = data['categorical_features']
    num_features = data['numerical_features']
    
    # Recuperamos el diccionario real de categorías que guardaste en el Notebook
    opciones_categoricas = data.get('category_values', {}) 
    
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}. Asegúrate de que 'modelo_final.joblib' está en la carpeta y lo guardaste con el diccionario 'category_values'.")
    st.stop()

# ==========================================
# 2. LÓGICA DE ALEATORIZACIÓN REALISTA
# ==========================================
# Baremos aproximados para que los inputs numéricos tengan sentido
baremos_numericos = {
    'age': (18, 90),
    'balance': (-500, 15000),
    'duration': (0, 1200),
    'campaign': (1, 15),
    'pdays': (-1, 30), 
    'previous': (0, 7)
}

def aleatorizar_datos():
    # Numéricas más conservadoras (para que el modelo pueda decir que NO)
    for feat in num_features:
        if feat == 'duration':
            # 80% de las veces la llamada dura menos de 3 minutos (180s)
            st.session_state[feat] = float(random.randint(10, 180) if random.random() < 0.8 else random.randint(180, 1000))
        elif feat == 'pdays':
            # 95% de las veces es -1 en la vida real (sin contacto previo)
            st.session_state[feat] = -1.0 if random.random() < 0.95 else float(random.randint(1, 15))
        elif feat == 'previous':
            st.session_state[feat] = 0.0 if random.random() < 0.85 else float(random.randint(1, 4))
        elif feat == 'campaign':
            # La mayoría de veces se contacta 1 o 2 veces
            st.session_state[feat] = float(random.randint(1, 3) if random.random() < 0.8 else random.randint(4, 10))
        else:
            mini, maxi = baremos_numericos.get(feat, (0, 100))
            st.session_state[feat] = float(random.randint(mini, maxi))
            
    # Categóricas más conservadoras
    for feat in cat_features:
        opciones = opciones_categoricas.get(feat, ["unknown"])
        if feat == 'poutcome' and "nonexistent" in opciones:
            # Forzamos que 'nonexistent' sea lo más común (pesos para failure, nonexistent, success)
            pesos = [10 if op == 'failure' else 85 if op == 'nonexistent' else 5 if op == 'success' else 1 for op in opciones]
            st.session_state[feat] = random.choices(opciones, weights=pesos)[0]
        else:
            # Para el resto, elegimos al azar entre las opciones disponibles reales
            if opciones:
                st.session_state[feat] = random.choice(opciones)

# ==========================================
# 3. INTERFAZ DE USUARIO (FORMULARIO)
# ==========================================
# Botón para inyectar datos aleatorios en el session_state
st.button("🎲 Aleatorizar Datos de Cliente", on_click=aleatorizar_datos)

with st.form("prediction_form"):
    
    st.subheader("Variables Numéricas")
    num_cols_ui = st.columns(2)
    for i, feat in enumerate(num_features):
        with num_cols_ui[i % 2]:
            st.number_input(label=feat, step=1.0, key=feat) # Vinculado al session_state
            
    st.subheader("Variables Categóricas")
    cat_cols_ui = st.columns(2)
    for i, feat in enumerate(cat_features):
        with cat_cols_ui[i % 2]:
            # Extrae las opciones reales que sacamos del Notebook
            opciones = opciones_categoricas.get(feat, ["unknown"])
            st.selectbox(label=feat, options=opciones, key=feat) # Vinculado al session_state
            
    st.markdown("---")
    submitted = st.form_submit_button("Predecir Subscripción", use_container_width=True)

# ==========================================
# 4. PREDICCIÓN Y RESULTADOS
# ==========================================
if submitted:
    # 1. Recogemos valores directamente de la memoria de la app respetando el orden exacto de entrenamiento
    X_new = pd.DataFrame([{f: st.session_state.get(f, 0) for f in feature_names}])
    
    # 2. Forzamos los tipos de datos correctos (evita fallos silenciosos de scikit-learn)
    for col in num_features:
        X_new[col] = pd.to_numeric(X_new[col])
    for col in cat_features:
        X_new[col] = X_new[col].astype(str)
        
    # --- BLOQUE DEBUG Opcional (Coméntalo cuando vayas a entregar) ---
    with st.expander("Ver datos enviados al modelo (Debug)"):
        st.dataframe(X_new)
    # -----------------------------------------------------------------

    try:
        # 3. Predicción
        y_pred = pipeline.predict(X_new)[0]
        
        # Evaluamos el resultado. Comprobamos si el modelo devuelve 1/'yes' o 0/'no'
        is_yes = (y_pred == 1 or str(y_pred).lower() == 'yes')
        resultado = "YES (Suscribirá)" if is_yes else "NO (No suscribirá)"
        color = "green" if is_yes else "red"
        
        st.markdown(f"### Resultado de la predicción: :{color}[**{resultado}**]")
        
        # 4. Probabilidades (si el modelo lo soporta)
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_new)[0]
            # Si el array tiene más de una clase, asumimos que el índice 1 es la probabilidad de "Yes"
            proba_yes = proba[1] if len(proba) > 1 else proba[0]
            st.metric(label="Probabilidad de Subscripción", value=f"{proba_yes * 100:.2f}%")
            
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")