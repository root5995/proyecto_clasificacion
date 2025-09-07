# EJECUTAR ESTO PRIMERO EN CONSOLA
# Si da error, debes ir a PowerShell de Window y:
#     Get-ExecutionPolicy                               Si es Restricted; ejecuta
#     Set-ExecutionPolicy RemoteSigned                  Colocar Sí
# Luego de usar este script; ir a PowerShell:    Set-ExecutionPolicy Restricted

# Crea un ambiente virtual (puedes usar otro nombre en lugar de 'venv')
#    python -m venv venv
#    .\venv\Scripts\activate   # En Windows

# Instala las dependencias necesarias
#    pip install streamlit pandas joblib scikit-learn

# -------------------------------------------------------------------------------------------------
# Desde la segunda vez: hacer:
# Si da error, debes ir a PowerShell de Window y:
#     Get-ExecutionPolicy                               Si es Restricted; ejecuta
#     Set-ExecutionPolicy RemoteSigned                  Colocar Sí
# En consola de VSC:  .\venv\Scripts\activate

import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# -------------------------PROCESO DE DESPLIEGUE------------------------------

# 01 --------------------------Cargar el modelo-------------------------------------------
try:
    clf = load('modelo_xgb_pipeline.joblib')
except FileNotFoundError:
    st.error("Error: El archivo del modelo 'modelo_diabetes_pipeline.joblib' no se encontró. Por favor, asegúrate de que el modelo entrenado esté en la misma carpeta que este script.")
    st.stop()

# 02---------------- Variables globales para los campos del formulario-----------------------

# Opciones para las variables categóricas
sexo_options = ["Masculino", "Femenino"]
tipo_dolor_toracico_options = ["Angina típica", "Angina atípica", "Dolor no anginoso", "Asintomático"]
angina_ejercicio_options = ["No", "Sí"]
pendiente_st_options = ["Creciente", "Plana", "Decreciente"]

# Mapeos a valores numéricos que inician desde 0, según tus datos.
# El orden en la lista `_options` determina el valor numérico.
sexo_map = {option: i for i, option in enumerate(sexo_options)}
tipo_dolor_toracico_map = {option: i for i, option in enumerate(tipo_dolor_toracico_options)}
angina_ejercicio_map = {option: i for i, option in enumerate(angina_ejercicio_options)}
pendiente_st_map = {option: i for i, option in enumerate(pendiente_st_options)}

# Inicializar valores de las variables
edad = 50.0
presion_arterial = 120.0
colesterol = 200.0
frecuencia_cardiaca = 150.0
oldpeak = 1.0

# 03 Reseteo------------- Flag para seguir errores---------------------------------------
error_flag = False

# Función para resetear los inputs
def reset_inputs():
    global edad, presion_arterial, colesterol, frecuencia_cardiaca, oldpeak, error_flag
    edad = 50.0
    presion_arterial = 120.0
    colesterol = 200.0
    frecuencia_cardiaca = 150.0
    oldpeak = 1.0
    error_flag = False

# Inicializar variables
# reset_inputs() # No es necesario llamar aquí, Streamlit maneja el estado.
# -----------------------------------------------------------------------------------------------

# ------------------------Título centrado-------------------------------------------------
st.title("Modelo Predictivo de Enfermedad Cardíaca")
st.markdown("Este modelo predice la probabilidad de que una persona tenga enfermedad cardíaca en base a diferentes características médicas.")
st.markdown("---")

# ----------------------- Función para validar los campos del formulario----------------------------
def validate_inputs(edad, presion_arterial, colesterol, frecuencia_cardiaca, oldpeak):
    global error_flag
    if edad < 0 or presion_arterial < 0 or colesterol < 0 or frecuencia_cardiaca < 0 or oldpeak < 0:
        st.error("No se permiten valores negativos. Por favor, ingrese valores válidos.")
        error_flag = True
    else:
        error_flag = False

# ------------------------------------ Formulario en dos columnas------------------------------------
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)

    # Campos de entrada en la primera columna
    with col1:
        edad = st.number_input("**Edad**", min_value=0.0, value=edad, step=1.0)
        presion_arterial = st.number_input("**Presión arterial en reposo**", min_value=0.0, value=presion_arterial, step=1.0)
        colesterol = st.number_input("**Colesterol**", min_value=0.0, value=colesterol, step=1.0)
        frecuencia_cardiaca = st.number_input("**Frecuencia cardíaca máxima**", min_value=0.0, value=frecuencia_cardiaca, step=1.0)
        oldpeak = st.number_input("**Oldpeak (depresión del ST)**", min_value=0.0, value=oldpeak, step=0.1)

    # Campos de entrada en la segunda columna
    with col2:
        sexo = st.selectbox("**Sexo**", sexo_options)
        tipo_dolor_toracico = st.selectbox("**Tipo de dolor torácico**", tipo_dolor_toracico_options)
        angina_ejercicio = st.selectbox("**Angina inducida por ejercicio**", angina_ejercicio_options)
        pendiente_st = st.selectbox("**Pendiente del segmento ST**", pendiente_st_options)
        
    # Botón de Predecir
    predict_button = st.form_submit_button("Predecir")

# Validar y ejecutar predicción
if predict_button:
    validate_inputs(edad, presion_arterial, colesterol, frecuencia_cardiaca, oldpeak)

    if not error_flag:
        # Crear DataFrame con los datos de entrada
        data = {
            'edad': [edad],
            'sexo': [sexo_map[sexo]],
            'tipo de dolor torácico': [tipo_dolor_toracico_map[tipo_dolor_toracico]],
            'presión arterial en reposo': [presion_arterial],
            'colesterol': [colesterol],
            'frecuencia cardíaca máxima': [frecuencia_cardiaca],
            'angina inducida por ejercicio': [angina_ejercicio_map[angina_ejercicio]],
            'pendiente del segmento ST': [pendiente_st_map[pendiente_st]],
            'oldpeak': [oldpeak]
        }
        df = pd.DataFrame(data)

                # Forzar los tipos de datos para que coincidan con los del modelo entrenado
        df = df.astype({
            'edad': 'float64',
            'sexo': 'str',
            'tipo de dolor torácico': 'str',
            'presión arterial en reposo': 'float64',
            'colesterol': 'float64',
            'frecuencia cardíaca máxima': 'float64',
            'angina inducida por ejercicio': 'str',
            'pendiente del segmento ST': 'str',
            'oldpeak': 'float64'
        })

        # Convertir las columnas categóricas a tipo de dato 'object'
        #cols_to_object = ['sexo', 'tipo de dolor torácico', 'angina inducida por ejercicio', 'pendiente del segmento ST']
        #df[cols_to_object] = df[cols_to_object].astype("object")

        # Mostrar los datos de entrada y sus tipos de datos
        st.subheader("Datos de entrada para la predicción")
        st.dataframe(df)
        st.write("Tipos de datos:", df.dtypes)

        # Realizar la predicción
        probabilities_classes = clf.predict_proba(df)[0]
        class_predicted = np.argmax(probabilities_classes)

        # Mostrar resultado con estilo personalizado
        if class_predicted == 0:
            outcome = "Sin Enfermedad Cardíaca"
            probability = probabilities_classes[0]
            style_result = 'background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px;'
        else:
            outcome = "Con Enfermedad Cardíaca"
            probability = probabilities_classes[1]
            style_result = 'background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;'

        result_html = f"<div style='{style_result}'>**Resultado:** '{outcome}' con una probabilidad del {round(float(probability * 100), 2)}%</div>"
        st.markdown(result_html, unsafe_allow_html=True)

# --------------------------- Botón de Resetear-------------------------------------
if st.button("Resetear campos"):
    reset_inputs()