# Proyecto de Generación de Texto Automático

## Descripción
Este proyecto implementa un modelo de Machine Learning (ML) entrenado para generar texto automáticamente a través de una interfaz web simple (Flask).

## Configuración y Ejecución

1.  **Crear y activar el entorno virtual:**
    `
    python -m venv venv
    .\venv\Scripts\activate
    `

2.  **Instalar dependencias:**
    `
    pip install -r requirements.txt
    `

3.  **Descargar y colocar el Modelo:**
    * Coloca los archivos de pesos (\model_weights.h5\, etc.) en \model/trained_model/\.
    * Coloca el vocabulario/configuración del tokenizador en \model/tokenizer/\.

4.  **Ejecutar la aplicación web:**
    `
    python app\app.py
    `
    Accede a \http://127.0.0.1:5000\ en tu navegador.
