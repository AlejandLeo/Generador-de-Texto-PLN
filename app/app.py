# app/app.py
from flask import Flask, request, render_template
import os
import sys

# Añadir el directorio superior al path para importar el módulo 'model'
# Necesario para que el import 'from model.processing' funcione correctamente
# en entornos de ejecución estándar.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.processing import load_ml_artifacts, generate_text_with_model, preprocess_text

app = Flask(__name__)

# Referencias globales para el modelo ML y tokenizador
global_ml_model, global_tokenizer = load_ml_artifacts()

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = None
    input_prompt = ""
    
    # Verifica que al menos la simulación de carga fue exitosa
    if global_ml_model is None:
        return render_template('index.html', error="Error grave: No se pudieron inicializar los artefactos del ML.")

    if request.method == 'POST':
        # Obtener el texto del formulario
        input_prompt = request.form.get('prompt', '').strip()
        
        if input_prompt:
            try:
                # 1. Pre-procesamiento
                processed_input = preprocess_text(input_prompt)
                
                # 2. Inferencia y Generación
                # La simulación no necesita pasar el modelo, pero en real sí se usa la referencia
                generated_text = generate_text_with_model(processed_input)
                
            except Exception as e:
                generated_text = f"Error durante la generación: {e}"

    return render_template('index.html', generated_text=generated_text, input_prompt=input_prompt)

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    app.run(debug=True)
