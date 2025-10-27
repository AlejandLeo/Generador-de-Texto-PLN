# app/app.py
from flask import Flask, request, render_template
import os
import sys

# Añadir el directorio superior al path para importar el módulo 'model'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from model.processing import load_ml_artifacts, generate_text_with_model

app = Flask(__name__)

# Referencias globales para el modelo ML y tokenizador
global_ml_models, global_tokenizer = load_ml_artifacts()

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = None
    input_prompt = ""
    selected_model_name = ""

    # Nombres de modelos disponibles para el selector del frontend
    available_models = list(global_ml_models.keys())

    if not global_ml_models:
        return render_template('index.html', error="Error: No se cargó ningún modelo (.keras o .h5).", available_models=[])

    if request.method == 'POST':
        input_prompt = request.form.get('prompt', '').strip()
        selected_model_name = request.form.get('model_select', available_models[0])

        if input_prompt and selected_model_name:
            try:
                generated_text = generate_text_with_model(
                    selected_model_name, 
                    input_prompt, 
                    global_ml_models, 
                    global_tokenizer
                )
            except Exception as e:
                generated_text = f"Error durante la generación con {selected_model_name}: {e}"

    return render_template(
        'index.html', 
        generated_text=generated_text, 
        input_prompt=input_prompt,
        available_models=available_models,
        selected_model_name=selected_model_name
    )

if __name__ == '__main__':
    app.run(debug=True)
