# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  (Ejecutar esto si tienes problemas de permisos)

Write-Host "Iniciando la creación de la estructura del proyecto de Generación de Texto..."

# 1. Crear directorios principales
Write-Host "Creando directorios..."
New-Item -ItemType Directory -Force app/static
New-Item -ItemType Directory -Force app/templates
New-Item -ItemType Directory -Force model/trained_model
New-Item -ItemType Directory -Force model/tokenizer
New-Item -ItemType Directory -Force notebooks

# 2. Crear archivos de configuración base
Write-Host "Creando archivos de configuración base..."

# .gitignore
@"
# Entorno virtual
venv
env

# Archivos de Python
*.pyc
__pycache__
my_env

# Archivos del Modelo Grande (evitar subir pesos pesados)
model/trained_model/*.pt
model/trained_model/*.h5
model/trained_model/*.bin

# Logs
*.log
"@ | Set-Content -Path .gitignore

# requirements.txt
@"
# Dependencias necesarias para la aplicación
flask
numpy
transformers  # Librería para modelos de texto (Hugging Face)
torch         # Opcional: si el modelo usa PyTorch
"@ | Set-Content -Path requirements.txt

# README.md
@"
# Proyecto de Generación de Texto Automático

## Descripción
Este proyecto implementa un modelo de Machine Learning (ML) entrenado para generar texto automáticamente a través de una interfaz web simple (Flask).

## Configuración y Ejecución

1.  **Crear y activar el entorno virtual:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Descargar y colocar el Modelo:**
    * Coloca los archivos de pesos (\`model_weights.h5\`, etc.) en \`model/trained_model/\`.
    * Coloca el vocabulario/configuración del tokenizador en \`model/tokenizer/\`.

4.  **Ejecutar la aplicación web:**
    ```bash
    python app\app.py
    ```
    Accede a \`http://127.0.0.1:5000\` en tu navegador.
"@ | Set-Content -Path README.md

# LICENSE (Ejemplo de licencia MIT)
@"
MIT License

Copyright (c) 2024 [Tu Nombre/Organización]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"@ | Set-Content -Path LICENSE

# 3. Crear archivos del Modelo de Simulación
Write-Host "Creando archivos de simulación de modelo..."

# model/processing.py
@"
# model/processing.py
import time
import json
import os

# Simulación de la carga del modelo y tokenizador
global_model = None
global_tokenizer = None

def load_ml_artifacts(model_path='model/trained_model/', tokenizer_path='model/tokenizer/'):
    """Carga el modelo y el tokenizador una única vez."""
    global global_model
    global global_tokenizer
    
    # Simulación de carga del modelo (pesos)
    if os.path.exists(os.path.join(model_path, 'model_weights.h5')):
        print("-> Modelo ML cargado (simulado).")
        global_model = {"status": "ready"}
    else:
        print("-> Advertencia: Pesos del modelo no encontrados.")
        global_model = {"status": "dummy"}
        
    # Simulación de carga del tokenizador (vocabulario)
    # En un proyecto real: global_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if os.path.exists(os.path.join(tokenizer_path, 'vocab.json')):
        with open(os.path.join(tokenizer_path, 'vocab.json'), 'r') as f:
            global_tokenizer = json.load(f)
            print(f"-> Tokenizador cargado con {len(global_tokenizer)} tokens (simulado).")
    else:
        print("-> Advertencia: Vocabulario del tokenizador no encontrado.")
        global_tokenizer = {"<START>": 0, "token": 1} # Contenido dummy
    
    return global_model, global_tokenizer

def preprocess_text(text: str) -> list:
    """Convierte el texto de entrada a un formato que el modelo entiende."""
    # Simulación de tokenización
    tokens = text.lower().split()
    return tokens

def postprocess_text(tokens: list) -> str:
    """Convierte la salida del modelo a texto legible."""
    # Simulación de decodificación
    return " ".join(tokens).replace(" .", ".")

def generate_text_with_model(processed_input: list) -> str:
    """Simula la generación de texto usando el modelo cargado."""
    print(f"-> Generando texto para input procesado: {processed_input}")
    
    time.sleep(1) # Simula el tiempo de ejecución del modelo

    # Lógica de generación simulada (añade contenido)
    generated_tokens = processed_input + ["y", "el", "modelo", "generó", "este", "texto", "coherente", "."]
    
    final_text = postprocess_text(generated_tokens)
    return final_text
"@ | Set-Content -Path model/processing.py

# Crear archivos dummy del modelo para evitar fallos de ruta al cargar
@"
{"hola": 1, "mundo": 2, "texto": 3}
"@ | Set-Content -Path model/tokenizer/vocab.json

# Crear archivo de pesos vacío (para simular existencia)
New-Item -Path model/trained_model/model_weights.h5 -ItemType File

# 4. Crear archivos de la Aplicación Web (Flask)
Write-Host "Creando archivos de la aplicación web (Flask)..."

# app/app.py
@"
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
    # Asegúrate de haber instalado 'flask' (pip install flask)
    app.run(debug=True)
"@ | Set-Content -Path app/app.py

# app/templates/index.html
@"
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generador de Texto ML</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f7f6; }
        .container { max-width: 800px; margin: auto; background-color: #fff; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
        h1 { color: #1a73e8; text-align: center; }
        textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; min-height: 100px; }
        button { padding: 10px 15px; background-color: #1a73e8; color: white; border: none; border-radius: 4px; cursor: pointer; }
        .result-box { margin-top: 20px; padding: 15px; background-color: #e8f0fe; border-left: 4px solid #1a73e8; border-radius: 4px; }
        .error-box { background-color: #fce8e6; border-left: 4px solid #d93025; color: #d93025; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Generador de Texto con ML</h1>
        
        {% if error %}
        <div class="error-box">
            <strong>Error:</strong> {{ error }}
        </div>
        {% endif %}

        <form method="POST">
            <label for="prompt">Ingresa tu prompt o inicio de texto:</label><br>
            <textarea id="prompt" name="prompt" placeholder="Escribe aquí el texto inicial..." required>{{ input_prompt }}</textarea><br>
            <button type="submit">Generar Texto</button>
        </form>

        {% if generated_text %}
        <div class="result-box">
            <h2>Resultado de la Generación:</h2>
            <p>{{ generated_text }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"@ | Set-Content -Path app/templates/index.html

# 5. Crear __init__.py (para que Python trate 'app' como un módulo)
New-Item -Path app/__init__.py -ItemType File
New-Item -Path model/__init__.py -ItemType File # Opcional, pero buena práctica

Write-Host "Estructura de proyecto creada exitosamente en la carpeta actual."
Write-Host "Siguiente paso: pip install -r requirements.txt y python app\app.py"