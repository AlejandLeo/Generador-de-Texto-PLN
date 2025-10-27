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
