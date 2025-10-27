import time
import os
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model 

MODEL_DIR = os.path.join('model', 'trained_model')
TOKENIZER_PATH = os.path.join('model', 'tokenizer', 'tokeizer.pkl')

global_model = {}
global_tokenizer = None

def load_ml_artifacts():
    """Carga el modelo .keras / .h5 y el tokenizador una única vez."""
    global global_model
    global global_tokenizer

    try:
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as tkn:
                global_tokenizer = ´pickle.load(tkn) 
                print(f"-> Tokenizador (.pkl) cargado correctamente.")
        else:
            print(f"-> Advertencia: Tokenizador no encontrado en {TOKENIZER_PATH}. Usando dummy.")
            # Asignar un objeto simple si no se encuentra (para evitar errores, pero la inferencia fallará)
            global_tokenizer = None
    except Exception as e:
        print(f"Error al cargar tokenizador: {e}")

    for filename in os.listdir(MODEL_DIR):
        # Filtrar solo archivos de Keras/TensorFlow
        if filename.endswith(('.keras', '.h5')):
            model_path = os.path.join(MODEL_DIR, filename)
            try:
                model = load_model(model_path) 
                global_models[filename] = model
                print(f"-> Modelo '{filename}' cargado correctamente.")
            except Exception as e:
                print(f"Error al cargar '{filename}': {e}")

    return global_models, global_tokenizer

def preprocess_text(text: str, tokenizer: dict) -> tf.Tensor:
    """Convierte texto de entrada a secuencias numéricas usando el objeto Tokenizer."""
    if not tokenizer:
        raise ValueError("El objeto Tokenizer no está cargado.")

    # El método principal para tokenizar una lista de textos
    sequence = tokenizer.texts_to_sequences([text]) 
    
    # IMPORTANTE: Aquí deberías añadir el padding y el truncamiento que usaste en el entrenamiento.
    # Por ejemplo: sequence = pad_sequences(sequence, maxlen=MAX_LEN)
    
    return tf.constant(sequence, dtype=tf.int32)

def generate_text_with_model(model_name: str, input_prompt: str, models: dict, tokenizer) -> str:
    """Genera texto y lo decodifica."""
    if model_name not in models:
        return f"Error: Modelo '{model_name}' no encontrado."
    if not tokenizer:
        return "Error: Tokenizador no cargado."

    model = models[model_name]
    
    processed_input = preprocess_text(input_prompt, tokenizer)

    try:
        # En una generación real, usarías un bucle para generar token por token.
        # Aquí, simulamos una secuencia de salida para el ejemplo:
        # output_sequence = model.predict(processed_input)
        
        # Simulación: (ej. secuencia generada por el modelo)
        # Nota: Los índices deben existir en el vocabulario del tokenizador.
        simulated_output_sequence = [[1, 5, 20, 15, 2]] 
        
    except Exception as e:
        return f"Error de inferencia en {model_name}: {e}"

    # 3. Post-procesamiento (Decodificación de la secuencia)
    # El método de Keras para decodificar una lista de secuencias
    generated_text = tokenizer.sequences_to_texts(simulated_output_sequence)[0]
    
    return generated_text