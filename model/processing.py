import os
import tensorflow as tf
from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences
import pickle 
import numpy as np
import time
import re

# --- CONSTANTES EXTRAÍDAS DE TU COLAB ---
MAX_LEN = 80
MAX_NEW_TOKENS = 60
TOP_K = 50
TEMPERATURE = 0.9
REPETITION_PENALTY = 1.2
PADDING_ID = 0

# Rutas
# Nota: Las rutas son relativas al directorio principal del proyecto (donde está 'model')
MODEL_DIR = os.path.join('model', 'trained_model')
TOKENIZER_PATH = os.path.join('model', 'tokenizer', 'tokenizer.pkl') 

# Variables globales para los artefactos ML
global_models = {} 
global_tokenizer = None

# ------------------------------------
# A. CARGA DE ARTEFACTOS
# ------------------------------------

def load_ml_artifacts():
    """Carga todos los modelos .keras/.h5 y el tokenizador .pkl."""
    global global_models
    global global_tokenizer

    # --- DEBUGGING DE RUTAS ---
    print("\n--- DEBUGGING DE ARTEFACTOS ML ---")
    print(f"Buscando Modelos en: {os.path.abspath(MODEL_DIR)}")
    print(f"Buscando Tokenizador en: {os.path.abspath(TOKENIZER_PATH)}")
    print("---------------------------------")
    
    # 1. Cargar Tokenizador (.pkl)
    try:
        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, 'rb') as handle: 
                global_tokenizer = pickle.load(handle) 
            print(" Tokenizador (.pkl) cargado correctamente.")
        else:
            print(" ERROR FATAL: Tokenizador no encontrado.")
            
    except Exception as e:
        print(f" ERROR al cargar tokenizador .pkl: {e}")
        
    # 2. Cargar Múltiples Modelos
    try:
        archivos_en_dir = os.listdir(MODEL_DIR)
        modelos_encontrados = [f for f in archivos_en_dir if f.endswith(('.keras', '.h5'))]
        
        if not modelos_encontrados:
            print(f"⚠️ Advertencia: No se encontraron archivos .keras o .h5 en {MODEL_DIR}.")
            return global_models, global_tokenizer # Devuelve lo que tiene hasta ahora

        for filename in modelos_encontrados:
            model_path = os.path.join(MODEL_DIR, filename)
            try:
                # Cargar sin compilar si es solo para inferencia
                model = load_model(model_path, compile=False) 
                global_models[filename] = model
                print(f" Modelo '{filename}' cargado correctamente.")
            except Exception as e:
                # Esto captura errores internos de Keras al intentar leer el archivo
                print(f" ERROR al cargar '{filename}'. Posible corrupción del archivo. Error: {e}")

    except FileNotFoundError:
        print(f" ERROR FATAL: Directorio de modelos no encontrado: {MODEL_DIR}")
    except Exception as e:
        print(f" ERROR inesperado durante la carga de modelos: {e}")


    return global_models, global_tokenizer

# ------------------------------------
# B. PRE-PROCESAMIENTO
# ------------------------------------

def preprocess_text(text: str, tokenizer) -> tf.Tensor:
    """Convierte texto de entrada a secuencias numéricas y aplica padding."""
    if not tokenizer:
        raise ValueError("El objeto Tokenizer no está cargado.")
        
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    sequence = tokenizer.texts_to_sequences([text]) 
    
    sequence_padded = pad_sequences(
        sequence, 
        maxlen=MAX_LEN, 
        padding='pre',      
        truncating='pre'    
    )
    
    return tf.constant(sequence_padded, dtype=tf.int32) 

# ------------------------------------
# C. INFERENCIA Y POST-PROCESAMIENTO
# ------------------------------------

def generate_text_with_model(model_name: str, input_prompt: str, models: dict, tokenizer) -> str:
    """
    Ejecuta el bucle de generación secuencial con Top-K, Temperatura y Repetition Penalty.
    """
    # Verificación de carga. Si el tokenizador es None, esto fallará.
    if model_name not in models or tokenizer is None:
        raise ValueError("Modelo o Tokenizador no cargado o no encontrado.")

    model = models[model_name]
    
    generated_text = input_prompt
    generated_words = input_prompt.lower().split()

    for _ in range(MAX_NEW_TOKENS):
        
        processed_input = preprocess_text(generated_text, tokenizer)
        
        # ... (resto de la lógica de generación)
        predictions = model.predict(processed_input, verbose=0)[0]
        
        current_probs = predictions.copy()
        for word in set(generated_words[-25:]):
            word_idx = tokenizer.word_index.get(word, 0)
            if 0 < word_idx < len(current_probs):
                current_probs[word_idx] /= REPETITION_PENALTY
        
        current_probs = np.log(current_probs + 1e-10) / TEMPERATURE
        current_probs = np.exp(current_probs)
        current_probs = current_probs / np.sum(current_probs) 
        
        top_k_indices = np.argsort(current_probs)[-TOP_K:]
        top_k_probs = current_probs[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs) 
        
        predicted_index = np.random.choice(top_k_indices, p=top_k_probs)

        if predicted_index <= PADDING_ID:
            continue
            
        output_word = tokenizer.index_word.get(predicted_index, "")
        if not output_word:
            continue

        generated_text += " " + output_word
        generated_words.append(output_word)
        
    return generated_text
