# app.py
import base64
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)



# Carga tu modelo CNN (ajusta el nombre y la ruta del modelo)
model = tf.keras.models.load_model('MejorModelo.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Recibe la imagen en base64
    image_data = request.json.get('image')
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    # Decodifica la imagen
    image_bytes = io.BytesIO(base64.b64decode(image_data.split(',')[1]))
    image = Image.open(image_bytes).convert('RGB')
    
    # Preprocesa la imagen (ajusta según los requisitos de tu modelo)
    image = image.resize((128, 128))  # Ejemplo: redimensionar a 224x224
    image_array = np.array(image) / 255.0  # Normalizar entre 0 y 1
    image_array = np.expand_dims(image_array, axis=0)  # Añadir dimensión batch

    # Predice con el modelo
    predictions = model.predict(image_array)
    result = np.argmax(predictions, axis=1).tolist()

    return jsonify({'prediction': result[0]})  # Devuelve el resultado

if __name__ == '__main__':
    app.run(debug=True)
