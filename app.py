from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and encoder
model = joblib.load('irrigation_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        temp = float(data['temp'])
        hum = float(data['hum'])
        soil = int(data['soil'])
        ldr = int(data['ldr'])

        features = np.array([[temp, hum, soil, ldr]])
        prediction_encoded = model.predict(features)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)

        return jsonify({'prediction': prediction_label[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)