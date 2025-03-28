from flask import Flask, request, jsonify
from flask_cors import CORS
from heatwave_model import HeatwavePredictionModel
import os

app = Flask(__name__)
CORS(app)

# Initialize model
model = HeatwavePredictionModel()
model.model_file = os.path.join('model_files', 'heatwave_model.pkl')
model.scaler_file = os.path.join('model_files', 'heatwave_scaler.pkl')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'city' not in data:
            return jsonify({"error": "City parameter is required"}), 400
        
        city = data['city']
        forecast = model.predict_heatwave(city)
        
        return jsonify({
            "city": city,
            "predictions": forecast.to_dict('records'),
            "message": "Forecast generated successfully",
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)