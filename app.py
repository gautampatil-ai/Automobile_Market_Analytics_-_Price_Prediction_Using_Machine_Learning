import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load Model
MODEL_PATH = "Random_Forest_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# HTML Template with Embedded CSS Styling and Animations
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Vehicle Valuation Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-glow: #00f2fe;
            --secondary-glow: #4facfe;
            --bg-dark: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: radial-gradient(circle at top left, #1e1b4b, #0f172a);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            overflow-x: hidden;
        }

        /* Ambient Glow Background Effect */
        body::before {
            content: '';
            position: absolute;
            width: 300px;
            height: 300px;
            background: linear-gradient(135deg, var(--primary-glow), var(--secondary-glow));
            filter: blur(150px);
            border-radius: 50%;
            top: 10%;
            left: 15%;
            z-index: -1;
            animation: floatGlow 8s ease-in-out infinite alternate;
        }

        @keyframes floatGlow {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(50px, 50px) scale(1.2); }
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: fadeIn 1s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #00f2fe, #4facfe, #00f2fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: shine 6s linear infinite;
            background-size: 200% auto;
        }

        @keyframes shine {
            to { background-position: 200% center; }
        }

        .header p {
            color: var(--text-sub);
            margin-top: 0.5rem;
            font-size: 0.95rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.2rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-sub);
            margin-bottom: 0.4rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-group input {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus {
            border-color: var(--primary-glow);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.3);
            background: rgba(15, 23, 42, 0.8);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1.5rem;
            padding: 1rem;
            background: linear-gradient(135deg, var(--secondary-glow), var(--primary-glow));
            border: none;
            border-radius: 12px;
            color: #000;
            font-weight: 700;
            font-size: 1.1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px -5px rgba(0, 242, 254, 0.4);
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 25px -5px rgba(0, 242, 254, 0.6);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        .result-card {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(0, 242, 254, 0.05);
            border: 1px solid rgba(0, 242, 254, 0.2);
            border-radius: 16px;
            text-align: center;
            display: none;
            animation: pulseGlow 2s infinite alternate;
        }

        @keyframes pulseGlow {
            from { border-color: rgba(0, 242, 254, 0.2); }
            to { border-color: rgba(0, 242, 254, 0.8); }
        }

        .result-card h2 {
            font-size: 1.2rem;
            color: var(--text-sub);
        }

        .result-card .price {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-glow);
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Vehicle Price Predictor</h1>
        <p>Random Forest Machine Learning Engine</p>
    </div>

    <form id="predictForm" class="grid">
        <div class="input-group">
            <label>Year</label>
            <input type="number" name="Year" value="2018" required>
        </div>
        <div class="input-group">
            <label>Engine Size (L)</label>
            <input type="number" step="0.1" name="Engine_Size" value="2.0" required>
        </div>
        <div class="input-group">
            <label>Mileage (km)</label>
            <input type="number" name="Mileage" value="45000" required>
        </div>
        <div class="input-group">
            <label>Horsepower</label>
            <input type="number" name="Horsepower" value="180" required>
        </div>
        <div class="input-group">
            <label>Torque (Nm)</label>
            <input type="number" name="Torque" value="250" required>
        </div>
        <div class="input-group">
            <label>Owners</label>
            <input type="number" name="Owners" value="1" required>
        </div>
        <div class="input-group">
            <label>Fuel Efficiency (km/l)</label>
            <input type="number" step="0.1" name="Fuel_Efficiency" value="15.5" required>
        </div>

        <button type="submit" class="btn-submit">Estimate Price</button>
    </form>

    <div class="result-card" id="resultCard">
        <h2>Estimated Valuation</h2>
        <div class="price" id="predictedPrice">$0.00</div>
    </div>
</div>

<script>
    document.getElementById('predictForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();

            if (result.success) {
                const priceElement = document.getElementById('predictedPrice');
                const resultCard = document.getElementById('resultCard');
                
                resultCard.style.display = 'block';
                priceElement.innerText = '$' + Number(result.prediction).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            } else {
                alert('Error: ' + result.error);
            }
        } catch (err) {
            alert('Error connecting to backend model service.');
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'success': False, 'error': 'Model pickle file not loaded properly on server.'})

    try:
        data = request.get_json()
        
        # Expected model features based on your pickle file:
        # ['Make', 'Model', 'Year', 'Fuel_Type', 'Transmission', 'Engine_Size', 
        #  'Mileage', 'Horsepower', 'Torque', 'Owners', 'Accident_History', 
        #  'Service_History', 'Color', 'Body_Type', 'Drivetrain', 'Fuel_Efficiency', 'Location']
        
        # Mapping input numerical fields, defaulting categorical/missing attributes to dummy numeric index 0
        input_features = [
            0, # Make
            0, # Model
            float(data.get('Year', 2018)),
            0, # Fuel_Type
            0, # Transmission
            float(data.get('Engine_Size', 2.0)),
            float(data.get('Mileage', 50000)),
            float(data.get('Horsepower', 150)),
            float(data.get('Torque', 200)),
            float(data.get('Owners', 1)),
            0, # Accident_History
            0, # Service_History
            0, # Color
            0, # Body_Type
            0, # Drivetrain
            float(data.get('Fuel_Efficiency', 15.0)),
            0  # Location
        ]

        prediction = model.predict([input_features])[0]
        return jsonify({'success': True, 'prediction': float(prediction)})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
