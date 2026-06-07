import os
import pickle
import re
from flask import Flask, request, jsonify, render_template

# Initialize Flask App
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))

# Load Model with absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(base_dir, "spam_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"FATAL: Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None


def clean_text(text):
    """Clean and normalize the input text for prediction."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON payload'}), 400

        message = data.get('message', '')

        if not message.strip():
            return jsonify({'error': 'Message cannot be empty'}), 400

        cleaned_message = clean_text(message)
        msg_vector = vectorizer.transform([cleaned_message])

        prediction = model.predict(msg_vector)
        probability = model.predict_proba(msg_vector)

        spam_prob = round(float(probability[0][1] * 100), 2)
        ham_prob = round(float(probability[0][0] * 100), 2)

        result = {
            'is_spam': bool(prediction[0] == 1),
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'cleaned_message': cleaned_message
        }

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
