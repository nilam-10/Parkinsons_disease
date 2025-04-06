from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# ✅ Load regression model
model_bundle = joblib.load("models/regression.pkl")
model = model_bundle['model']
feature_cols = model_bundle['features']

# ✅ Load Q&A AI Model & Data
qa_model = joblib.load("models/qa_model.joblib")     # SentenceTransformer
qa_df = pd.read_pickle("models/qa_data.pkl")         # Pre-embedded Q&A data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_features = [data.get(col, 0) for col in feature_cols]
        input_array = np.array(input_features).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        return jsonify({
            'motor_UPDRS': round(prediction[0], 2),
            'total_UPDRS': round(prediction[1], 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask-ai', methods=['POST'])
def ask_ai():
    try:
        data = request.get_json()
        question = data.get("question", "")
        user_embedding = qa_model.encode([question])

        # Find best match using cosine similarity
        qa_df['similarity'] = qa_df['embedding'].apply(
            lambda emb: cosine_similarity([emb], user_embedding)[0][0]
        )

        best_match = qa_df.loc[qa_df['similarity'].idxmax()]
        if best_match['similarity'] > 0.4:
            return jsonify({"answer": best_match['answer']})
        else:
            return jsonify({"answer": "I couldn't find a precise answer. Try rephrasing?"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
