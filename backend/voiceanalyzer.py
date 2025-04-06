import streamlit as st
import numpy as np
import pandas as pd
import parselmouth
import joblib
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import RFE
import os
import warnings

warnings.filterwarnings("ignore")

# ===============================
# ✅ ParkinsonVoiceAnalyzer class (kept as-is)
# ===============================

class ParkinsonVoiceAnalyzer:
    def __init__(self, model_path):
        self.default_feature_value = 0.0
        self.model_components = {}
        self.load_model(model_path)
        self.initialize_feature_extractors()

    def initialize_feature_extractors(self):
        self.feature_extractors = {
            'pitch': self.extract_pitch_features,
            'intensity': self.extract_intensity_features,
            'harmonicity': self.extract_harmonicity_features,
            'mfcc': self.extract_mfcc_features,
            'formants': self.extract_formant_features,
        }

    def load_model(self, model_path):
        try:
            self.model_components = joblib.load(model_path)
            self.model = self.model_components.get('model')
            self.power_transformer = self.model_components.get('power_transformer')
            self.scaler = self.model_components.get('scaler')
            self.rfe_selector = self.model_components.get('rfe_selector')
            self.selected_features = self.model_components.get('selected_features', [])

            if self.power_transformer and hasattr(self.power_transformer, 'get_feature_names_out'):
                self.all_features = self.power_transformer.get_feature_names_out()
            elif self.scaler and hasattr(self.scaler, 'get_feature_names_out'):
                self.all_features = self.scaler.get_feature_names_out()
            else:
                raise Exception("Could not infer initial features from transformers.")

            if not self.selected_features and self.rfe_selector and hasattr(self.rfe_selector, 'get_feature_names_out'):
                selected_indices = [int(col[1:]) for col in self.rfe_selector.get_feature_names_out() if col.startswith('x')]
                self.selected_features = [self.all_features[i] for i in selected_indices]
            elif not self.selected_features:
                self.selected_features = self.all_features[:20]
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def safe_divide(self, numerator, denominator):
        return numerator / (denominator + 1e-10) if denominator != 0 else 0

    def extract_pitch_features(self, sound):
        features = {}
        try:
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=600)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]
            if len(pitch_values) > 0:
                features['MDVP:Fo(Hz)'] = np.mean(pitch_values)
                features['MDVP:Fhi(Hz)'] = np.max(pitch_values)
                features['MDVP:Flo(Hz)'] = np.min(pitch_values)
                features['PPE'] = np.std(pitch_values) / (np.mean(pitch_values) + 1e-10)
                diffs = np.diff(pitch_values)
                if len(diffs) > 0:
                    features['MDVP:Jitter(%)'] = np.mean(np.abs(diffs)) / np.mean(pitch_values)
                    features['MDVP:Jitter(Abs)'] = np.mean(np.abs(diffs))
                    features['MDVP:PPQ'] = np.mean(np.abs(diffs[:5])) / np.mean(pitch_values[:4]) if len(pitch_values) >= 5 else 0
                    features['MDVP:RAP'] = np.mean(np.abs(diffs[:3])) / np.mean(pitch_values[:3]) if len(pitch_values) >= 3 else 0
                    features['Jitter:DDP'] = 3 * features.get('MDVP:RAP', 0)
        except Exception as e:
            print(f"Pitch extraction error: {e}")
        return features

    def extract_intensity_features(self, sound):
        features = {}
        try:
            intensity = sound.to_intensity(time_step=0.01)
            amp = intensity.values.flatten()
            if len(amp) > 1:
                amp_diffs = np.diff(amp)
                features['MDVP:Shimmer'] = np.mean(np.abs(amp_diffs)) / np.mean(amp)
                features['MDVP:Shimmer(dB)'] = 20 * np.log10(np.mean(np.abs(amp_diffs)) / (np.mean(amp) + 1e-10))
                features['Shimmer:APQ3'] = np.mean(np.abs(amp_diffs[:3])) / np.mean(amp[:3]) if len(amp) >= 3 else 0
                features['Shimmer:APQ5'] = np.mean(np.abs(amp_diffs[:5])) / np.mean(amp[:4]) if len(amp) >= 5 else 0
                features['MDVP:APQ'] = np.mean(np.abs(amp_diffs)) / np.mean(amp)
                features['sqrt_MDVP:Shimmer'] = np.sqrt(features.get('MDVP:Shimmer', 0))
        except Exception as e:
            print(f"Intensity extraction error: {e}")
        return features

    def extract_harmonicity_features(self, sound):
        features = {}
        try:
            harmonicity = sound.to_harmonicity_cc()
            values = harmonicity.values[harmonicity.values > 0]
            if len(values) > 0:
                features['HNR'] = np.mean(values)
                features['NHR'] = 1 / (features['HNR'] + 1e-10)
                features['sqrt_NHR'] = np.sqrt(features['NHR'])
        except Exception as e:
            print(f"Harmonicity extraction error: {e}")
        return features

    def extract_mfcc_features(self, sound):
        features = {}
        try:
            mfcc = sound.to_mfcc().to_array()
            features['RPDE'] = np.std(mfcc) / (np.mean(mfcc) + 1e-10)
            features['spread1'] = np.std(mfcc)
            features['spread2'] = np.var(mfcc)
            features['D2'] = np.max(mfcc) - np.min(mfcc) if mfcc.size > 0 else 0.0
        except Exception as e:
            print(f"MFCC extraction error: {e}")
        return features

    def extract_formant_features(self, sound):
        features = {}
        try:
            formants = sound.to_formant_burg()
            f1_values = []
            f2_values = []
            for i in range(formants.n_frames):
                time = formants.xs()[i]
                f1 = formants.get_value_at_time(1, time)
                f2 = formants.get_value_at_time(2, time)
                if f1 > 0: f1_values.append(f1)
                if f2 > 0: f2_values.append(f2)
            if f1_values and f2_values:
                features['DFA'] = np.mean(f1_values) / (np.mean(f2_values) + 1e-10)
        except Exception as e:
            print(f"Formant extraction error: {e}")
        return features

    def extract_voice_features(self, audio_path):
        try:
            sound = parselmouth.Sound(audio_path)
            features = {feat: self.default_feature_value for feat in self.all_features}
            for extractor_func in self.feature_extractors.values():
                features.update(extractor_func(sound))
            # Derived features
            if 'MDVP:Jitter(%)' in features and 'MDVP:Shimmer' in features:
                features['jitter_shimmer_ratio'] = self.safe_divide(features['MDVP:Jitter(%)'], features['MDVP:Shimmer'])
            if 'MDVP:Fhi(Hz)' in features and 'MDVP:Flo(Hz)' in features:
                features['pitch_variation'] = features['MDVP:Fhi(Hz)'] - features['MDVP:Flo(Hz)']
            if 'MDVP:Jitter(Abs)' in features and 'MDVP:PPQ' in features:
                features['jitter_ppq_apq'] = features['MDVP:Jitter(Abs)'] * features['MDVP:PPQ']
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def prepare_features(self, features_dict):
        features_df = pd.DataFrame([features_dict])
        for feat in self.all_features:
            if feat not in features_df.columns:
                features_df[feat] = self.default_feature_value
        features_df = features_df[self.all_features]
        if self.power_transformer:
            features_df = self.power_transformer.transform(features_df)
        if self.scaler:
            features_df = self.scaler.transform(features_df)
        if self.rfe_selector:
            features_df = self.rfe_selector.transform(features_df)
        return features_df

    def predict_parkinson(self, audio_path):
        if not os.path.exists(audio_path):
            return {'status': 'error', 'message': 'File not found', 'file_name': os.path.basename(audio_path)}
        features = self.extract_voice_features(audio_path)
        if not features:
            return {'status': 'error', 'message': 'Feature extraction failed', 'file_name': os.path.basename(audio_path)}
        features_final = self.prepare_features(features)
        prediction = self.model.predict(features_final)[0]
        proba = self.model.predict_proba(features_final)[0][1] if hasattr(self.model, 'predict_proba') else 0.5
        confidence = 'High' if proba > 0.8 or proba < 0.2 else 'Medium' if proba > 0.6 or proba < 0.4 else 'Low'
        return {
            'status': 'success',
            'prediction': "Parkinson's" if prediction == 1 else 'Healthy',
            'probability': float(proba),
            'confidence': confidence,
            'file_name': os.path.basename(audio_path),
            'features_used': features_final.shape[1],
            'extracted_features': len(features)
        }

# ===============================
# ✅ Streamlit Web Interface
# ===============================

def main():
    st.set_page_config(page_title="Parkinson's Voice Analyzer", layout="centered")
    st.title("🧠 Parkinson's Detection from Voice")
    st.markdown("Upload a `.wav` audio file to analyze and detect Parkinson’s disease.")

    model_path = "models/parkinsons_ensemble_model.pkl"
    try:
        analyzer = ParkinsonVoiceAnalyzer(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return

    uploaded_file = st.file_uploader("🎤 Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())
        st.audio("temp_audio.wav", format="audio/wav")
        result = analyzer.predict_parkinson("temp_audio.wav")

        if result['status'] == 'success':
            st.success(f"✅ Prediction: {result['prediction']}")
            st.metric("🎯 Probability", f"{result['probability'] * 100:.2f}%")
            st.metric("🔒 Confidence", result['confidence'])
            st.markdown(f"🔍 Features Extracted: `{result['extracted_features']}`")
            st.markdown(f"⚙️ Model Input Features: `{result['features_used']}`")
        else:
            st.error(f"❌ {result['message']}")

if __name__ == "__main__":
    main()
