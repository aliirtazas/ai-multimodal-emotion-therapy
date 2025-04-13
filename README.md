# AI-Powered Multimodal Emotion Analysis System for Therapy Session

**🔍 An AI system for analyzing patient emotions in therapy using speech transcription, facial expression recognition, and speaker-based segmentation.**

## 📌 Overview
This system integrates **Natural Language Processing (NLP)** and **Computer Vision (CV)** to analyze client emotions in recorded therapy sessions (e.g., Zoom, Google Meet). It performs speaker-based segmentation, transcribes patient responses, and classifies both spoken and facial emotions using deep learning. Therapists can upload session recordings (Gallery View or Speaker View), and the system automatically extracts the patient's segments to generate detailed emotional timelines and visualizations.

## 🚀 Features
✔ **Speech-to-Text Conversion** – Extracts spoken responses using ASR (Automatic Speech Recognition).

✔ **Text-Based Emotion Analysis (NLP)** – Uses a fine-tuned BERT-based model to classify emotions from patient responses.

✔ **Facial Emotion Detection (CV)** – Detects emotions like joy, sadness, anger, and surprise using deep learning models.

✔ **Speaker-Based Segmentation** – Automatically separates therapist and client turns using diarization and speaker classification.

✔ **Video View Support** – Handles both Gallery View and Speaker View recordings for accurate face extraction.

✔ **Emotion Visualization** – Generates emotion graphs to track sentiment trends over time.

## Project Structure

```bash
📁 ai-multimodal-emotion-therapy
│── 📂 data/                # Dataset storage  
│── 📂 models/              # Pre-trained and fine-tuned models  
│── 📂 notebooks_code/      # Experimentation & visualization notebooks  
│── 📂 results/
│── 📂 uploads/
│── 📂 webapp/              # webapp created using flask
|── requirement.txt         # Python dependencies
│── README.md               # Project documentation
```

## 🔧 Installation
```bash
git clone https://github.com/your-username/ai-multimodal-emotion-therapy.git
cd ai-multimodal-emotion-therapy
pip install -r requirements.txt

# Run the Flask app
cd webapp
python app.py
```

## 📊 Datasets
We use publicly available emotion datasets for fine-tuning and training the models, such as:

- ISEAR (Emotion Classification)
- GoEmotions (Fine-grained Emotion Labels)
- DailyDialog (Conversational Emotion Data)
- MELD
- Google Affective Datasets
- AffectNet

## 🏗️ How It Works
1️⃣ Extract speech → Convert audio to text using ASR.

2️⃣ Analyze text emotion → Classify emotions using a BERT-based model.

3️⃣ Detect facial emotions → Process video frames to classify facial expressions.

4️⃣ Segment responses → Use motion detection to isolate patient responses.

5️⃣ Visualize insights → Generate emotion trajectories and reports.
