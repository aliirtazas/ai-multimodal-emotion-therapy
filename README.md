# AI-Powered Multimodal Emotion Analysis System for Therapy Session

**🔍 An AI-powered system for analyzing patient emotions in therapy using NLP, computer vision, and motion detection.**

## 📌 Overview
This project integrates natural language processing (NLP), computer vision (CV), and motion detection to analyze patient emotions in therapy sessions. It extracts speech, detects facial expressions, and segments video to provide therapists with a comprehensive emotional profile of each session.

## 🚀 Features
✔ **Speech-to-Text Conversion** – Extracts spoken responses using ASR (Automatic Speech Recognition).

✔ **Text-Based Emotion Analysis (NLP)** – Uses a fine-tuned BERT-based model to classify emotions from patient responses.

✔ **Facial Emotion Detection (CV)** – Detects emotions like joy, sadness, anger, and surprise using deep learning models.

✔ **Motion Detection** – Segments video dynamically to isolate patient responses.

✔ **Emotion Visualization** – Generates emotion graphs to track sentiment trends over time.

## Project Structure

```bash
📁 ai-multimodal-emotion-therapy
│── 📂 data/                # Dataset storage  
│── 📂 models/              # Pre-trained and fine-tuned models  
│── 📂 scripts/             # Core processing scripts  
│── 📂 notebooks/           # Experimentation & visualization notebooks  
│── main.py                 # End-to-end pipeline
|── requirement.txt         # Python dependencies
│── README.md               # Project documentation
```

## 🔧 Installation
```bash
git clone https://github.com/your-username/ai-multimodal-emotion-therapy.git
cd ai-multimodal-emotion-therapy
pip install -r requirements.txt
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
