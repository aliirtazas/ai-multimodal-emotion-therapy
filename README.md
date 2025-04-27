# EmotiAI: AI-Powered Multimodal Emotion Analysis System for Online Therapy Session

**🔍 An AI system for analyzing patient emotions in therapy using speech transcription, facial expression recognition, and speaker-based segmentation.**

## 📌 Overview

EmotIAI is an end-to-end AI support system that helps therapists understand their clients’ emotions more deeply during therapy sessions.
It combines Natural Language Processing and Computer Vision techniques to analyze Zoom therapy recordings, extracting emotional insights from both spoken words and facial expressions.

Therapists can simply upload a session video — whether in Gallery View or Speaker View — and EmotIAI automatically:

- Identifies patient segments

- Classifies emotions from speech and face

- Generates intuitive dashboards tracking emotional patterns throughout the session.

## 🚀 Features
✔️ Speech-to-Text Conversion
→ Converts audio into precise word-level transcripts using WhisperX.

✔️ Speaker Role Classification
→ Automatically separates therapist and client turns using a fine-tuned Roberta model.

✔️ Text-Based Emotion Analysis
→ Predicts patient emotions like Sadness, Anger, Happy, Fear, Confusion, Surprise from speech using a fine-tuned DeBERTa-v3 model.

✔️ Facial Emotion Recognition
→ Classifies subtle facial expressions using a fine-tuned ResNet-18 model trained on AffectNet.

✔️ Gallery and Speaker View Support
→ Works with different Zoom recording layouts, automatically identifying the client's face.

✔️ Multimodal Emotion Timeline
→ Merges both speech emotion and face emotion into a single dashboard view.

✔️ Professional Emotion Dashboard
→ Visualizes emotional trends, distributions, and transitions over time to assist therapists in analysis.

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

## 🔧 Installation Guide
```bash
git clone https://github.com/aliirtazas/ai-multimodal-emotion-therapy.git
cd ai-multimodal-emotion-therapy
pip install -r requirements.txt

# Run the Flask app
cd webapp
python app.py
```
The app will start locally at http://127.0.0.1:5000/.

## 📊 Datasets
We fine-tuned and trained our models using carefully selected open-source datasets:

Speech Emotion Datasets:

ISEAR — International Survey on Emotions

GoEmotions — Fine-grained Reddit emotion dataset

DailyDialog — Conversational dialogue emotions

EsConv — Emotional support conversations

Dair-ai Open Datasets

Facial Emotion Dataset:

AffectNet — Over 450,000 labeled images

(Compared with RAF-DB and FER++ for evaluation)

## 🏗️ How It Works
1. Upload a Zoom/Meet recording → Select View Type (Gallery/Speaker) and Face Option.

2. Audio Extraction → Convert audio to WAV format.

3. Transcription + Diarization → Extract speech and separate speakers (WhisperX).

4. Speaker Role Classification → Predict who is the Therapist vs Client (Roberta).

5. Face Extraction → Capture client’s face frames from video.

6. Emotion Predictions → Predict speech emotions (DeBERTa) and face emotions (ResNet18).

7. Dashboard Visualization → Generate emotion timelines, distribution charts, and transitions.

## Website

![1](https://github.com/user-attachments/assets/561bbab4-9b75-4e86-80e9-232974ee0b56)
![2](https://github.com/user-attachments/assets/56691d26-30d1-49e7-a03a-7711cf993cc4)
![3](https://github.com/user-attachments/assets/a9843eda-2b97-4500-847b-22e66b81ad58)
![4](https://github.com/user-attachments/assets/25e6dfef-1072-474e-864b-e940b7f2c83b)

## Example Dashboards

![1](https://github.com/user-attachments/assets/f70f35da-fe38-45c4-86c5-4cac23ec2410)
![2](https://github.com/user-attachments/assets/93d2abc8-7a2c-4623-a9f1-f49cfcdecc7a)
![3](https://github.com/user-attachments/assets/644a28d8-d7b4-47ae-86ef-880745611ac0)
![4](https://github.com/user-attachments/assets/d935050d-2dbf-42cc-a292-036e5603ab97)
![5](https://github.com/user-attachments/assets/c4de354d-ec0d-4d2a-be26-6d6272b8480c)
![6](https://github.com/user-attachments/assets/33be799a-1694-4b0d-b184-7a31086e2f72)
![7](https://github.com/user-attachments/assets/a186beab-17bd-4980-a8ca-6ee46a5d9abc)














