
import os
import torch
import whisperx
import pandas as pd
import cv2
import shutil
import joblib
import numpy as np
from PIL import Image
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DebertaV2TokenizerFast,
    DebertaV2ForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
from moviepy import *

# ----- Global Variables -----------------------------------------------------
emotion_labels = {
    0: "Anger", 1: "Fear", 2: "Happy", 3: "Sadness",
    4: "Neutral", 5: "Surprise", 6: "Confusion", 7: "Disgust"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- Audio & Transcription ---------------------------

def extract_audio(video_path):
    base = os.path.splitext(video_path)[0]
    audio_output_path = base + ".wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)
    return audio_output_path

def transcribe_and_diarize(video_path, hf_token, whisper_model="large-v2", device="cuda"):
    audio_path = extract_audio(video_path)
    model = whisperx.load_model(whisper_model, device=device, compute_type="float16")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    result = model.transcribe(audio_path, chunk_size=15)
    diarization_result = diarize_model(audio_path)
    result = whisperx.assign_word_speakers(diarization_result, result)

    data = []
    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"]
        start_time_sec = segment["start"]
        end_time_sec = segment["end"]
        data.append([start_time_sec, end_time_sec, speaker, text])

    df = pd.DataFrame(data, columns=["Start (sec)", "End (sec)", "Speaker", "Text"])
    df["total_duration"] = df["End (sec)"] - df["Start (sec)"]
    return df


# --------------------------- Speaker Prediction ---------------------------
# Load model components
def load_speaker_model(model_dir, device):
    model = RobertaForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    model.eval()
    return model, tokenizer, label_encoder

# Predict speaker label
def predict_speaker_class(df_subset, model, tokenizer, label_encoder, device):
    dataset = Dataset.from_pandas(df_subset[["Text"]].rename(columns={"Text": "utterance"}))
    dataset = dataset.map(lambda x: tokenizer(x["utterance"], padding="max_length", truncation=True, max_length=512), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    loader = DataLoader(dataset, batch_size=16)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, axis=-1).cpu().numpy()
            all_preds.extend(preds)

    return label_encoder.inverse_transform(all_preds)

def classify_and_map_speakers(df, model, tokenizer, label_encoder, device):
    # Normalize column names
    df = df.rename(columns={"Start (sec)": "start_sec", "End (sec)": "end_sec"})

    # SPEAKER_00
    s0 = df[df["Speaker"] == "SPEAKER_00"].copy()
    p0 = predict_speaker_class(s0, model, tokenizer, label_encoder, device)
    s0["Pred"] = p0
    vc0 = s0["Pred"].value_counts()
    maj0, pct0 = vc0.idxmax().lower(), vc0.max() / len(s0)

    # SPEAKER_01
    s1 = df[df["Speaker"] == "SPEAKER_01"].copy()
    p1 = predict_speaker_class(s1, model, tokenizer, label_encoder, device)
    s1["Pred"] = p1
    vc1 = s1["Pred"].value_counts()
    maj1, pct1 = vc1.idxmax().lower(), vc1.max() / len(s1)

    # Choose winner by confidence
    if pct0 >= pct1:
        win_key, win_lbl = "SPEAKER_00", maj0
        lose_key = "SPEAKER_01"
    else:
        win_key, win_lbl = "SPEAKER_01", maj1
        lose_key = "SPEAKER_00"

    opp_lbl = "therapist" if win_lbl == "client" else "client"
    speaker_map = {win_key: win_lbl.title(), lose_key: opp_lbl.title()}
    return df.replace({"Speaker": speaker_map})

# Convert seconds to MM:SS
def sec_to_min_sec(seconds):
    minutes = int(seconds) // 60
    seconds = int(seconds) % 60
    return f"{minutes:02d}:{seconds:02d}"

# Merge therapist segments
def merge_conversation_segments(df):
    df = df.sort_values("start_sec").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["Start","End","Speaker","Text"])

    merged = []
    curr_spk   = df.loc[0, "Speaker"]
    curr_start = df.loc[0, "start_sec"]
    curr_end   = df.loc[0, "end_sec"]
    curr_txt   = df.loc[0, "Text"]

    for i in range(1, len(df)):
        r = df.loc[i]
        if r["Speaker"] == curr_spk:
            curr_end  = r["end_sec"]
            curr_txt += " " + r["Text"]
        else:
            merged.append({
                "Start":   sec_to_min_sec(curr_start),
                "End":     sec_to_min_sec(curr_end),
                "Speaker": curr_spk,
                "Text":    curr_txt
            })
            curr_spk   = r["Speaker"]
            curr_start = r["start_sec"]
            curr_end   = r["end_sec"]
            curr_txt   = r["Text"]

    merged.append({
        "Start":   sec_to_min_sec(curr_start),
        "End":     sec_to_min_sec(curr_end),
        "Speaker": curr_spk,
        "Text":    curr_txt
    })

    return pd.DataFrame(merged)

# --------------------------- Face Extraction ---------------------------

def time_to_seconds(t):
    minutes, seconds = map(int, t.split(':'))
    return minutes * 60 + seconds

def extract_faces_for_client_segments(client_df, video_path, output_dir, view_type, option):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    client_df['Start_sec'] = client_df['Start'].apply(time_to_seconds)
    client_df['End_sec'] = client_df['End'].apply(time_to_seconds)
    client_df['Mid_sec'] = (client_df['Start_sec'] + client_df['End_sec']) / 2
    client_df['Mid_sec'] = client_df['Mid_sec'].round().astype(int)
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    def save_resized_face(face, prefix, idx):
        resized = cv2.resize(face, (256, 256), interpolation=cv2.INTER_AREA)
        path = os.path.join(output_dir, f"{prefix}_{idx}.jpg")
        cv2.imwrite(path, resized)
        return path

    cap = cv2.VideoCapture(video_path)
    for idx, row in client_df.iterrows():
        cap.set(cv2.CAP_PROP_POS_MSEC, round(row['Mid_sec']) * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        face = None
        if view_type == "Gallery View":
            half = frame[:, :w//2] if option == "Left" else frame[:, w//2:]
            faces = face_cascade.detectMultiScale(half, 1.3, 5)
            if len(faces) == 1:
                x, y, fw, fh = faces[0]
                face = half[y:y+fh, x:x+fw]
                client_df.at[idx, 'Image_Path'] = save_resized_face(face, option.lower(), idx)
        elif view_type == "Speaker View":
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            if len(faces) == 2:
                areas = [(fw * fh, (x, y, fw, fh)) for (x, y, fw, fh) in faces]
                areas.sort(reverse=True)
                chosen = areas[0][1] if option == "Large" else areas[1][1]
                x, y, fw, fh = chosen
                face = frame[y:y+fh, x:x+fw]
                client_df.at[idx, 'Image_Path'] = save_resized_face(face, option.lower(), idx)

    cap.release()
    return client_df


# --------------------------- Text Emotion Prediction ---------------------------

class WeightedDeBERTa(nn.Module):
    def __init__(self, model_name, num_labels, class_weights):
        super(WeightedDeBERTa, self).__init__()
        self.num_labels = num_labels
        self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(p=0.3)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.dropout(outputs.logits)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return SequenceClassifierOutput(loss=loss, logits=logits)
        return SequenceClassifierOutput(logits=logits)

def add_speech_emotions_to_client_df(client_df, model_path):
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_path, local_files_only=True)
    checkpoint = torch.load(os.path.join(model_path, 'custom_model.pth'))
    class_weights = checkpoint['class_weights']

    model = WeightedDeBERTa("microsoft/deberta-v3-small", 8, class_weights).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.model.config.id2label = emotion_labels
    model.model.config.label2id = {v: k for k, v in emotion_labels.items()}

    def predict_emotion(text):
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        with torch.no_grad():
            logits = model(**inputs).logits
            return emotion_labels[torch.argmax(logits, dim=-1).item()]

    client_df["speech_predicted_emotion"] = client_df["Text"].apply(predict_emotion)
    return client_df

# --------------------------- Face Emotion Prediction ---------------------------

def load_cv_emotion_model(weights_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.BatchNorm1d(num_ftrs), nn.Dropout(0.5), nn.Linear(num_ftrs, 8))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def add_face_emotions_to_client_df(client_df, model_path):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    model = load_cv_emotion_model(model_path)

    def predict_emotion(path):
        try:
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                pred = model(tensor)
                return emotion_labels[torch.argmax(pred, dim=1).item()]
        except Exception as e:
            return "Error"

    client_df["face_emotion_prediction"] = client_df["Image_Path"].apply(predict_emotion)
    return client_df

# --------------------------- Main Processing Pipeline ---------------------------

def process_video_pipeline(video_path, hf_token, view_type, face_option):
    # Paths
    speaker_model_path = r"D:\Data Science Projects Github\ai-multimodal-emotion-therapy\models\speaker_prediction_roberta_model"
    speech_model_path = r"D:\Data Science Projects Github\ai-multimodal-emotion-therapy\models\deberta_model"
    face_model_path = r"D:\Data Science Projects Github\ai-multimodal-emotion-therapy\models\cv model\best_model (1).pth"
    output_dir = r"D:\Data Science Projects Github\ai-multimodal-emotion-therapy\webapp\Extracted_Images"

    df = transcribe_and_diarize(video_path, hf_token=hf_token)

    #Speaker classification
    spk_model, spk_tokenizer, spk_encoder = load_speaker_model(speaker_model_path, device)
    df = classify_and_map_speakers(df, spk_model, spk_tokenizer, spk_encoder, device)

    #Merge segments
    final_df = merge_conversation_segments(df)

    #Prepare client segments
    client_df = final_df[final_df["Speaker"].str.lower() == "client"].copy()
    client_df["Start_sec"] = client_df["Start"].apply(time_to_seconds)
    client_df["End_sec"] = client_df["End"].apply(time_to_seconds)
    client_df["Mid_sec"] = (client_df["Start_sec"] + client_df["End_sec"]) / 2
    client_df["Mid_sec"] = client_df["Mid_sec"].round().astype(int)
    client_df = client_df.reset_index(drop=True)

    #Face extraction
    client_df = extract_faces_for_client_segments(client_df, video_path, output_dir, view_type, face_option)
    client_df = client_df.dropna()
    client_df = client_df.reset_index(drop=True)

    #Text emotion prediction
    client_df = add_speech_emotions_to_client_df(client_df, speech_model_path)

    #Face emotion prediction
    client_df = add_face_emotions_to_client_df(client_df, face_model_path)
    client_df = client_df.dropna()
    client_df = client_df.reset_index(drop=True)

    #Merge back into final_df
    for idx in client_df.index:
        final_df.loc[client_df.index[idx], 'Image_Path'] = client_df.at[idx, 'Image_Path']
        final_df.loc[client_df.index[idx], 'speech_predicted_emotion'] = client_df.at[idx, 'speech_predicted_emotion']
        final_df.loc[client_df.index[idx], 'face_emotion_prediction'] = client_df.at[idx, 'face_emotion_prediction']

    return final_df, client_df

