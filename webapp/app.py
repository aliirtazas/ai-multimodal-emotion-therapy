from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from pipeline import process_video_pipeline
from wordcloud import WordCloud
from collections import Counter
import shutil
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

@app.context_processor
def inject_auth():
    return dict(logged_in=('user_id' in session))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Account created. Please log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('upload'))
        else:
            flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        flash('Please log in to upload videos.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        video = request.files['video']
        view_type = request.form.get('view_type')
        face_option = request.form.get('face_option')

        if video and view_type and face_option:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(filepath)

            hf_token = "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"

            try:
                final_df, client_df = process_video_pipeline(
                    video_path=filepath,
                    hf_token=hf_token,
                    view_type=view_type,
                    face_option=face_option
                )

                session_dir = os.path.join("static", "sessions")
                os.makedirs(session_dir, exist_ok=True)
                client_df_path = os.path.join(session_dir, f"{os.path.splitext(video.filename)[0]}_client.json")
                final_df_path = os.path.join(session_dir, f"{os.path.splitext(video.filename)[0]}_final.json")

                client_df.to_json(client_df_path)
                final_df.to_json(final_df_path)

                session['client_df_path'] = client_df_path
                session['final_df_path'] = final_df_path

                return jsonify({'redirect': url_for('dashboard')})

            except Exception as e:
                print(f"Error during processing: {e}")
                flash("Processing failed. Check logs.")
                return redirect(url_for('upload'))

        flash("Please provide all required inputs.")
        return redirect(url_for('upload'))

    return render_template('upload.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to view the dashboard.')
        return redirect(url_for('login'))

    if 'client_df_path' not in session or 'final_df_path' not in session:
        flash('No data available. Please upload a video first.')
        return redirect(url_for('upload'))

    client_df = pd.read_json(session['client_df_path'])
    final_df = pd.read_json(session['final_df_path'])

    # Emotion mappings
    emotion_to_id = {
        "Anger": 0, "Fear": 1, "Happy": 2, "Sadness": 3,
        "Neutral": 4, "Surprise": 5, "Confusion": 6, "Disgust": 7
    }

    def time_to_sec(t):
        try:
            m, s = map(int, t.split(":"))
            return m * 60 + s
        except:
            return None

    client_df["start_sec"] = client_df["Start"].apply(time_to_sec)
    client_df["end_sec"] = client_df["End"].apply(time_to_sec)  

    client_df["duration"] = client_df["end_sec"] - client_df["start_sec"]
    client_df["speech_emotion_id"] = client_df["speech_predicted_emotion"].map(emotion_to_id)
    client_df["face_emotion_id"] = client_df["face_emotion_prediction"].map(emotion_to_id)

    custom_order = ["Happy","Surprise","Neutral","Confusion",
                    "Sadness","Fear","Disgust","Anger"]

    # --- Emotion Timeline ---------------------------------------------------------------------
    custom_order = ["Happy","Surprise","Neutral","Confusion",
                    "Sadness","Fear","Disgust","Anger"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=client_df["start_sec"],
        y=client_df["speech_predicted_emotion"],     # use the string labels
        mode='lines+markers',
        name='Speech Emotion',
        marker=dict(size=10, color='#1f77b4'),
        line=dict(width=2),
        hovertemplate='Time: %{x}s<br>Speech: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=client_df["start_sec"],
        y=client_df["face_emotion_prediction"],      # use the string labels
        mode='lines+markers',
        name='Face Emotion',
        marker=dict(size=10, color='#ff7f0e'),
        line=dict(width=2),
        hovertemplate='Time: %{x}s<br>Face: %{y}<extra></extra>'
    ))

    # Categorical y-axis ordering
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis=dict(
            title="Emotion",
            type="category",
            categoryorder="array",
            categoryarray=custom_order
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        height=600
    )

    graph_html = fig.to_html(full_html=False)

    # --- Emotion Match Pie Chart ----------------------------------------------------------------------------
    client_df["emotion_match"] = client_df["speech_predicted_emotion"] == client_df["face_emotion_prediction"]
    match_counts = client_df["emotion_match"].value_counts()
    labels = ["Match" if x else "Mismatch" for x in match_counts.index]
    colors_pie = ['#1f77b4', '#ff7f0e']  # Green for match, Orange for mismatch
    plt.figure(figsize=(6, 7))
    plt.pie(match_counts, labels=labels, autopct="%1.1f%%", colors=colors_pie)
    plt.title("Emotion Agreement Between Speech & Face")
    plt.tight_layout()
    plt.savefig("static/emotion_match_pie.png")
    plt.close()

    # --- Emotion Distribution Bar Chart -----------------------------------------------------------------------
    speech_counts = client_df["speech_predicted_emotion"].value_counts()
    face_counts = client_df["face_emotion_prediction"].value_counts()
    all_emotions = sorted(set(speech_counts.index) | set(face_counts.index))
    speech_counts = speech_counts.reindex(all_emotions, fill_value=0)
    face_counts = face_counts.reindex(all_emotions, fill_value=0)
    x = np.arange(len(all_emotions))
    width = 0.35
    plt.figure(figsize=(10, 7))
    plt.bar(x - width/2, speech_counts, width, label="Speech", color='#1f77b4')
    plt.bar(x + width/2, face_counts, width, label="Face", color='#ff7f0e')
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.title("Emotion Distribution (Speech vs Face)")
    plt.xticks(x, all_emotions, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("static/emotion_distribution_bar.png")
    plt.close()

    # --- Speaker Timeline --------------------------------------------------------------------------------------
    speaker_df = final_df.copy()
    speaker_df["start_sec"] = speaker_df["Start"].apply(time_to_sec)
    speaker_df["end_sec"] = speaker_df["End"].apply(time_to_sec)
    speaker_df["duration"]  = speaker_df["end_sec"] - speaker_df["start_sec"]
    speaker_df["Speaker_Title"] = speaker_df["Speaker"].str.title()

    totals = (
        speaker_df
        .groupby("Speaker_Title", as_index=False)["duration"]
        .sum()
    )

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.75, 0.25],
        specs=[[{"type":"xy"}, {"type":"domain"}]],
        subplot_titles=("Speaking Segments Over Time", "Time Share"),
        horizontal_spacing=0.1
    )

    colors = {"Client":"#1f77b4", "Therapist":"#ff7f0e"}

    # Left: Gantt-style bars
    for speaker in speaker_df["Speaker_Title"].unique():
        df_s = speaker_df[speaker_df["Speaker_Title"] == speaker]
        fig.add_trace(
            go.Bar(
                x=df_s["duration"],
                y=df_s["Speaker_Title"],
                base=df_s["start_sec"],
                orientation="h",
                name=speaker,
                marker=dict(
                    color=colors[speaker],
                    line=dict(color="white", width=1)
                ),
                hovertemplate=(
                    "Speaker: %{y}<br>"
                    "Start: %{base}s<br>"
                    "Duration: %{x}s<br>"
                    "<extra></extra>"
                )
            ),
            row=1, col=1
        )

    fig.update_xaxes(
        title_text="Time (seconds)",
        row=1, col=1,
        tickformat=".0f",
        showgrid=True,
        gridcolor="lightgrey"
    )
    fig.update_yaxes(
        autorange="reversed",
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=totals["Speaker_Title"],
            values=totals["duration"],
            hole=0.5,
            marker=dict(colors=[colors[s] for s in totals["Speaker_Title"]]),
            textinfo="label+percent",
            sort=False,
            domain=dict(x=[0.78, 0.98], y=[0.1, 0.9])
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Speaking Segments and Time Share",
        width=1200,
        height=500,
        margin=dict(l=80, r=40, t=80, b=40),
        template="plotly_white",
        showlegend=False,
        bargap=0.05,
        font=dict(size=12)
    )

    speaker_html = fig.to_html(full_html=False)

    # --- Emotion Transition Sankey -------------------------------------------------------------------------------
    transitions = list(zip(client_df["speech_predicted_emotion"], client_df["speech_predicted_emotion"].shift(-1)))
    transitions = [t for t in transitions if t[0] != t[1] and pd.notna(t[1])]
    transition_counts = Counter(transitions)
    labels = list(set([src for src, _ in transitions] + [tgt for _, tgt in transitions]))
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    node_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#17becf', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    node_color_map = [node_colors[i % len(node_colors)] for i in range(len(labels))]

    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_color_map
        ),
        link=dict(
            source=[label_to_index[src] for src, _ in transition_counts],
            target=[label_to_index[tgt] for _, tgt in transition_counts],
            value=list(transition_counts.values())
        )
    )])
    sankey_fig.update_layout(height=500, template="plotly_white")
    sankey_html = sankey_fig.to_html(full_html=False)

    # --- Word Clouds ---------------------------------------------------------------------------------------------
    wordcloud_dir = "static/wordclouds"
    shutil.rmtree(wordcloud_dir, ignore_errors=True)
    os.makedirs(wordcloud_dir, exist_ok=True)
    emotion_texts = client_df.groupby("speech_predicted_emotion")["Text"].apply(lambda x: " ".join(x)).to_dict()
    wordcloud_paths = []
    for emotion, text in emotion_texts.items():
        wc = WordCloud(width=500, height=400, background_color="white").generate(text)
        filepath = os.path.join(wordcloud_dir, f"{emotion}.png")
        wc.to_file(filepath)
        wordcloud_paths.append((emotion, f"wordclouds/{emotion}.png"))

    #----Speech Emotion heatmap over time---------------------------------------------------------------------------
    heatmap_df = client_df.copy()
    heatmap_df['time_bin'] = pd.cut(heatmap_df['start_sec'], bins=10, labels=False)

    heatmap_data = pd.crosstab(heatmap_df['speech_predicted_emotion'], heatmap_df['time_bin'])

    plt.figure(figsize=(12, 7))
    sns.heatmap(heatmap_data, cmap='Blues', annot=True, fmt='d', linewidths=.5)
    plt.xlabel("Time Segments")
    plt.ylabel("Emotion")
    plt.title("Emotion Frequency Heatmap Over Time (Speech)")
    plt.tight_layout()
    heatmap_path = "static/emotion_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()

    #- Face Emotion Heatmap over Time-----------------------------------------------------------------------------------
    heatmap_face_df = client_df.copy()
    heatmap_face_df['time_bin'] = pd.cut(heatmap_face_df['start_sec'], bins=10, labels=False)

    heatmap_face_data = pd.crosstab(heatmap_face_df['face_emotion_prediction'], heatmap_face_df['time_bin'])

    plt.figure(figsize=(12, 7))
    sns.heatmap(heatmap_face_data, cmap='Oranges', annot=True, fmt='d', linewidths=.5)
    plt.xlabel("Time Segments")
    plt.ylabel("Emotion")
    plt.title("Emotion Frequency Heatmap Over Time (Face)")
    plt.tight_layout()
    heatmap_face_path = "static/emotion_heatmap_face.png"
    plt.savefig(heatmap_face_path)
    plt.close()

    #--- Render the dashboard template ---------------------------------------------------------------------------------

    return render_template(
        'dashboard.html',
        graph_html=graph_html,
        speaker_html=speaker_html,
        sankey_html=sankey_html,
        wordcloud_paths=wordcloud_paths,
        speech_emotions=client_df['speech_predicted_emotion'].unique(),
        face_emotions=client_df['face_emotion_prediction'].unique()
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
