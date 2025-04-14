from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from pipeline import process_video_pipeline

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
    client_df["speech_emotion_id"] = client_df["speech_predicted_emotion"].map(emotion_to_id)
    client_df["face_emotion_id"] = client_df["face_emotion_prediction"].map(emotion_to_id)

    # --- Emotion Timeline ---
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from collections import Counter
    import shutil
    import os
    import numpy as np

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=client_df["start_sec"],
        y=client_df["speech_emotion_id"],
        mode='lines+markers',
        name='Speech Emotion',
        marker=dict(size=10, color='royalblue'),
        line=dict(width=2),
        text=client_df["speech_predicted_emotion"],
        hovertemplate='Time: %{x}s<br>Speech: %{text}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=client_df["start_sec"],
        y=client_df["face_emotion_id"],
        mode='lines+markers',
        name='Face Emotion',
        marker=dict(size=10, color='orange'),
        line=dict(width=2),
        text=client_df["face_emotion_prediction"],
        hovertemplate='Time: %{x}s<br>Face: %{text}<extra></extra>'
    ))
    fig.update_layout(
        title="🎭 Dynamic Emotion Timeline (Speech vs Face)",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            title="Emotion",
            tickmode='array',
            tickvals=list(emotion_to_id.values()),
            ticktext=list(emotion_to_id.keys())
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        height=500
    )
    graph_html = fig.to_html(full_html=False)

    # --- Emotion Match Pie Chart ---
    client_df["emotion_match"] = client_df["speech_predicted_emotion"] == client_df["face_emotion_prediction"]
    match_counts = client_df["emotion_match"].value_counts()
    labels = ["Match" if x else "Mismatch" for x in match_counts.index]
    plt.figure(figsize=(6, 6))
    plt.pie(match_counts, labels=labels, autopct="%1.1f%%", colors=["lightgreen", "tomato"])
    plt.title("Emotion Agreement Between Speech & Face")
    plt.tight_layout()
    plt.savefig("static/emotion_match_pie.png")
    plt.close()

    # --- Emotion Distribution Bar Chart ---
    speech_counts = client_df["speech_predicted_emotion"].value_counts()
    face_counts = client_df["face_emotion_prediction"].value_counts()
    all_emotions = sorted(set(speech_counts.index) | set(face_counts.index))
    speech_counts = speech_counts.reindex(all_emotions, fill_value=0)
    face_counts = face_counts.reindex(all_emotions, fill_value=0)
    x = np.arange(len(all_emotions))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, speech_counts, width, label="Speech", color='skyblue')
    plt.bar(x + width/2, face_counts, width, label="Face", color='salmon')
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.title("Emotion Distribution (Speech vs Face)")
    plt.xticks(x, all_emotions, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("static/emotion_distribution_bar.png")
    plt.close()

    # --- Speaker Timeline ---
    speaker_df = final_df.copy()
    speaker_df["start_sec"] = speaker_df["Start"].apply(time_to_sec)
    speaker_df["end_sec"] = speaker_df["End"].apply(time_to_sec)
    speaker_plot = go.Figure()
    colors = {"Client": "#00BFFF", "therapist": "#FF7F50"}
    for speaker in speaker_df["Speaker"].unique():
        df_s = speaker_df[speaker_df["Speaker"] == speaker]
        speaker_plot.add_trace(go.Bar(
            x=df_s["end_sec"] - df_s["start_sec"],
            y=df_s["Speaker"],
            base=df_s["start_sec"],
            orientation='h',
            name=speaker,
            marker=dict(color=colors.get(speaker, "#999")),
            hovertext=df_s["Text"]
        ))
    speaker_plot.update_layout(
        title="👥 Speaker Timeline",
        xaxis_title="Time (seconds)",
        barmode='stack',
        height=300,
        template="plotly_white"
    )
    speaker_html = speaker_plot.to_html(full_html=False)

    # --- Emotion Transition Sankey ---
    transitions = list(zip(client_df["speech_predicted_emotion"], client_df["speech_predicted_emotion"].shift(-1)))
    transitions = [t for t in transitions if t[0] != t[1] and pd.notna(t[1])]
    transition_counts = Counter(transitions)
    labels = list(set([src for src, _ in transitions] + [tgt for _, tgt in transitions]))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    sankey_fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color="lightgray"
        ),
        link=dict(
            source=[label_to_index[src] for src, _ in transition_counts],
            target=[label_to_index[tgt] for _, tgt in transition_counts],
            value=list(transition_counts.values())
        )
    )])
    sankey_fig.update_layout(title_text="🔁 Emotion Transition Flow", height=400, template="plotly_white")
    sankey_html = sankey_fig.to_html(full_html=False)

    # --- Word Clouds ---
    wordcloud_dir = "static/wordclouds"
    shutil.rmtree(wordcloud_dir, ignore_errors=True)
    os.makedirs(wordcloud_dir, exist_ok=True)
    emotion_texts = client_df.groupby("speech_predicted_emotion")["Text"].apply(lambda x: " ".join(x)).to_dict()
    wordcloud_paths = []
    for emotion, text in emotion_texts.items():
        wc = WordCloud(width=500, height=300, background_color="white").generate(text)
        filepath = os.path.join(wordcloud_dir, f"{emotion}.png")
        wc.to_file(filepath)
        wordcloud_paths.append((emotion, f"wordclouds/{emotion}.png"))

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
