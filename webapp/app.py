from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import plotly.express as px
import time

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
    flash('You have been logged out.')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        flash('Please log in to upload videos.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        video = request.files['video']
        if video:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
            video.save(filepath)

            # Simulate processing time
            time.sleep(2)

            # TODO: Integrate your backend processing here

            return jsonify({'redirect': url_for('dashboard')})
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to view the dashboard.')
        return redirect(url_for('login'))

    df = pd.DataFrame({
        'Time': ['00:01', '00:02', '00:03', '00:04'],
        'Speech_Emotion': ['Happy', 'Sad', 'Angry', 'Neutral'],
        'Face_Emotion': ['Neutral', 'Sad', 'Angry', 'Happy']
    })

    # Prepare combined line plot for comparison
    df_long = pd.melt(df, id_vars='Time', value_vars=['Speech_Emotion', 'Face_Emotion'], 
                      var_name='Source', value_name='Emotion')

    fig = px.line(df_long, x='Time', y='Emotion', color='Source', title='Speech vs Face Emotion Over Time')
    graph_html = fig.to_html(full_html=False)

    speech_emotions = df['Speech_Emotion'].unique()
    face_emotions = df['Face_Emotion'].unique()

    return render_template('dashboard.html', 
                           graph_html=graph_html,
                           speech_emotions=speech_emotions,
                           face_emotions=face_emotions)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)