from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

# Load model dan tools
model = joblib.load("model_hukuman_kekerasan_seksual.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")
data = pd.read_csv("kasus_seksual(4).csv", sep=';')



# Preprocessing
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return stemmer.stem(' '.join(words))

@app.route("/")
def home():
    return render_template(
        "index.html",
        current_year=datetime.now().year,
        current_user=session.get('username')
    )



@app.route("/pasal")
def pasal():
    # Prepare pasal data (you might want to load this from your CSV)
    pasal_data = ["Penjelasan_pasal"]
    
    return render_template(
        "index.html",
        pasal_data=pasal_data,
        current_year=datetime.now().year,
        current_user=session.get('username')
    )

@app.route("/konsultasi", methods=["GET", "POST"])
def konsultasi():
    teks = ""
    rekomendasi = penjelasan = penjara = denda = None

    if request.method == "POST":
        teks = request.form.get("teks_kasus", "")
        teks_clean = clean_text(teks)
        tfidf = vectorizer.transform([teks_clean])
        pred = model.predict(tfidf)[0]
        rekomendasi = le.inverse_transform([pred])[0]

        # Cari data yang sesuai
        filtered_data = data[data["Respon_Hukum"] == rekomendasi]

        if not filtered_data.empty:
            hasil = filtered_data.iloc[0]
            penjelasan = hasil.get("Penjelasan_pasal", "Tidak tersedia")
            penjara = hasil.get("Hukuman_Penjara", "Tidak tersedia")
            denda = hasil.get("Denda", "Tidak tersedia")
        else:
            penjelasan = "Data penjelasan tidak ditemukan."
            penjara = "Data hukuman penjara tidak ditemukan."
            denda = "Data denda tidak ditemukan."

    return render_template(
        "index.html",
        teks=teks,
        rekomendasi=rekomendasi,
        penjelasan=penjelasan,
        penjara=penjara,
        denda=denda,
        current_year=datetime.now().year,
        current_user=session.get('username')
    )

if __name__ == "__main__":
    app.run(debug=True)
    
    