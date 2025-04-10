import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
from io import BytesIO
from urllib.parse import quote
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import unicodedata
import requests
import qrcode
from streamlit_extras.metric_cards import style_metric_cards
import streamlit.components.v1 as components
import plotly.graph_objs as go
import io

# Page setup
st.set_page_config(page_title="Fake News Detector", layout="wide", initial_sidebar_state="expanded")
st.title("🕵️‍♂️ Fake News Detector with Live News, PDF Report & Sharing")

# ----- THEME TOGGLE -----
# ----- THEME TOGGLE -----
def set_theme(theme):
    if theme == "Dark":
        st.markdown("""
            <style>
                html, body, [class*="css"]  {
                    background-color: #0e1117;
                    color: white;
                }
                textarea, input, select {
                    background-color: #262730 !important;
                    color: white !important;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                html, body, [class*="css"]  {
                    background-color: white;
                    color: black;
                }
                textarea, input, select {
                    background-color: white !important;
                    color: black !important;
                }
            </style>
        """, unsafe_allow_html=True)

# Sidebar theme switch
st.sidebar.title("⚙️ Settings")
theme_choice = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
set_theme(theme_choice)



# Load model and vectorizers
@st.cache_resource
def load_assets():
    word_vectorizer = joblib.load("word_vectorizer.joblib")
    char_vectorizer = joblib.load("char_vectorizer.joblib")
    model = joblib.load("fake_news_model.joblib")
    train_df = pd.read_csv("train.csv")
    return word_vectorizer, char_vectorizer, model, train_df

word_vectorizer, char_vectorizer, model, train_df = load_assets()

suspicious_keywords = [
    "miracle cure", "100% effective", "aliens", "patented formula",
    "FDA approval imminent", "no peer-reviewed studies", "private trials",
    "instant cure", "secret ingredient", "cure COVID-19"
]

# Fetch live news
def fetch_latest_news(api_key, query="news", language="en", page_size=10):
    url = f"https://newsapi.org/v2/top-headlines?q={query}&language={language}&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        return [(a["title"], a.get("description", ""), a.get("content", "")) for a in articles]
    else:
        st.error("Failed to fetch news articles.")
        return []

NEWSAPI_KEY = "5302ace1673a4e78a6ab85ba8058c3c3"

st.subheader("📰 Fetch Live News")

news_input = ""
if st.checkbox("Fetch live news articles"):
    with st.spinner("Fetching articles..."):
        news_list = fetch_latest_news(NEWSAPI_KEY)
    if news_list:
        news_titles = [title for title, _, _ in news_list]
        selected_title = st.selectbox("Select an article", news_titles)
        if selected_title:
            selected_article = next(
                (desc + "\n\n" + content for title, desc, content in news_list if title == selected_title), ""
            )
            news_input = selected_article
            st.text_area("Fetched Article", value=news_input, height=200, key="fetched_input")

# Manual input
st.subheader("✍️ Or Paste Your Own News Article")
user_input = st.text_area("Paste your news article here", height=200, key="manual_input")
if not news_input:
    news_input = user_input

if st.button("Detect"):
    if not news_input.strip():
        st.warning("Please enter or fetch some news text.")
    else:
        vec_word = word_vectorizer.transform([news_input])
        vec_char = char_vectorizer.transform([news_input])
        vec_combined = hstack([vec_word, vec_char])

        proba = model.predict_proba(vec_combined)[0]
        fake_prob, real_prob = proba[0], proba[1]
        threshold = 0.40
        pred = 0 if fake_prob > threshold else 1

        st.subheader("Result")
        result_label = "REAL" if pred == 1 else "FAKE"
        confidence = max(fake_prob, real_prob) * 100

        if pred == 1:
            st.success("✅ This news is likely REAL.")
        else:
            st.error("❌ This news is likely FAKE.")

        # Gauge chart with Plotly
        st.subheader("🔍 Model Confidence (Gauge Meter)")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Confidence that it is {result_label}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if pred == 1 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcccc"},
                    {'range': [50, 75], 'color': "#ffffcc"},
                    {'range': [75, 100], 'color': "#ccffcc"}
                ],
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

        result_text = f"Prediction: {result_label}\n"
        result_text += f"Confidence: {confidence:.2f}%\n\n"
        result_text += "Article:\n" + news_input + "\n\n"

        if pred == 0:
            st.subheader("🔍 Similar Real News Articles")
            real_articles = train_df[train_df['label'] == 1]['text'].dropna()
            real_vecs = word_vectorizer.transform(real_articles)
            similarities = cosine_similarity(vec_word, real_vecs)
            top_indices = similarities[0].argsort()[-3:][::-1]
            for i, idx in enumerate(top_indices, 1):
                snippet = real_articles.iloc[idx][:300]
                st.markdown(f"**{i}.** {snippet}")
                result_text += f"Similar Real News {i}:\n{snippet}\n\n"
            query = quote(real_articles.iloc[top_indices[0]][:100])
            st.markdown(f"[🔎 Search top match on Google](https://www.google.com/search?q={query})")

        # Suspicious keywords
        red_flags = [kw for kw in suspicious_keywords if kw in news_input.lower()]
        if red_flags:
            st.warning(f"⚠️ Contains suspicious phrase(s): {', '.join(red_flags)}")

        # PDF Export
        def clean_text(text):
            return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "Fake News Detection Report", ln=True, align="C")
            def footer(self):
                self.set_y(-15)
                self.set_font("Arial", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        cleaned_result = clean_text(result_text)
        for line in cleaned_result.strip().split("\n"):
            pdf.multi_cell(0, 10, line)

        buffer = BytesIO()
        buffer.write(pdf.output(dest='S').encode('latin1'))
        buffer.seek(0)

        st.download_button(
            label="📄 Download PDF Report",
            data=buffer,
            file_name="FakeNews_Report.pdf",
            mime="application/pdf"
        )

        # Shareable link (local mode only) and QR
        base_url = "https://fakenews.streamlit.app"  # Replace this with your app's public URL
        shared_url = f"{base_url}?text={quote(news_input)}"

        # --- Step 2: Generate QR code
        qr = qrcode.make(shared_url)
        qr_resized = qr.resize((350, 350))

        # --- Step 3: Convert QR image to bytes (for Streamlit display)
        qr_buffer = BytesIO()
        qr_resized.save(qr_buffer, format='PNG')
        qr_buffer.seek(0)

        # --- Step 4: Display QR code and shareable link in app
        st.subheader("🔗 Share This Result")
        st.image(qr_buffer, caption="📱 Scan to share", use_container_width=False)
        st.markdown(f"[🌐 Click here to open result in a new tab]({shared_url})")
