import streamlit as st
import joblib
import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Streamlit page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)

# Preserve negation words
stop_words = set(stopwords.words("english"))
stop_words -= {"no", "not", "nor", "never"}

lemmatizer = WordNetLemmatizer()

# ---------------------------
# Text Preprocessing Functions
# ---------------------------


def remove_html(text):
    return re.sub(r"<.*?>", "", text)


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text):
    return re.sub(r"\d+", "", text)


def preprocess_text(text):
    text = text.lower()
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)

    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# ---------------------------
# Custom CSS
# ---------------------------

st.markdown(
    """
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Global overrides ── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }

/* Dark background with subtle animated gradient */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1333 25%, #24243e 50%, #1a1333 75%, #0f0c29 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Hero banner ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
    margin-bottom: 1rem;
}

.hero-icon {
    font-size: 4rem;
    display: inline-block;
    animation: float 3s ease-in-out infinite;
    filter: drop-shadow(0 0 24px rgba(139, 92, 246, 0.5));
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-12px); }
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #818cf8, #6366f1, #c084fc);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin: 0.5rem 0 0.3rem 0;
    letter-spacing: -0.02em;
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #a5b4cb;
    font-weight: 400;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(139, 92, 246, 0.3);
    box-shadow: 0 8px 40px rgba(139, 92, 246, 0.1);
}

/* ── Section label ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #a78bfa;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(167, 139, 250, 0.3), transparent);
}

/* ── Text area styling ── */
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
    resize: none !important;
}

.stTextArea textarea:focus {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
}

.stTextArea textarea::placeholder {
    color: #475569 !important;
}

.stTextArea label {
    display: none !important;
}

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.35) !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.5) !important;
    background: linear-gradient(135deg, #8b5cf6, #7c3aed, #6d28d9) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0 0.5rem 0;
    text-align: center;
    animation: resultSlideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.result-card.positive {
    border: 1px solid rgba(52, 211, 153, 0.3);
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
}

.result-card.negative {
    border: 1px solid rgba(251, 113, 133, 0.3);
    box-shadow: 0 8px 32px rgba(244, 63, 94, 0.15);
}

/* Glow effect behind result */
.result-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    border-radius: 50%;
    opacity: 0.06;
    z-index: 0;
}

.result-card.positive::before {
    background: radial-gradient(circle, #10b981, transparent 70%);
}

.result-card.negative::before {
    background: radial-gradient(circle, #f43e5e, transparent 70%);
}

@keyframes resultSlideIn {
    from {
        opacity: 0;
        transform: translateY(20px) scale(0.97);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.result-emoji {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 0.75rem;
    animation: popIn 0.4s cubic-bezier(0.68, -0.55, 0.27, 1.55) 0.2s both;
    position: relative;
    z-index: 1;
}

@keyframes popIn {
    from { transform: scale(0); opacity: 0; }
    to   { transform: scale(1); opacity: 1; }
}

.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    position: relative;
    z-index: 1;
}

.result-label.positive { color: #34d399; }
.result-label.negative { color: #fb7185; }

.result-sublabel {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
    position: relative;
    z-index: 1;
}

/* ── Confidence gauge ── */
.gauge-container {
    margin: 0 auto;
    max-width: 280px;
    position: relative;
    z-index: 1;
}

.gauge-label-top {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 0.6rem;
}

.gauge-bar-bg {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 100px;
    overflow: hidden;
    position: relative;
}

.gauge-bar-fill {
    height: 100%;
    border-radius: 100px;
    animation: fillBar 1s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    transform-origin: left;
}

.gauge-bar-fill.positive {
    background: linear-gradient(90deg, #059669, #34d399, #6ee7b7);
}

.gauge-bar-fill.negative {
    background: linear-gradient(90deg, #e11d48, #fb7185, #fda4af);
}

@keyframes fillBar {
    from { width: 0; }
}

.gauge-value {
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.5rem;
    letter-spacing: -0.02em;
}

.gauge-value.positive { color: #34d399; }
.gauge-value.negative { color: #fb7185; }

.gauge-value span {
    font-size: 1rem;
    font-weight: 500;
    opacity: 0.7;
}

/* ── Sample review chips ── */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

.chip {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 100px;
    padding: 0.45rem 1rem;
    font-size: 0.8rem;
    color: #94a3b8;
    cursor: default;
    transition: all 0.25s ease;
    line-height: 1.4;
}

.chip:hover {
    border-color: rgba(139, 92, 246, 0.4);
    color: #c4b5fd;
    background: rgba(139, 92, 246, 0.08);
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0 0 0;
}

.stat-item {
    flex: 1;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    transition: all 0.25s ease;
}

.stat-item:hover {
    border-color: rgba(139, 92, 246, 0.2);
}

.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
}

.stat-label {
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    color: #475569;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
}

.footer a {
    color: #7c3aed;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease;
}

.footer a:hover {
    color: #a78bfa;
}

/* ── Divider ── */
.soft-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.15), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── Hide Streamlit default elements ── */
.stDeployButton, .stDecoration { display: none !important; }

div[data-testid="stStatusWidget"] { display: none !important; }

/* ── Warning styling ── */
.stAlert {
    border-radius: 14px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# Hero Section
# ---------------------------

st.markdown(
    """
<div class="hero-container">
    <div class="hero-icon">🎬</div>
    <div class="hero-title">Sentiment Analyzer</div>
    <div class="hero-subtitle">
        Paste any movie review and our ML model will instantly detect
        whether the sentiment is positive or negative.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------
# Main Input Card
# ---------------------------

st.markdown(
    '<div class="glass-card">'
    '<div class="section-label">✍️ Your Review</div>',
    unsafe_allow_html=True,
)

review = st.text_area(
    "Enter your movie review",
    height=170,
    placeholder="Type or paste a movie review here…\n\ne.g. The cinematography was breathtaking and the storyline kept me hooked until the very end!",
    key="review_input",
)

predict_clicked = st.button("🔮  Analyze Sentiment")

st.markdown("</div>", unsafe_allow_html=True)




# ---------------------------
# Prediction Logic & Results
# ---------------------------

if predict_clicked:
    if not review.strip():
        st.warning("⚠️  Please enter a movie review before analyzing.")
    else:
        with st.spinner(""):
            cleaned_review = preprocess_text(review)
            review_vector = tfidf.transform([cleaned_review])

            prediction = model.predict(review_vector)[0]
            proba = model.predict_proba(review_vector)
            confidence = proba.max() * 100

            word_count = len(review.split())
            char_count = len(review)

            is_positive = prediction.lower() == "positive"
            sentiment_class = "positive" if is_positive else "negative"
            emoji = "😊" if is_positive else "😞"
            label_text = "Positive Sentiment" if is_positive else "Negative Sentiment"
            sublabel = (
                "The model detected an overall positive tone."
                if is_positive
                else "The model detected an overall negative tone."
            )

            # Result card
            st.markdown(
                f"""
            <div class="result-card {sentiment_class}">
                <span class="result-emoji">{emoji}</span>
                <div class="result-label {sentiment_class}">{label_text}</div>
                <div class="result-sublabel">{sublabel}</div>

                <div class="gauge-container">
                    <div class="gauge-label-top">Confidence Score</div>
                    <div class="gauge-bar-bg">
                        <div class="gauge-bar-fill {sentiment_class}" style="width: {confidence:.1f}%;"></div>
                    </div>
                    <div class="gauge-value {sentiment_class}">{confidence:.1f}<span>%</span></div>
                </div>

                <div class="stats-row">
                    <div class="stat-item">
                        <div class="stat-value">{word_count}</div>
                        <div class="stat-label">Words</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{char_count}</div>
                        <div class="stat-label">Characters</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{len(cleaned_review.split())}</div>
                        <div class="stat-label">Processed Tokens</div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


# ---------------------------
# Footer
# ---------------------------

st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="footer">
    Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a>
    &nbsp;·&nbsp; Powered by Scikit-learn &nbsp;·&nbsp; NLP with NLTK
</div>
""",
    unsafe_allow_html=True,
)