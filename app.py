import streamlit as st
import pickle
import base64
import PyPDF2  # for PDF reading
import io     # for handling file streams

# -------------------------------
# ✅ Load Model and Vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------------
# ✅ Background Image Function
# -------------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_file):
    base64_img = get_base64(image_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# ✅ Set Background
# -------------------------------
set_background("background.jpg")  # Make sure this image is in the same folder

# -------------------------------
# ✅ Title
# -------------------------------
st.title("📰 Fake News Detection Web App")

# -------------------------------
# ✅ Input Option
# -------------------------------
option = st.radio("Choose Input Type", ("✍️ Manual Text", "📄 Upload PDF/TXT"))

news_text = ""

if option == "✍️ Manual Text":
    news_text = st.text_area("Enter News Text:", height=200)

elif option == "📄 Upload PDF/TXT":
    uploaded_file = st.file_uploader("Upload a PDF or .txt file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            news_text = text
        elif uploaded_file.name.endswith(".txt"):
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            news_text = stringio.read()

        st.success("✅ File successfully read!")
        st.text_area("Extracted Text:", news_text, height=200)

# -------------------------------
# ✅ Prediction
# -------------------------------
if st.button("Check if Fake or Real"):
    if news_text.strip() == "":
        st.warning("Please provide text (manual or file) to analyze.")
    else:
        input_vector = vectorizer.transform([news_text])
        prediction = model.predict(input_vector)[0]
        if prediction == 1:
            st.error("❌ This is **Fake News**")
        else:
            st.success("✅ This is **Real News**")
