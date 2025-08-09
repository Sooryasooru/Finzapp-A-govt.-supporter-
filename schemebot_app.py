
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime

# Load data
df = pd.read_csv("unique_govt_schemes.csv")
df["full_text"] = (
    df["Scheme Name"].fillna("") + ". " +
    df["Target Group"].fillna("") + ". " +
    df["Support Type"].fillna("") + ". " +
    df["Eligibility"].fillna("") + ". " +
    df["State/National"].fillna("")
)

# Load model and index
embedder = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = embedder.encode(df["full_text"].tolist(), convert_to_tensor=False)
dimension = len(corpus_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(corpus_embeddings))

# FAISS search function
def generate_answer(query, top_k=3):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_embedding), top_k)

    results = []
    for idx in I[0]:
        scheme = df.iloc[idx]
        text = (
            f"**{scheme['Scheme Name']}**\n"
            f"👥 Target Group: {scheme['Target Group']}\n"
            f"💡 Support Type: {scheme['Support Type']}\n"
            f"✅ Eligibility: {scheme['Eligibility']}\n"
            f"📍 Type: {scheme['State/National']}\n"
            f"🔗 [More Info]({scheme['Link']})"
        )
        results.append(text)
    
    return results

# Time-based greeting
hour = datetime.now().hour
if hour < 12:
    greeting = "Good morning ☀️"
elif hour < 18:
    greeting = "Good afternoon 🌞"
else:
    greeting = "Good evening 🌙"

# Streamlit config
st.set_page_config(page_title="SchemeBot 💬", layout="centered")
st.title("👩‍💼 Financial Scheme Chat Assistant")

# Initialize session state for chat steps
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.category = ""
    st.session_state.financial_input = ""
    st.session_state.final_prompt = ""

# Step 1: Greeting and Category
if st.session_state.step == 1:
    user_input = st.text_input("You: 👋", placeholder="Say hi or hello...")
    if user_input:
        st.markdown(f"**Bot:** {greeting}! I'm your friendly assistant 🤖")
        st.markdown("Tell me who you are: student, farmer, entrepreneur, housewife, graduate, etc.")
        st.session_state.step = 2

# Step 2: Category selection
elif st.session_state.step == 2:
    user_input = st.text_input("You:", placeholder="I'm a student / I'm a farmer / etc...")
    if user_input:
        st.session_state.category = user_input
        st.markdown(f"**Bot:** Awesome! 👏 You're a {user_input}.")
        st.markdown("How much income or savings do you currently have?")
        st.session_state.step = 3

# Step 3: Financial input
elif st.session_state.step == 3:
    user_input = st.text_input("You:", placeholder="e.g., I have 1 lakh / no income / etc...")
    if user_input:
        st.session_state.financial_input = user_input
        st.markdown("**Bot:** Okay, let me search for some government schemes that might help you 💡")
        st.session_state.final_prompt = f"I'm a {st.session_state.category} with {user_input}. Suggest schemes."
        st.session_state.step = 4

# Step 4: Show Results
elif st.session_state.step == 4:
    results = generate_answer(st.session_state.final_prompt)
    st.markdown("**Bot:** Based on your situation, here are some schemes you might be eligible for:")
    for res in results:
        st.markdown(res)
        st.markdown("---")
    st.success("All the best for your journey! 🚀 Feel free to ask again anytime.")
    if st.button("🔄 Start Over"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()
