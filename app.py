import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load data
df = pd.read_csv("unique_govt_schemes.csv")
df["full_text"] = (
    df["Scheme Name"].fillna("") + ". " +
    df["Target Group"].fillna("") + ". " +
    df["Support Type"].fillna("") + ". " +
    df["Eligibility"].fillna("") + ". " +
    df["State/National"].fillna("")
)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = embedder.encode(df["full_text"].tolist(), convert_to_tensor=False)
dimension = len(corpus_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(corpus_embeddings))

# Search function
def get_scheme_summary(prompt, top_k=3):
    query_embedding = embedder.encode([prompt], convert_to_tensor=False)
    D, I = index.search(np.array(query_embedding), top_k)
    summaries = []
    for idx in I[0]:
        scheme = df.iloc[idx]
        summaries.append(
            f"🔹 **{scheme['Scheme Name']}** offers {scheme['Support Type']} to {scheme['Target Group']}.\n"
            f"Eligibility: {scheme['Eligibility']}.\n"
            f"Type: {scheme['State/National']} level.\n"
            f"🔗 [More Info]({scheme['Link']})\n"
        )
    return summaries

# Streamlit UI
st.set_page_config(page_title="Finzapp", layout="centered")
st.title(" Smart Government Scheme Finder")

st.markdown("Answer 3 simple questions to find the best schemes for you:")

# Step 1: User type
category = st.text_input("1️⃣ Who are you? (e.g., student, farmer, housewife, etc.)")

# Step 2: Income
income = st.text_input("2️⃣ What is your current income or savings? (e.g., No income, ₹1 lakh)")

# Step 3: Need
need = st.text_input("3️⃣ What kind of help do you need? (e.g., loan, scholarship, training, etc.)")

# Trigger search
if category and income and need:
    query = f"I am a {category} with {income} and I need {need}."
    st.markdown("As per your search Finzapp is searching for you...")

    results = get_scheme_summary(query)

    st.markdown("###  Best Matches for You:")
    for scheme in results:
        st.markdown(scheme)
        st.markdown("---")

    st.success("Thank you for using Finzapp, All the best dear! ")
