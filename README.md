
# Government Scheme Recommendation System

## 1. Overview
This project is a **semantic search and retrieval system** for government schemes.
It uses:
- **Sentence Transformers** for embeddings.
- **FAISS** for similarity search.
- **FLAN-T5** for natural language understanding.
- **Streamlit** for a user-friendly interface.

## 2. Workflow
1. **Install & Import Libraries**
2. **Upload CSV File** (`unique_govt_schemes.csv`)
3. **Data Preparation** – Combine columns into `full_text`
4. **Embedding Model** – `all-MiniLM-L6-v2`
5. **FAISS Index Creation**
6. **Define Search Functions** (`semantic_search`, `generate_answer`)
7. **Load FLAN-T5**
8. **User Query**
9. **Semantic Search**
10. **Result Formatting**
11. **Display Output**

## 3. Technologies Used
- FAISS – Similarity search
- Sentence Transformers – Embeddings
- Transformers (FLAN-T5) – Language model
- Pandas – Data handling
- Streamlit – Web UI

## 4. Example Output
**Query:**
```
I'm a student with low income. Any scholarship?
```
**Output:**
```
🎓 National Scholarship Scheme
- Target Group: Students from low-income families
- Support Type: Financial Aid
- Eligibility: Annual income < ₹2,50,000
- Type: National
- 🔗 More Info: [Link]
```

---

## Quick Documentation

**Purpose:**
Helps users find relevant government schemes using **semantic search** and **AI-powered query matching**.

**Steps:**
1. Load Data – Upload CSV
2. Prepare Text – Merge important fields
3. Create Embeddings – Using Sentence Transformers
4. Build FAISS Index
5. Search & Retrieve Top-k Schemes
6. Format Output
7. Display via Console/Streamlit

**Tech Used:**
- FAISS
- Sentence Transformers
- FLAN-T5
- Pandas
- Streamlit

**Example Query:**
> "I'm a student with low income. Any scholarship?"

**Example Output:**
- Scheme name
- Target group
- Support type
- Eligibility
- State/National
- Link
