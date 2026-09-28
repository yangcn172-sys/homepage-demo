import os, json, pickle
import streamlit as st
import faiss
import jieba
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="CPO产业链财报 RAG", layout="wide")
st.title("CPO产业链财报 RAG 问答知识库")

@st.cache_resource
def load_all():
    """加载切块、FAISS 索引、BM25 索引和嵌入模型"""
    chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]
    index = faiss.read_index("data/faiss.index")
    bm25 = pickle.load(open("data/bm25.pkl", "rb"))
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    return chunks, index, bm25, model

chunks, index, bm25, model = load_all()

def hybrid_search(query, top_k=8):
    """向量 + BM25 混合检索，RRF 融合"""
    # 向量检索
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k * 3)
    vec_hits = [(int(i), float(d)) for d, i in zip(D[0], I[0])]

    # BM25 检索
    q_tokens = list(jieba.cut(query))
    bm_scores = bm25.get_scores(q_tokens)
    bm_hits = sorted([(i, float(s)) for i, s in enumerate(bm_scores)], key=lambda x: -x[1])[:top_k * 3]

    # RRF 融合
    rrf = {}
    for rank, (i, _) in enumerate(vec_hits):
        rrf[i] = rrf.get(i, 0) + 1 / (60 + rank + 1)
    for rank, (i, _) in enumerate(bm_hits):
        rrf[i] = rrf.get(i, 0) + 1 / (60 + rank + 1)

    final = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
    return [chunks[i] for i, _ in final]

# 输入框
q = st.text_input("输入问题，例如：中际旭创2026上半年营业收入、归母净利润分别是多少？")

if q:
    hits = hybrid_search(q, top_k=8)

    st.subheader("召回块")
    for i, h in enumerate(hits):
        with st.expander(f"[{i+1}] {h['company']} · {h['section']} · 第{h['page']}页"):
            st.write(h["text"])

    st.subheader("回答")
    ctx = "\n\n".join([f"[{i+1}] {h['text']}" for i, h in enumerate(hits)])
    prompt = f"""只依据下面材料回答，每句话带角标[1][2]。材料没有就说“材料未覆盖”。
问题：{q}
材料：
{ctx}
"""

    # 从环境变量读取 API Key（推荐）
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.warning("未设置环境变量 DEEPSEEK_API_KEY，无法生成回答。请先设置环境变量。")
    else:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(resp.choices[0].message.content)