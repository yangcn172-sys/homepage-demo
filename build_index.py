import os, re, json, pickle
import pdfplumber
import jieba
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# 改成你本地实际目录（Windows 用正斜杠或 r""）
PDF_DIR = r"C:\Users\yangz\Desktop\AI\finance_rag\pdf_reports"
OUT_DIR = r"C:\Users\yangz\Desktop\AI\finance_rag\data"
os.makedirs(OUT_DIR, exist_ok=True)

SECTION_PAT = re.compile(r'^(第[一二三四五六七八九十]+[节章]\s+[^\n]{2,40})$', re.M)

def extract_company(fn):
    # 从 "烽火通信2026半年度报告.pdf" 提取 "烽火通信"
    name = re.sub(r'20\d{2}.*', '', fn.replace('.pdf', ''))
    return name.strip()

def parse_pdf(path, company):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            table_texts = []
            for t in tables:
                rows = []
                for row in t:
                    row = [(c or "").replace("\n", " ").strip() for c in row]
                    rows.append(" | ".join(row))
                table_texts.append("\n".join(rows))
            full = text + "\n\n" + "\n\n".join(table_texts)
            pages.append({"company": company, "page": i, "text": full})

    current = "未知"
    for p in pages:
        m = SECTION_PAT.search(p["text"])
        if m:
            current = m.group(1)
        p["section"] = current
    return pages

def chunk_pages(pages, size=800, overlap=120):
    chunks = []
    for p in pages:
        text = p["text"]
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + size, len(text))
            seg = text[start:end]
            chunks.append({
                "chunk_id": f'{p["company"]}-{p["page"]}-{idx}',
                "company": p["company"],
                "section": p["section"],
                "page": p["page"],
                "text": f'《{p["company"]}2026半年报·{p["section"]}·第{p["page"]}页》\n{seg}'
            })
            if end == len(text):
                break
            start = end - overlap
            idx += 1
    return chunks

def main():
    all_chunks = []
    files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]
    print(f"发现 {len(files)} 份 PDF")
    for fn in files:
        company = extract_company(fn)
        path = os.path.join(PDF_DIR, fn)
        print("解析：", fn, "→ 公司名：", company)
        pages = parse_pdf(path, company)
        all_chunks.extend(chunk_pages(pages))

    with open(os.path.join(OUT_DIR, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("切块数：", len(all_chunks))

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    texts = [c["text"] for c in all_chunks]
    emb = model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
    emb = np.asarray(emb).astype("float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))

    tokenized = [list(jieba.cut(t)) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(OUT_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print("索引完成")

if __name__ == "__main__":
    main()