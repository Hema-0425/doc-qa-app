import os
import re
import sys
import math
from collections import Counter
from google import genai
from google.genai import types

LLM_MODEL = "gemini-3.5-flash-lite"
TOP_K = 3
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "and",
             "or", "for", "on", "with", "as", "by", "at", "it", "this", "that",
             "what", "which", "who", "how", "why", "does", "do", "be", "from"}


def load_document(path):
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_paragraphs(text, min_len=40, max_words=150):
    chunks = []
    for p in re.split(r"\n\s*\n", text):
        p = re.sub(r"\s+", " ", p).strip()
        words = p.split()
        if len(words) <= max_words:
            if len(p) >= min_len:
                chunks.append(p)
        else:
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
    return chunks


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class Retriever:
    def __init__(self, paragraphs):
        self.paragraphs = paragraphs
        counts = [Counter(tokenize(p)) for p in paragraphs]
        n = len(paragraphs)
        df = Counter()
        for c in counts:
            df.update(c.keys())
        self.idf = {w: math.log((n + 1) / (d + 1)) + 1 for w, d in df.items()}
        self.vecs = [self._vec(c) for c in counts]

    def _vec(self, counts):
        v = {w: tf * self.idf.get(w, 0) for w, tf in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {w: x / norm for w, x in v.items()}

    def search(self, question, k=TOP_K):
        q = self._vec(Counter(tokenize(question)))
        scores = [sum(x * v.get(w, 0) for w, x in q.items()) for v in self.vecs]
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, scores[i], self.paragraphs[i]) for i in idx]


class Generator:
    def __init__(self):
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("Set the GOOGLE_API_KEY environment variable first.")
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=120000),
        )

    def answer(self, question, contexts):
        context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            "Answer the question using the context below. "
            "If the context only partly covers it, answer with what it says. "
            "Say 'Not found in the document.' only if the context has nothing related. "
            "Be concise.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}"
        )
        try:
            response = self.client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            return f"ERROR: {e}"


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python doc_qa_api.py <document.txt | document.pdf>")

    paragraphs = split_paragraphs(load_document(sys.argv[1]))
    if not paragraphs:
        sys.exit("No text found in the document.")
    print(f"Document split into {len(paragraphs)} paragraphs.")

    retriever = Retriever(paragraphs)
    generator = Generator()

    print("\nAsk questions about the document. Type 'exit' to quit.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        results = retriever.search(question)
        answer = generator.answer(question, [r[2] for r in results])

        print("\n--- Retrieved Paragraph(s) ---")
        for rank, (idx, score, para) in enumerate(results, 1):
            print(f"[{rank}] (paragraph #{idx + 1}, score {score:.2f})\n{para}\n")
        print("--- Generated Answer ---")
        print(answer)
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()