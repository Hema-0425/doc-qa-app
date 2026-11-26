import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox

from doc_qa_api import (
    load_document, split_paragraphs, Retriever, Generator
)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Document Q&A")
        root.geometry("800x650")

        self.retriever = None
        self.generator = Generator()

        top = tk.Frame(root)
        top.pack(fill="x", padx=10, pady=8)
        tk.Button(top, text="Open Document", command=self.open_doc).pack(side="left")
        self.file_label = tk.Label(top, text="No document loaded")
        self.file_label.pack(side="left", padx=10)

        q = tk.Frame(root)
        q.pack(fill="x", padx=10, pady=4)
        tk.Label(q, text="Question:").pack(side="left")
        self.entry = tk.Entry(q, font=("Arial", 12))
        self.entry.pack(side="left", fill="x", expand=True, padx=8)
        self.entry.bind("<Return>", lambda e: self.ask())
        self.btn = tk.Button(q, text="Ask", command=self.ask)
        self.btn.pack(side="left")

        tk.Label(root, text="Retrieved Paragraph(s)", anchor="w",
                 font=("Arial", 11, "bold")).pack(fill="x", padx=10, pady=(10, 0))
        self.retrieved = scrolledtext.ScrolledText(root, height=12, wrap="word")
        self.retrieved.pack(fill="both", expand=True, padx=10, pady=4)

        tk.Label(root, text="Generated Answer", anchor="w",
                 font=("Arial", 11, "bold")).pack(fill="x", padx=10, pady=(10, 0))
        self.answer = scrolledtext.ScrolledText(root, height=8, wrap="word")
        self.answer.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    def open_doc(self):
        path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf *.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        paragraphs = split_paragraphs(load_document(path))
        if not paragraphs:
            messagebox.showerror("Error", "No text found in the document.")
            return
        self.retriever = Retriever(paragraphs)
        self.file_label.config(
            text=f"{path.split('/')[-1]} ({len(paragraphs)} paragraphs)"
        )

    def ask(self):
        question = self.entry.get().strip()
        if not question:
            return
        if self.retriever is None:
            messagebox.showwarning("No document", "Open a document first.")
            return

        results = self.retriever.search(question)
        self.retrieved.delete("1.0", "end")
        for rank, (idx, score, para) in enumerate(results, 1):
            self.retrieved.insert(
                "end", f"[{rank}] (paragraph #{idx + 1}, score {score:.2f})\n{para}\n\n"
            )
        self.answer.delete("1.0", "end")
        self.answer.insert("end", "Thinking...")
        self.btn.config(state="disabled")

        contexts = [r[2] for r in results]
        threading.Thread(
            target=self.fetch_answer, args=(question, contexts), daemon=True
        ).start()

    def fetch_answer(self, question, contexts):
        text = self.generator.answer(question, contexts)
        self.root.after(0, lambda: self.show_answer(text))

    def show_answer(self, text):
        self.answer.delete("1.0", "end")
        self.answer.insert("end", text)
        self.btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()