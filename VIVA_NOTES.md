# Viva Notes — Invertis Campus Assistant

## What is RAG?

Retrieval-Augmented Generation combines information retrieval with a generative language model. Instead of asking a model to rely only on learned parameters, the application searches a curated knowledge collection at question time and supplies the most relevant passages as context. This improves freshness, traceability, and refusal behavior for facts not present in the collection.

## Pipeline, step by step

1. `Scraper` requests and parses the official site's `robots.txt`; if policy cannot be verified it refuses to crawl.
2. The crawler seeds useful public routes, checks host and robots rules for every URL, caches HTML, applies request delay, and yields normalized `DocumentRecord` objects.
3. `WebLoader` and `PDFLoader` normalize website/PDF content to LangChain `Document` objects with source metadata. Text and Markdown files use the local ingestion path.
4. `DocumentProcessor` cleans whitespace and lazily splits documents into overlapping chunks while preserving URL/title/category/page metadata.
5. Gemini embeddings convert chunks to vectors. Chroma persists vectors locally under `data/index/`.
6. A question may be rewritten into a standalone query using recent conversation turns, then Chroma retrieves diversified candidates with category-filtered MMR.
7. Retrieval confidence and empty-index checks enforce the safe fallback before generation.
8. LCEL composes the strict prompt, Gemini chat model, and string output parser. The UI consumes the model's token stream.
9. Source URL/document and PDF page citations appear with each answer. Feedback and query analytics are stored in SQLite.

## Why chunking, embeddings, and a vector database?

- **Chunking** limits context to coherent passages and supports long documents beyond a model's prompt capacity. Overlap retains facts that cross chunk boundaries.
- **Embeddings** map semantically similar text and questions to nearby vectors, so paraphrases can retrieve useful material even without exact keyword overlap.
- **Vector database (Chroma)** stores embeddings with metadata, persists them across sessions, and supports efficient similarity/MMR retrieval and category constraints.

## Advanced Python concepts in this code

| Concept | Where it appears |
|---|---|
| OOP and encapsulation | `Scraper`, `DocumentProcessor`, `VectorStoreManager`, `RAGEngine`, `ConversationMemory`, `FeedbackStore` |
| Abstract base classes and polymorphism | `loaders.py`: `BaseLoader`, `WebLoader`, `PDFLoader` |
| Dataclasses | `utils.py`: `DocumentRecord`, `Source`; `rag.py`: `AnswerResult`, `PreparedAnswer`, `ConversationMemory` |
| Decorators | `decorators.py`: `timer`, `retry`, `log_calls`, `cache`; applied in crawler and RAG code |
| Generators / lazy evaluation | `Scraper.crawl`, `DocumentProcessor.chunks`, RAG token iterators |
| Context managers | `utils.py`: `timed_block`, `safe_text_file`; SQLite uses connection context managers |
| Type hints | Package APIs, generic decorators, and domain models |
| Exception hierarchy | `exceptions.py`; errors remain distinguishable for graceful UI/CLI handling |
| Comprehensions | Metadata conversion, source selection, and UI/admin tabular rendering |
| pathlib | Configured paths and file discovery in ingestion |
| argparse | `ingest.main` CLI parser |
| SQLite and parameterized SQL | `feedback.py` |
| LCEL composition | `rag.py`: `prompt | model | StrOutputParser()` |
| Pydantic settings | `config.py`: environment validation and `.env` loading |

## 10 likely viva questions and answers

1. **What problem does RAG solve?**  
   It grounds model output in retrievable documents so answers can use up-to-date evidence and show citations, rather than relying solely on model memory.

2. **Why are documents split into chunks?**  
   Smaller passages fit the model context window and improve retrieval precision. Overlap reduces information loss across chunk boundaries.

3. **What is an embedding?**  
   It is a numeric vector representation of text. Semantic similarity between question and passage can be approximated with vector distance/similarity.

4. **Why use Chroma instead of a Python list?**  
   Chroma persists vector records and efficiently performs similarity retrieval with metadata filtering; a list would require manual, linear search and would not scale well.

5. **What does MMR add?**  
   Maximal Marginal Relevance balances relevance with diversity so retrieved chunks are less redundant.

6. **What does LCEL mean here?**  
   LangChain Expression Language composes runnables, e.g. prompt → chat model → output parser, using a clear pipeline interface that supports invocation and streaming.

7. **How are follow-up questions handled?**  
   `ConversationMemory` retains a bounded number of exchanges. A rewrite prompt uses history only to clarify references, then retrieval uses the resulting standalone query.

8. **How does the bot avoid hallucinating?**  
   It retrieves source text, uses a strict system prompt, measures retrieval confidence, and returns a fallback with official contact/website when evidence is absent or weak. Citations show provenance.

9. **Why use an abstract `BaseLoader`?**  
   It defines a common `load()` contract. Web and PDF loaders can be substituted by ingestion code while preserving source-specific parsing behavior.

10. **How do `retry` and `cache` decorators differ?**  
    `retry` repeats failed transient operations with exponential delay; `cache` stores successful function results to avoid repeating identical work. Retries are applied only to web requests; query rewrite caching is bounded.
