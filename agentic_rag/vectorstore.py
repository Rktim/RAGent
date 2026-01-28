from rank_bm25 import BM25Okapi
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

class HybridRetriever:
    def __init__(self, docs, embeddings, k=5):
        self.k = k
        self.embeddings = embeddings

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        if isinstance(docs[0], str):
            self.chunks = splitter.create_documents(docs)
        else:
            self.chunks = splitter.split_documents(docs)

        dim = len(embeddings.embed_query("hello"))
        index = faiss.IndexFlatL2(dim)

        self.faiss = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.faiss.add_documents(self.chunks)

        tokenized = [c.page_content.lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str):
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[: self.k]

        bm25_docs = [self.chunks[i] for i in bm25_top]
        faiss_docs = self.faiss.similarity_search(query, k=self.k)

        seen = set()
        merged = []

        for d in bm25_docs + faiss_docs:
            key = d.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append(d)

        return merged[: self.k]
