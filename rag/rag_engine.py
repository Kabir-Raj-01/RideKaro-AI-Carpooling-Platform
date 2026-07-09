"""
rag/rag_engine.py
RideKaro RAG Engine — fully offline-capable
Embedding: TF-IDF + LSA (128-dim, no internet required)
Vector DB: FAISS IndexFlatIP
LLM:       Claude claude-sonnet-4-6 via Anthropic API (streaming SSE)
"""
from __future__ import annotations
import os, time, json, re, threading
from typing import List, Dict, Tuple, Optional, Generator

import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from rag.knowledge_base import get_document_texts


# ════════════════════════════════════════════════════════════
# OFFLINE EMBEDDER  (TF-IDF + LSA → L2-norm)
# ════════════════════════════════════════════════════════════

class OfflineEmbedder:
    DIM = 128  # may be reduced if corpus is small

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=8000,
            sublinear_tf=True, min_df=1, stop_words="english",
        )
        self.svd = None
        self._fitted = False

    def fit(self, corpus: List[str]):
        tfidf = self.vectorizer.fit_transform(corpus)
        max_comp = min(self.DIM, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        actual = max(8, max_comp)
        self.svd = TruncatedSVD(n_components=actual, random_state=42)
        self.svd.fit(tfidf)
        self.DIM = actual
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf).astype(np.float32)
        return normalize(dense, norm="l2")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])


# ════════════════════════════════════════════════════════════
# VECTOR STORE
# ════════════════════════════════════════════════════════════

class VectorStore:
    def __init__(self):
        self.embedder   = OfflineEmbedder()
        self.index      = None
        self.doc_ids:   List[str]  = []
        self.doc_texts: List[str]  = []
        self.doc_metas: List[Dict] = []
        self._built     = False
        self._lock      = threading.Lock()

    def build(self):
        with self._lock:
            if self._built:
                return
            docs = get_document_texts()
            texts = [t for _, t, _ in docs]
            self.embedder.fit(texts)
            embs = self.embedder.encode(texts)
            self.index = faiss.IndexFlatIP(self.embedder.DIM)
            self.index.add(embs)
            self.doc_ids   = [d[0] for d in docs]
            self.doc_texts = texts
            self.doc_metas = [d[2] for d in docs]
            self._built = True

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self._built:
            self.build()
        q = self.embedder.encode_one(query)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append({
                "id":       self.doc_ids[idx],
                "text":     self.doc_texts[idx],
                "meta":     self.doc_metas[idx],
                "score":    float(score),
                "category": self.doc_metas[idx]["category"],
                "title":    self.doc_metas[idx]["title"],
            })
        return results

    def keyword_search(self, query: str, top_k: int = 3) -> List[Dict]:
        qwords = set(re.findall(r'\w+', query.lower()))
        scored = []
        for meta in self.doc_metas:
            kws = set(w.lower() for kw in meta.get("keywords", []) for w in kw.split())
            cwds = set(re.findall(r'\w+', meta.get("content","").lower()))
            overlap = len(qwords & (kws | cwds))
            if overlap > 0:
                scored.append((overlap, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"meta": m, "text": m["title"] + "\n\n" + m["content"],
                 "score": min(s / 15.0, 1.0), "category": m["category"],
                 "title": m["title"], "id": m.get("id","")}
                for s, m in scored[:top_k]]

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        sem = self.search(query, top_k=top_k)
        kw  = self.keyword_search(query, top_k=3)
        seen, combined = set(), []
        for doc in sem:
            did = doc.get("id") or doc.get("title","")
            if did not in seen:
                seen.add(did); combined.append(doc)
        for doc in kw:
            did = doc.get("id") or doc.get("title","")
            if did not in seen:
                seen.add(did); combined.append(doc)
        return combined[:top_k + 1]


# ════════════════════════════════════════════════════════════
# CONVERSATION MEMORY
# ════════════════════════════════════════════════════════════

class ConversationMemory:
    def __init__(self, max_turns: int = 8):
        self.max_turns  = max_turns
        self._sessions: Dict[str, List[Dict]] = {}

    def add(self, session_id: str, role: str, content: str):
        hist = self._sessions.setdefault(session_id, [])
        hist.append({"role": role, "content": content})
        if len(hist) > self.max_turns * 2:
            self._sessions[session_id] = hist[-(self.max_turns * 2):]

    def get_history(self, session_id: str) -> List[Dict]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str):
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        return len(self._sessions)


# ════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════

def build_system_prompt(live_stats: Optional[Dict] = None) -> str:
    stats_block = ""
    if live_stats:
        stats_block = (
            "\nLIVE PLATFORM STATISTICS:\n"
            f"- Users: {live_stats.get('total_users', 0)}\n"
            f"- Rides posted: {live_stats.get('total_rides', 0)}\n"
            f"- Bookings: {live_stats.get('total_bookings', 0)}\n"
            f"- Completed rides: {live_stats.get('completed_rides', 0)}\n"
            f"- CO2 saved: {live_stats.get('co2_saved_kg', 0)} kg\n"
            f"- Avg rating: {live_stats.get('avg_rating', 5.0)} stars\n"
            f"- AI sessions run: {live_stats.get('match_sessions', 0)}\n"
            f"- Revenue: PKR {live_stats.get('total_revenue_pkr', 0)}\n"
        )
    return (
        "You are RideBot, the intelligent AI assistant for RideKaro — "
        "Karachi's AI-powered carpooling platform built at FAST-NUCES.\n\n"
        "RideKaro uses 11 AI agents, 10 search algorithms (A*, BFS, DFS, IDA*, Dijkstra, "
        "Greedy, Hill Climbing, Beam Search, BiDir BFS, UCS), Genetic Algorithm, "
        "CSP+AC-3, Bayesian Naive Bayes, and ML (Random Forest, Gradient Boosting, PCA, K-Means).\n\n"
        + stats_block +
        "\nYOUR ROLE:\n"
        "- Help users understand and use the RideKaro platform\n"
        "- Answer questions about Karachi traffic, routes, and travel\n"
        "- Explain AI/ML concepts used in the platform clearly\n"
        "- Guide users through booking rides, posting rides, running the AI pipeline\n"
        "- Provide Karachi-specific safety advice\n"
        "- Calculate fare estimates (PKR 305/litre petrol, 2024)\n"
        "- Be concise and conversational (2-4 paragraphs)\n"
        "- Use bullet points for step-by-step guides\n"
        "- Never fabricate statistics — use retrieved context only\n"
        "- Respond in English or Roman Urdu matching the user\n"
    )


def build_context_block(docs: List[Dict]) -> str:
    if not docs:
        return "No specific documentation retrieved."
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[Doc {i}] [{doc.get('category','general').upper()}] "
            f"{doc.get('title','Document')} (similarity: {doc.get('score',0):.2f})\n"
            f"{doc.get('text','')}"
        )
    return "\n\n---\n\n".join(parts)


# ════════════════════════════════════════════════════════════
# RAG ENGINE
# ════════════════════════════════════════════════════════════

class RAGEngine:
    def __init__(self):
        self.vector_store = VectorStore()
        self.memory       = ConversationMemory(max_turns=8)
        self._ready       = False

    def warm_up(self):
        def _build():
            self.vector_store.build()
            self._ready = True
        threading.Thread(target=_build, daemon=True).start()

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _client(self):
        import anthropic
        return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self._ready:
            self.vector_store.build()
            self._ready = True
        return self.vector_store.hybrid_search(query, top_k=top_k)

    def _build_messages(self, query: str, docs: List[Dict], session_id: str) -> List[Dict]:
        ctx = build_context_block(docs)
        user_msg = (
            "RETRIEVED KNOWLEDGE BASE CONTEXT:\n" + ctx +
            "\n\n---\n\nUSER QUESTION: " + query +
            "\n\nPlease answer using the retrieved context and your knowledge of RideKaro."
        )
        return self.memory.get_history(session_id) + [{"role": "user", "content": user_msg}]

    def _fallback_answer(self, query: str, docs: List[Dict], error: str = "") -> str:
        """Return KB-grounded answer without LLM when API key missing or error."""
        key_set = bool(os.environ.get("ANTHROPIC_API_KEY",""))
        if not key_set:
            prefix = (
                "🔑 **API key not configured.** Set ANTHROPIC_API_KEY to enable full AI responses. "
                "Here is what the knowledge base found:\n\n"
            )
        elif error:
            prefix = (
                "⚠️ **AI generation error.** "
                "Here is what the knowledge base retrieved:\n\n"
            )
        else:
            prefix = ""

        if not docs:
            return prefix + "No relevant documents found. Try asking about Karachi traffic, fares, safety, or platform how-to guides."

        parts = [prefix]
        for doc in docs[:3]:
            title   = doc.get("title", "Document")
            score   = doc.get("score", 0)
            text    = doc.get("text", "")
            content = text.split("\n\n", 1)[-1] if "\n\n" in text else text
            excerpt = content[:280].strip()
            parts.append(f"**{title}** (match: {score:.0%})\n{excerpt}...\n")
        return "\n".join(parts)

    # ── NON-STREAMING ─────────────────────────────────────────
    def generate(self, query: str, session_id: str = "default",
                 live_stats: Optional[Dict] = None, top_k: int = 5) -> Dict:
        t0   = time.time()
        docs = self.retrieve(query, top_k=top_k)
        msgs = self._build_messages(query, docs, session_id)

        api_key = os.environ.get("ANTHROPIC_API_KEY","")
        if not api_key:
            answer = self._fallback_answer(query, docs)
            self.memory.add(session_id, "user", query)
            self.memory.add(session_id, "assistant", answer)
            return {
                "answer": answer,
                "retrieved_docs": [{"title": d["title"], "category": d["category"],
                                    "score": round(d["score"],3)} for d in docs],
                "tokens_used": 0,
                "latency_ms":  round((time.time()-t0)*1000, 1),
                "session_id":  session_id,
                "doc_count":   len(docs),
                "fallback":    True,
            }

        try:
            resp = self._client().messages.create(
                model="claude-sonnet-4-6", max_tokens=1024,
                system=build_system_prompt(live_stats), messages=msgs,
            )
            answer = resp.content[0].text
            tokens = resp.usage.input_tokens + resp.usage.output_tokens
        except Exception as e:
            answer = self._fallback_answer(query, docs, str(e))
            tokens = 0

        self.memory.add(session_id, "user", query)
        self.memory.add(session_id, "assistant", answer)
        return {
            "answer": answer,
            "retrieved_docs": [{"title": d["title"], "category": d["category"],
                                "score": round(d["score"],3)} for d in docs],
            "tokens_used": tokens,
            "latency_ms":  round((time.time()-t0)*1000, 1),
            "session_id":  session_id,
            "doc_count":   len(docs),
        }

    # ── STREAMING SSE ─────────────────────────────────────────
    def stream_generate(self, query: str, session_id: str = "default",
                        live_stats: Optional[Dict] = None,
                        top_k: int = 5) -> Generator[str, None, None]:
        docs = self.retrieve(query, top_k=top_k)

        # Send retrieved sources immediately
        src_payload = json.dumps({
            "type": "sources",
            "docs": [{"title": d["title"], "category": d["category"],
                      "score": round(d["score"],3)} for d in docs]
        })
        yield "data: " + src_payload + "\n\n"

        # Fallback if no API key
        api_key = os.environ.get("ANTHROPIC_API_KEY","")
        if not api_key:
            fallback = self._fallback_answer(query, docs)
            for word in fallback.split(" "):
                payload = json.dumps({"type": "token", "content": word + " "})
                yield "data: " + payload + "\n\n"
            self.memory.add(session_id, "user", query)
            self.memory.add(session_id, "assistant", fallback)
            done_payload = json.dumps({"type": "done", "session_id": session_id, "fallback": True})
            yield "data: " + done_payload + "\n\n"
            return

        msgs = self._build_messages(query, docs, session_id)
        full = []
        try:
            with self._client().messages.stream(
                model="claude-sonnet-4-6", max_tokens=1024,
                system=build_system_prompt(live_stats), messages=msgs,
            ) as stream:
                for chunk in stream.text_stream:
                    full.append(chunk)
                    payload = json.dumps({"type": "token", "content": chunk})
                    yield "data: " + payload + "\n\n"
        except Exception as e:
            err_msg = self._fallback_answer(query, docs, str(e))
            for word in err_msg.split(" "):
                payload = json.dumps({"type": "token", "content": word + " "})
                yield "data: " + payload + "\n\n"
            done_payload = json.dumps({"type": "done", "session_id": session_id})
            yield "data: " + done_payload + "\n\n"
            return

        answer = "".join(full)
        self.memory.add(session_id, "user", query)
        self.memory.add(session_id, "assistant", answer)
        done_payload = json.dumps({"type": "done", "session_id": session_id})
        yield "data: " + done_payload + "\n\n"

    def clear_session(self, session_id: str):
        self.memory.clear(session_id)

    def get_stats(self) -> Dict:
        return {
            "ready":           self._ready,
            "doc_count":       len(self.vector_store.doc_ids),
            "active_sessions": self.memory.session_count(),
            "model":           "claude-sonnet-4-6",
            "embedder":        "TF-IDF + LSA (TruncatedSVD, offline)",
            "vector_dim":      self.vector_store.embedder.DIM,
            "index_type":      "FAISS IndexFlatIP (cosine)",
        }


# ── SINGLETON ─────────────────────────────────────────────────
_rag: Optional[RAGEngine] = None

def get_rag_engine() -> RAGEngine:
    global _rag
    if _rag is None:
        _rag = RAGEngine()
        _rag.warm_up()
    return _rag
