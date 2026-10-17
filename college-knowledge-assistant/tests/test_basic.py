from __future__ import annotations

from services.document_service import DocumentService
from services.rag_service import RAGService
from services.retrieval_service import RetrievalService
from utils.validators import is_valid_query, validate_upload


class DummyUpload:
    def __init__(self, name: str, payload: bytes):
        self.name = name
        self._payload = payload

    def read(self):
        return self._payload


class DummyEmbeddings:
    def embed(self, text: str):
        return [0.0, 0.0, 0.0]


class DummyVectorStore:
    def __init__(self, results):
        self.results = results

    def query(self, query_vector, top_k=5, where=None):
        return self.results


class DummyDocumentService:
    def __init__(self, docs):
        self.docs = docs

    def list_documents(self):
        return self.docs


class DummyLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def generate(self, prompt: str, model_name: str):
        if self.error:
            raise RuntimeError(self.error)
        return self.response


def build_rag_service(docs, retrieval_results, llm_response=None, llm_error=None):
    service = RAGService.__new__(RAGService)
    service.document_service = DummyDocumentService(docs)
    service.llm_service = DummyLLM(llm_response, llm_error)
    service.retrieval_service = type("DummyRetrievalService", (), {"retrieve": lambda self, query, top_k=5, subject_filter=None: retrieval_results})()
    service.last_retrieval_debug = None
    return service


def test_valid_query_detection() -> None:
    assert is_valid_query("What is the syllabus?") is True
    assert is_valid_query("   ") is False


def test_upload_validation_rejects_invalid_extension() -> None:
    try:
        validate_upload("notes.txt", 100)
        assert False, "Expected validation failure"
    except ValueError:
        pass


def test_document_indexing_stores_embedded_chunks() -> None:
    import fitz
    import uuid
    from pathlib import Path

    pdf_name = f"tmp_upload_test_{uuid.uuid4().hex}.pdf"
    pdf_path = Path(pdf_name)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "College knowledge assistant upload test content.")
    doc.save(pdf_path)
    doc.close()

    class PdfUpload:
        def __init__(self, path: Path):
            self.name = path.name
            self._bytes = path.read_bytes()

        def read(self):
            return self._bytes

    service = DocumentService(upload_dir="temp_uploads")
    service.documents_path.unlink(missing_ok=True)

    try:
        result = service.index_pdf_upload(PdfUpload(pdf_path))
        assert result["chunks"] >= 1
        assert result["vector_ids"]
    finally:
        pdf_path.unlink(missing_ok=True)
        service.documents_path.unlink(missing_ok=True)


def test_chunking_keeps_content() -> None:
    service = DocumentService(upload_dir="temp_uploads")
    chunks = service.chunk_text("A " * 1000, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(chunk["content"] for chunk in chunks)


def test_grounded_answer_uses_retrieved_document_context() -> None:
    retrieval_results = [{
        "filename": "syllabus.pdf",
        "page_number": 2,
        "content": "The exam is scheduled for Friday at 9:00 AM.",
        "similarity": 0.91,
        "distance": 0.09,
    }]
    service = build_rag_service([{"filename": "syllabus.pdf"}], retrieval_results, "According to syllabus.pdf (page 2), the exam is scheduled for Friday at 9:00 AM.")

    result = service.answer_question("When is the exam?", "qwen2.5:3b")

    assert result["grounded"] is True
    assert result["sources"][0]["filename"] == "syllabus.pdf"
    assert "syllabus.pdf" in result["answer"].lower()


def test_answer_absent_from_documents_abstains() -> None:
    service = build_rag_service([{"filename": "syllabus.pdf"}], [], "The model would answer here.")

    result = service.answer_question("What is the attendance policy?", "qwen2.5:3b")

    assert result["grounded"] is False
    assert result["answer"] == "I couldn't find sufficient information to answer this question in the uploaded college documents."


def test_irrelevant_retrieval_result_is_filtered() -> None:
    vector_store = DummyVectorStore([
        {"content": "The sky is blue on most days.", "metadata": {"filename": "random.pdf", "page_number": 1}, "distance": 0.95}
    ])
    retrieval = RetrievalService(vector_store, DummyEmbeddings())

    assert retrieval.retrieve("What is the attendance policy?") == []


def test_empty_vector_database_returns_no_results() -> None:
    vector_store = DummyVectorStore([])
    retrieval = RetrievalService(vector_store, DummyEmbeddings())

    assert retrieval.retrieve("What is the exam date?") == []


def test_question_with_insufficient_evidence_abstains() -> None:
    retrieval_results = [{
        "filename": "notes.pdf",
        "page_number": 4,
        "content": "Some exam rules may apply.",
        "similarity": 0.30,
        "distance": 0.70,
    }]
    service = build_rag_service([{"filename": "notes.pdf"}], retrieval_results, "There is some information that may be relevant.")

    result = service.answer_question("What is the exam policy?", "qwen2.5:3b")

    assert result["grounded"] is False
    assert result["answer"] == "I couldn't find sufficient information to answer this question in the uploaded college documents."


def test_source_citation_must_match_retrieved_document() -> None:
    retrieval_results = [{
        "filename": "student-handbook.pdf",
        "page_number": 11,
        "content": "Students may appeal final grades within 10 days.",
        "similarity": 0.94,
        "distance": 0.06,
    }]
    service = build_rag_service([{"filename": "student-handbook.pdf"}], retrieval_results, "According to student-handbook.pdf (page 11), students may appeal final grades within 10 days.")

    result = service.answer_question("What is the appeal window?", "qwen2.5:3b")

    assert result["grounded"] is True
    assert "student-handbook.pdf" in result["answer"].lower()
    assert any(source["filename"] == "student-handbook.pdf" for source in result["sources"])


def test_model_failure_is_explicitly_handled() -> None:
    retrieval_results = [{
        "filename": "policy.pdf",
        "page_number": 7,
        "content": "Late submissions lose 5 points.",
        "similarity": 0.90,
        "distance": 0.10,
    }]
    service = build_rag_service([{"filename": "policy.pdf"}], retrieval_results, llm_error="model unavailable")

    result = service.answer_question("What happens for late submissions?", "qwen2.5:3b")

    assert result["grounded"] is False
    assert "model unavailable" in result["error"]


def test_grounding_is_consistent_across_llm_providers() -> None:
    retrieval_results = [{
        "filename": "academic-regulations.pdf",
        "page_number": 5,
        "content": "Students must complete 75% of coursework to pass.",
        "similarity": 0.88,
        "distance": 0.12,
    }]

    service1 = build_rag_service([{"filename": "academic-regulations.pdf"}], retrieval_results, "According to academic-regulations.pdf (page 5), students must complete 75% of coursework to pass.")
    service2 = build_rag_service([{"filename": "academic-regulations.pdf"}], retrieval_results, "According to academic-regulations.pdf (page 5), students must complete 75% of coursework to pass.")

    result1 = service1.answer_question("What is the pass requirement?", "ollama-model")
    result2 = service2.answer_question("What is the pass requirement?", "api-model")

    assert result1["grounded"] is True
    assert result2["grounded"] is True
    assert result1["answer"] == result2["answer"]
