from backend import retrieval
from pypdf import PdfWriter


def test_real_pdf_chroma_ingestion_reuse_and_search(tmp_path, monkeypatch):
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    data = tmp_path / "data"
    data.mkdir()
    for filename, _, _ in retrieval.PDFS.values():
        writer = PdfWriter()
        page = writer.add_blank_page(width=600, height=800)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 50 700 Td (Security deposit evidence for a rental document.) Tj ET")
        page[NameObject("/Contents")] = writer._add_object(content)
        writer.write(str(data / filename))
    monkeypatch.setattr(retrieval, "ROOT", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-network")
    calls = []
    class FakeEmbedding:
        def embed_documents(self, texts):
            calls.append(texts)
            return [[1.0, 0.5, 0.25] for _ in texts]
        def embed_query(self, text):
            return [1.0, 0.5, 0.25]
    monkeypatch.setattr(retrieval, "embedding_model", FakeEmbedding)
    from langsmith import tracing_context
    with tracing_context(enabled=False):
        result = retrieval.build_indexes()
        assert result == {"policy": 1, "cases": 1, "reused": False}
        assert retrieval.index_ready()
        assert retrieval.build_indexes()["reused"] is True
    assert len(calls) == 2
    docs = retrieval.search("policy", "deposit", 5)
    assert len(docs) == 1
    assert docs[0]["page"] == 1
    assert "Security deposit" in docs[0]["text"]
    assert docs[0]["source"] == "tenants_rights.pdf"
