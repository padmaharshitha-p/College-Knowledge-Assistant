from pathlib import Path
import fitz

from services.document_service import DocumentService

pdf_path = Path(r"C:\Users\harsh\College\college-knowledge-assistant\test_upload.pdf")
doc = fitz.open()
page = doc.new_page()
page.insert_text((72, 72), "hello college knowledge assistant")
doc.save(pdf_path)
doc.close()

class FileLike:
    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()

    def read(self):
        return self._data

service = DocumentService(upload_dir=r"C:\Users\harsh\College\college-knowledge-assistant\data\uploads")
try:
    result = service.index_pdf_upload(FileLike(pdf_path))
    print('OK', result)
except Exception as exc:
    import traceback
    traceback.print_exc()
    print('ERR', type(exc).__name__, exc)
