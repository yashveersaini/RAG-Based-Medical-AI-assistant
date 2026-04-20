# test.py
from pathlib import Path
from src.ingestion import ingest_file_from_path


# always resolve to absolute path from the project root
filepath = Path(__file__).parent / "data" / "Medical_book.pdf"

print(f"Looking for file at: {filepath.resolve()}")
print(f"File exists: {filepath.exists()}")

result = ingest_file_from_path(str(filepath.resolve()))
print(result)