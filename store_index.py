# from dotenv import load_dotenv
# from pathlib import Path
# from pinecone import Pinecone, ServerlessSpec
# from src.ingestion import ingest_file_from_path, SUPPORTED_EXTENSIONS
# from config import settings

# load_dotenv()

# # Create index if it doesn't exist
# pc = Pinecone(api_key=settings.pinecone_api_key)

# if not pc.has_index(settings.pinecone_index_name):
#     pc.create_index(
#         name=settings.pinecone_index_name,
#         dimension=384,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
#     print(f"Created index: {settings.pinecone_index_name}")


# data_dir = Path(__file__).parent / "data"
# all_files = [
#     f for f in data_dir.iterdir()
#     if f.suffix.lower() in SUPPORTED_EXTENSIONS
# ]

# if not all_files:
#     print("No supported files found in /data — add PDF, DOCX, or TXT files")
#     exit(0)

# print(f"Found {len(all_files)} files to ingest: {[f.name for f in all_files]}")

# # Ingest each file and print a summary
# results = []
# for filepath in all_files:
#     print(f"\nIngesting: {filepath.name}")
#     result = ingest_file_from_path(str(filepath))
#     results.append(result)

# # Print ingestion summary table
# print("\n" + "="*55)
# print(f"{'File':<25} {'Type':<6} {'Chunks':>7} {'Status'}")
# print("-"*55)
# for r in results:
#     print(f"{r.filename:<25} {r.file_type:<6} {r.chunks_indexed:>7}  {r.status}")
#     if r.error:
#         print(f"  Error: {r.error}")
# print("="*55)
# print(f"Total chunks indexed: {sum(r.chunks_indexed for r in results)}")

from pathlib import Path
from src.ingestion import ingest_file_from_path


# always resolve to absolute path from the project root
filepath = Path(__file__).parent / "data" / "Medical_book.pdf"

print(f"Looking for file at: {filepath.resolve()}")
print(f"File exists: {filepath.exists()}")

result = ingest_file_from_path(str(filepath.resolve()))
print(result)