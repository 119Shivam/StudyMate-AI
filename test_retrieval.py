
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. LOAD DOCUMENT
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "document_storage"

target_folder = None

for folder in STORAGE_DIR.glob("chat_*"):
    metadata_path = folder / "metadata.json"

    if not metadata_path.exists():
        continue

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    filenames = metadata.get("processed_files", [])

    if any("THERMODYNAMICS" in name.upper() for name in filenames):
        target_folder = folder
        break

if target_folder is None:
    raise FileNotFoundError("Thermodynamics document not found.")

with open(target_folder / "chunks.json", "r", encoding="utf-8") as file:
    document_chunks = json.load(file)

index = faiss.read_index(str(target_folder / "index.faiss"))

print("Document loaded:", target_folder.name)
print("Total chunks:", len(document_chunks))


# --------------------------------------------------
# 2. LOAD MODEL
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 3. MMR FUNCTION
# --------------------------------------------------

def apply_mmr(candidates, top_k=5, diversity=0.75):

    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:

        best_candidate = None
        best_score = float("-inf")

        for candidate in remaining:

            relevance = candidate["similarity"]

            if not selected:
                redundancy = 0
            else:
                redundancy = max(
                    float(np.dot(
                        candidate["vector"],
                        chosen["vector"]
                    ))
                    for chosen in selected
                )

            mmr_score = (
                diversity * relevance
                - (1 - diversity) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


# --------------------------------------------------
# 4. QUESTIONS
# --------------------------------------------------

questions = [
    "Explain the concept of a quasi-static process.",

    (
        "Previous topic/question: Explain the concept of a quasi-static process.\n"
        "Current follow-up question: Explain its characteristics in simple language "
        "with an example."
    ),

    (
        "What happens to the work done by a system and the work done on the system "
        "as the number of steps in a finite-step process increases?"
    )
]


# --------------------------------------------------
# 5. THRESHOLD COMPARISON
# --------------------------------------------------

thresholds = [0.30, 0.33, 0.35]

for question_number, question in enumerate(questions, start=1):

    print("\n" + "=" * 70)
    print(f"QUESTION {question_number}")
    print("=" * 70)

    print(question)

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        index.ntotal
    )

    candidates = []

    for score, chunk_index in zip(scores[0], indices[0]):

        if chunk_index < 0:
            continue

        vector = index.reconstruct(int(chunk_index))

        candidates.append({
            "index": int(chunk_index),
            "similarity": float(score),
            "vector": vector,
            "chunk": document_chunks[int(chunk_index)]
        })

    for threshold in thresholds:

        filtered = [
            candidate
            for candidate in candidates
            if candidate["similarity"] >= threshold
        ]

        selected = apply_mmr(filtered)

        print("\n" + "-" * 60)
        print(f"THRESHOLD: {threshold}")
        print("Chunks remaining:", len(filtered))
        print("Chunks selected by MMR:", len(selected))

        for rank, candidate in enumerate(selected, start=1):

            chunk = candidate["chunk"]

            print(
                f"\nRank: {rank} | "
                f"Similarity: {candidate['similarity']:.4f} | "
                f"Page: {chunk.get('page_number', 'Unknown')}"
            )

            print(chunk.get("text", "")[:250])


print("\nTHRESHOLD COMPARISON COMPLETED!")
