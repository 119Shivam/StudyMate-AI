
import os
import re

from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# 1. GEMINI CONFIGURATION
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"


# --------------------------------------------------
# 2. BUILD PDF CONTEXT
# --------------------------------------------------

def build_pdf_context(retrieved_chunks):

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        source_text = f"""
SOURCE {i}

Document: {chunk['document_name']}
Page: {chunk['page_number']}

Extracted text:
{chunk['text']}
"""

        context_parts.append(source_text)

    return "\n".join(context_parts)


# --------------------------------------------------
# 3. GENERATE SUGGESTED QUESTIONS
# --------------------------------------------------

def generate_suggested_questions(document_chunks, count=4):

    if not document_chunks:
        return []

    sample_count = min(12, len(document_chunks))

    if sample_count == 1:
        selected_chunks = document_chunks

    else:
        selected_indices = [
            round(
                i * (len(document_chunks) - 1) / (sample_count - 1)
            )
            for i in range(sample_count)
        ]

        selected_chunks = [
            document_chunks[i]
            for i in selected_indices
        ]

    context = build_pdf_context(selected_chunks)

    prompt = f"""
You are StudyMate AI, an academic study assistant.

Generate exactly {count} useful questions based ONLY on
the supplied PDF content.

PDF CONTENT:
{context}

INSTRUCTIONS:

1. Generate questions that can actually be answered
   using the supplied content.
2. Cover different concepts wherever possible.
3. Prefer important definitions, explanations,
   characteristics, processes and comparisons.
4. Use simple student-friendly English.
5. Do not invent topics or facts.
6. Avoid duplicate or very similar questions.
7. Keep each question reasonably short.
8. Return only the questions.
9. Put each question on a separate line.
10. Do not include numbering, bullets, headings
    or explanations.

Generate the questions now.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response or not response.text:
        return []

    questions = []

    for line in response.text.strip().splitlines():

        question = line.strip()

        question = re.sub(
            r"^\s*(?:\d+[.)]\s*|[-*•]\s*)",
            "",
            question
        )

        question = question.strip().strip('"').strip("'")

        if (
            question
            and len(question.split()) >= 3
            and question not in questions
        ):
            questions.append(question)

        if len(questions) >= count:
            break

    return questions


# --------------------------------------------------
# 4. BUILD CONVERSATION HISTORY
# --------------------------------------------------

def build_history(chat_history):

    history_parts = []

    if chat_history:

        for message in chat_history:

            role = message.get(
                "role",
                "unknown"
            ).upper()

            content = message.get(
                "content",
                ""
            )

            if content:
                history_parts.append(
                    f"{role}: {content}"
                )

    if not history_parts:
        return "No previous conversation."

    return "\n".join(history_parts)


# --------------------------------------------------
# 5. GENERATE INITIAL ANSWER
# --------------------------------------------------

def generate_draft(
    user_query,
    context,
    history
):

    prompt = f"""
You are StudyMate AI, an academic PDF assistant.

Answer the student's question using only the retrieved
PDF passages as factual evidence.

RECENT CONVERSATION:
{history}

CURRENT QUESTION:
{user_query}

RETRIEVED PDF CONTEXT:
{context}

INSTRUCTIONS:

1. Answer the current question directly.
2. Use simple, student-friendly English.
3. Do not invent facts or examples.
4. Keep definitions and characteristics separate.
5. Do not mix characteristics of related concepts.
6. Preserve the meaning and terminology of the PDF.
7. If information is missing, say so.
8. If OCR text is unclear, mention the uncertainty.
9. Cite factual statements using document name and page.
10. Do not use conversation history as factual evidence.
11. Do not treat similarity scores as confidence scores.
12. For comparisons, use a table when appropriate.
13. For characteristics, use numbered points.
14. Avoid unnecessary information.

IMPORTANT:

If the PDF discusses a related concept separately,
do not automatically treat its characteristics as
characteristics of the current concept.

Now prepare the answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini returned an empty draft answer."
        )

    return response.text.strip()


# --------------------------------------------------
# 6. VERIFY DRAFT ANSWER AGAINST PDF SOURCES
# --------------------------------------------------

def verify_answer(
    user_query,
    draft_answer,
    context
):

    verification_prompt = f"""
You are the source verification system of StudyMate AI.

Your job is to check whether the draft answer is supported
by the retrieved PDF passages.

STUDENT QUESTION:
{user_query}

RETRIEVED PDF PASSAGES:
{context}

DRAFT ANSWER:
{draft_answer}

VERIFICATION RULES:

1. Check every factual statement in the draft.

2. Verify that each statement is actually supported
   by the retrieved PDF passages.

3. Check whether each cited document name and page
   number matches the supplied passages.

4. Remove or rewrite unsupported factual claims.

5. Never invent replacement facts.

6. If a statement is only partially supported,
   rewrite it to match exactly what the PDF establishes.

7. Pay special attention to definitions,
   characteristics, conditions and examples.

8. Do not combine two related concepts simply because
   they appear in the same PDF.

9. If the PDF discusses a condition separately from
   a definition, preserve that distinction.

10. If two passages appear inconsistent, explain the
    inconsistency rather than silently resolving it.

11. If OCR text is unclear, retain an uncertainty note
    rather than guessing what the original text said.

12. Do not introduce outside knowledge.

13. Do not remove relevant supported information
    merely because it is difficult to explain.

OUTPUT INSTRUCTIONS:

Return the corrected final answer only.

Keep the original question's required level of detail.

Use simple academic English.

Preserve appropriate headings, numbered lists and tables.

Keep document and page citations beside the claims
they support.

If the sources do not provide enough information,
clearly mention that limitation.

Do not include your internal verification process.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=verification_prompt
    )

    if not response or not response.text:
        raise RuntimeError(
            "Gemini returned an empty verification response."
        )

    return response.text.strip()


# --------------------------------------------------
# 7. MAIN ANSWER GENERATION FUNCTION
# --------------------------------------------------

def generate_answer(
    user_query,
    retrieved_chunks,
    chat_history=None
):

    """
    Two-pass RAG answer generation.

    Pass 1: Generate an answer using retrieved PDF context.
    Pass 2: Verify the answer against the same PDF context.
    """

    if not retrieved_chunks:
        return (
            "I could not find relevant information in the "
            "uploaded PDFs to answer this question."
        )

    try:

        context = build_pdf_context(
            retrieved_chunks
        )

        history = build_history(
            chat_history
        )

        draft_answer = generate_draft(
            user_query,
            context,
            history
        )

        final_answer = verify_answer(
            user_query,
            draft_answer,
            context
        )

        return final_answer

    except Exception as error:

        raise RuntimeError(
            f"StudyMate answer generation failed: {error}"
        ) from error
