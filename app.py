import os
import re
from collections import Counter

import PyPDF2
import torch
from docx import Document
from flask import Flask, redirect, render_template, request, url_for
from sentence_transformers import util
from werkzeug.utils import secure_filename

from model_loader import load_hf_model, load_local_models

COMMON_TECH_KEYWORDS = [
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "go",
    "rust",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "tensorflow",
    "pytorch",
    "keras",
    "react",
    "angular",
    "vue",
    "django",
    "flask",
    "spring",
    "node",
    "express",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "sql",
    "postgres",
    "mysql",
    "mongodb",
    "redis",
    "spark",
    "hadoop",
    "nlp",
    "computer vision",
    "microservices",
    "rest",
    "graphql",
    "ci/cd",
    "terraform",
    "ansible",
    "linux",
]


def read_cv_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext == ".pdf":
            with open(filepath, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:

                    page_text = page.extract_text() or ""
                    # Clean up common PDF extraction issues
                    page_text = re.sub(r"\s+", " ", page_text)
                    page_text = re.sub(r"[^\x20-\x7E\n]", "", page_text)
                    text.append(page_text.strip())

                extracted_text = "\n".join(text).strip()
                if not extracted_text:
                    raise ValueError("No text could be extracted from the PDF")
                return extracted_text

        elif ext == ".docx":
            doc = Document(filepath)
            # Extract both paragraphs and tables
            text_parts = []

            # Get text from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())

            # Get text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        text_parts.append(row_text)

            return "\n".join(text_parts)

        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()
                if not text.strip():
                    raise ValueError("The text file is empty")
                return text

        else:
            raise ValueError(f"Unsupported file format: {ext}")

    except Exception as e:
        raise Exception(f"Failed to read file: {str(e)}")


def extract_technical_terms(text, qa_model, embedder, top_n=5):

    def clean_text(text):

        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    try:

        questions = [
            "List all programming languages mentioned in the text.",
            "What specific technical skills or technologies are mentioned in the text?",
            "What software applications or development tools are listed in the text?",
            "What frameworks, libraries, or packages are mentioned in the text?",
            "What databases, storage systems, or data technologies are discussed in the text?",
        ]

        chunk_size = 512
        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        all_skills = set()

        for chunk in chunks:
            clean_chunk = clean_text(chunk)

            for question in questions:
                try:

                    result = qa_model(
                        question=question,
                        context=clean_chunk,
                        handle_impossible_answer=True,
                        max_answer_len=100,
                    )

                    if result and result.get("answer") and result.get("score", 0) > 0.1:
                        # Split the answer into individual terms
                        extracted_terms = re.split(
                            r"[,;\n]|\band\b|\bor\b|\bas well as\b|\bsuch as\b|\bincluding\b",
                            result["answer"].lower(),
                        )

                        # Clean and add terms
                        for term in extracted_terms:
                            term = term.strip("., ()[]{}")
                            if term and len(term) > 2:  # Avoid very short terms
                                # Remove common filler words
                                term = re.sub(
                                    r"\b(the|a|an|in|on|at|with|using|for|to|of)\b",
                                    "",
                                    term,
                                )
                                term = re.sub(r"\s+", " ", term).strip()
                                if term and len(term) > 2:
                                    all_skills.add(term)

                except Exception as e:
                    print(f"Extraction failed for chunk with question {question}: {e}")
                    continue

        text_lower = text.lower()
        for tech in COMMON_TECH_KEYWORDS:
            if re.search(rf"\b{re.escape(tech)}\b", text_lower):
                all_skills.add(tech.lower())

        unique_skills = list(dict.fromkeys(all_skills))

        if embedder and unique_skills:
            cv_embedding = embedder.encode(text_lower, convert_to_tensor=True)
            verified_skills = []

            for skill in unique_skills:

                skill_pattern = re.escape(skill)
                context_matches = re.finditer(skill_pattern, text_lower)
                contexts = []

                for match in context_matches:
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    contexts.append(text[start:end])

                if contexts:

                    context_embeddings = embedder.encode(
                        contexts, convert_to_tensor=True
                    )
                    similarities = util.cos_sim(cv_embedding, context_embeddings)
                    max_sim = float(similarities.max())

                    if max_sim > 0.4:
                        verified_skills.append((skill, max_sim))
                else:

                    skill_embedding = embedder.encode(skill, convert_to_tensor=True)
                    similarity = float(util.cos_sim(cv_embedding, skill_embedding))
                    if similarity > 0.4:
                        verified_skills.append((skill, similarity))

            verified_skills.sort(key=lambda x: x[1], reverse=True)
            return [skill for skill, _ in verified_skills[:top_n]]

        return unique_skills[:top_n]

    except Exception as e:
        print(f"Skill extraction failed: {e}")
        # Fallback to basic keyword matching
        text_lower = text.lower()
        matches = [
            tech
            for tech in COMMON_TECH_KEYWORDS
            if re.search(rf"\b{re.escape(tech)}\b", text_lower)
        ]
        return matches[:top_n]


