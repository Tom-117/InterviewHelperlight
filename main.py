import os
import random
import re
import sys
from collections import Counter

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

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


def extract_technical_terms(text, keyword_extractor, top_n=5):

    try:
        ner_results = keyword_extractor(text)
        raw_keywords = [
            entity["word"].lower().strip()
            for entity in ner_results
            if entity["score"] > 0.3
        ]

        found_tech = []
        for kw in raw_keywords:
            for tech in COMMON_TECH_KEYWORDS:
                if tech in kw or kw in tech:
                    found_tech.append(tech)
                    break

        unique_tech = list(dict.fromkeys(found_tech))
        return unique_tech[:top_n]

    except Exception as e:
        print(f"NER extraction failed (falling back to regex): {e}")
        text_lower = text.lower()
        found = [
            kw
            for kw in COMMON_TECH_KEYWORDS
            if re.search(rf"\b{re.escape(kw)}\b", text_lower)
        ]
        freq = Counter(re.findall(r"\w[\w\+\#\-/\.]*", text_lower))
        return sorted(found, key=lambda k: freq.get(k, 0), reverse=True)[:top_n]


def main():
    try:
        flan_tokenizer, flan_model, embedder, keyword_extractor = load_local_models()
    except ValueError as e:
        print(f"FATAL ERROR: Expected 4 models, got mismatch: {e}")
        sys.exit(1)

    try:
        with open("cv.txt", "r", encoding="utf-8") as f:
            cv_text = f.read().strip()
            if not cv_text:
                print("ERROR: cv.txt is empty.")
                return
    except FileNotFoundError:
        print("ERROR: cv.txt not found. Create it with your CV text.")
        return

    detected_techs = extract_technical_terms(cv_text, keyword_extractor, top_n=5)
    tech_summary = ", ".join(detected_techs) if detected_techs else "programming and IT"
    print(f"Detected technical keywords: {tech_summary}\n")

    asked_questions = set()
    all_answers = []

    for i, q_type in enumerate(["technical", "soft_skills", "motivation"], 1):

        persona = "You are a sharp, creative, and insightful interviewer."

        if q_type == "technical":
            angle = random.choice(
                [
                    "ask about a complex debugging scenario they faced.",
                    "ask about a key design decision and its trade-offs.",
                    "ask about optimizing performance in a past project.",
                    "ask about scalability challenges they've encountered.",
                    f"ask about a time they had to learn a new tool from {tech_summary} quickly.",
                    "ask about their experience with code reviews and testing.",
                ]
            )
            role = (
                f"{persona} You are a senior technical interviewer focusing on {tech_summary}. "
                f"Your task: {angle}"
            )
        elif q_type == "soft_skills":
            angle = random.choice(
                [
                    "ask a behavioral question about handling difficult feedback.",
                    "ask a situational question about managing conflicting priorities.",
                    "ask about a time they had to persuade a team member to their point of view.",
                    "ask about a project failure and what they learned from it.",
                    "ask about a time they resolved a team conflict.",
                ]
            )
            role = f"{persona} You are an HR interviewer. Your task: {angle}"
        else:
            angle = random.choice(
                [
                    "ask about their long-term career vision (5+ years).",
                    "ask what kind of work environment helps them thrive.",
                    "ask about a project from their CV that they found most motivating and why.",
                    "ask how they stay up-to-date with industry trends.",
                    "ask about a time they went above and beyond their assigned role.",
                ]
            )
            role = f"{persona} You are a recruiter. Your task: {angle}"

        prompt = f"""
CV Context:
{cv_text}

Interviewer Role and Task: {role}

Rules for your generated question:
1.  Generate ONE SINGLE interview question.
2.  The question MUST be based on the Role, Task, and CV Context.
3.  DO NOT use generic, common questions (e.g., "What is your greatest weakness?", "Tell me about yourself.").
4.  The question must be unique, specific, and insightful.
5.  Start with a capital letter and end with a single question mark '?'.
6.  Do not add any preamble like "Here is your question:".

Question:
"""

        print(f"\n--- Generating {q_type.capitalize()} Question (Angle: {angle}) ---")
        question = None
        for _ in range(3):
            try:
                inputs = flan_tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=1024
                ).to(flan_model.device)

                outputs = flan_model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=True,
                    temperature=0.85,
                    top_k=50,
                    top_p=0.95,
                    num_beams=1,
                    no_repeat_ngram_size=2,
                )

                q = flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                q = re.sub(r"[\[\]\{\}]", "", q)
                if q and q.endswith("?") and len(q) > 15 and q not in asked_questions:
                    question = q
                    break
            except Exception as e:
                print(f"Generation failed: {e}")

        if not question:
            question = f"Tell me about a challenging {q_type} situation you faced."
            print("Used fallback question.")

        asked_questions.add(question)
        print(f"\nQuestion {i}/3: {question}")

        user_answer = input("\nYour answer (in English): ").strip()

        # Generate ideal English answer
        ideal_prompt = f"""
Question: {question}

Provide a strong, concise English answer using the STAR method (Situation, Task, Action, Result).
Use specific tools ({tech_summary}), metrics, and outcomes.
Be concise (1-2 paragraphs).
"""
        try:
            inputs = flan_tokenizer(
                ideal_prompt, return_tensors="pt", truncation=True, max_length=1024
            ).to(flan_model.device)
            # Use simple generation for the ideal answer, no need for complex sampling
            outputs = flan_model.generate(**inputs, max_new_tokens=200, num_beams=2)
            ideal_english = flan_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

        except Exception as e:
            ideal_english = "(Ideal answer generation failed)"

        # Score similarity
        score = 0.0
        if user_answer and ideal_english and "failed" not in ideal_english:
            try:
                embeddings = embedder.encode(
                    [user_answer, ideal_english], convert_to_tensor=True
                )
                score = util.cos_sim(embeddings[0], embeddings[1]).item()
            except Exception as e:
                print(f"Embedding/scoring failed: {e}")
                score = 0.0

        all_answers.append(
            {
                "question": question,
                "user_answer": user_answer,
                "ideal_answer": ideal_english,
                "score": score,
            }
        )

        # Feedback
        print("\n=== Feedback ===")
        print(f"Ideal: {ideal_english}")
        print(f"Score: {score:.2%}")
        if score > 0.8:
            print("Outstanding!")
        elif score > 0.6:
            print("Solid, but add more detail.")
        else:
            print("Needs more structure and specifics.")

        if i < 3:
            input("\nPress Enter for next question...")

    # Final Summary
    print("\n" + "=" * 50)
    print("FINAL INTERVIEW REPORT")
    print("=" * 50)
    avg_score = (
        (sum(a["score"] for a in all_answers) / len(all_answers))
        if all_answers
        else 0.0
    )
    for i, a in enumerate(all_answers, 1):
        print(f"Q{i} [{a['score']:.1%}]: {a['question'][:60]}...")
    print(f"\nOverall Score: {avg_score:.2%}")
    if avg_score > 0.75:
        print("Strong candidate!")
    elif avg_score > 0.5:
        print("Good potential with practice.")
    else:
        print("Recommend more preparation.")


if __name__ == "__main__":
    main()
