import os
import sys
import torch
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util



def load_local_models():
    try:
        torch.cuda.empty_cache()
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # FLAN-T5 model
        flan_path = "./models/flan-t5-base"
        if not os.path.exists(flan_path):
            print("Downloading FLAN-T5-Base model...")
            flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            flan_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base",
                dtype=torch_dtype,
                device_map="auto"
            )
            flan_tokenizer.save_pretrained(flan_path)
            flan_model.save_pretrained(flan_path)
        else:
            flan_tokenizer = AutoTokenizer.from_pretrained(flan_path)
            flan_model = AutoModelForSeq2SeqLM.from_pretrained(
                flan_path,
                dtype=torch_dtype,
                device_map="auto"
            )

        # Translation model (English → Hungarian)
        translation_path = "./models/opus-mt-en-hu"
        if not os.path.exists(translation_path):
            print("Downloading English-Hungarian translation model...")
            gpt_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hu")
            gpt_model = AutoModelForSeq2SeqLM.from_pretrained(
                "Helsinki-NLP/opus-mt-en-hu",
                dtype=torch_dtype,
                device_map="auto"
            )
            gpt_tokenizer.save_pretrained(translation_path)
            gpt_model.save_pretrained(translation_path)
        else:
            gpt_tokenizer = AutoTokenizer.from_pretrained(translation_path)
            gpt_model = AutoModelForSeq2SeqLM.from_pretrained(
                translation_path,
                dtype=torch_dtype,
                device_map="auto"
            )

        # Sentence Transformer for similarity scoring
        embedder_path = "./models/all-MiniLM-L6-v2"
        if not os.path.exists(embedder_path):
            print("Downloading Sentence Transformer model...")
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embedder.save(embedder_path)
        else:
            embedder = SentenceTransformer(embedder_path)

        print("All models loaded successfully.\n")
        return flan_tokenizer, flan_model, gpt_tokenizer, gpt_model, embedder

    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)



COMMON_TECH_KEYWORDS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust", "ruby", "php", "swift", "kotlin",
    "tensorflow", "pytorch", "keras", "react", "angular", "vue", "django", "flask", "spring", "node", "express",
    "docker", "kubernetes", "aws", "azure", "gcp", "sql", "postgres", "mysql", "mongodb", "redis", "spark", "hadoop",
    "nlp", "computer vision", "microservices", "rest", "graphql", "ci/cd", "terraform", "ansible", "linux"
]

def extract_technical_terms(text, top_n=5):
    text = text.lower()
    found = []
    for kw in COMMON_TECH_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            found.append(kw)
    freq = Counter(re.findall(r"\w[\w\+\#\-/\.]*", text))
    found_sorted = sorted(found, key=lambda k: freq.get(k, 0), reverse=True)
    return found_sorted[:top_n]



def main():
    flan_tokenizer, flan_model, gpt_tokenizer, gpt_model, embedder = load_local_models()

    # Read CV text
    try:
        with open("cv.txt", "r", encoding="utf-8") as f:
            cv_text = f.read()
    except FileNotFoundError:
        print("CV file not found (cv.txt). Please add a CV text file in the same directory.")
        return

    # Detect key technologies from CV
    detected_techs = extract_technical_terms(cv_text, top_n=5)
    tech_summary = ", ".join(detected_techs) if detected_techs else "various programming and IT skills"
    print(f"Detected technical keywords: {tech_summary}\n")

    # --- SMART QUESTION GENERATION LOOP ---
    asked_questions = set()
    all_answers = []

    for i, q_type in enumerate(["technical", "soft_skills", "motivation"], 1):
        if q_type == "technical":
            interviewer_role = (
                f"You are a senior technical interviewer. The candidate's main technologies include {tech_summary}. "
                "Your task: generate ONE realistic technical interview question that tests practical knowledge. "
                "Avoid personal or motivational questions. Ask about real challenges, debugging, optimization, or system design."
            )
        elif q_type == "soft_skills":
            interviewer_role = (
                "You are an HR interviewer. Generate ONE behavioral interview question about teamwork, leadership, or communication. "
                "Avoid generic phrasing — base it on realistic situations."
            )
        else:
            interviewer_role = (
                "You are a senior recruiter. Generate ONE question about the candidate’s career motivation or future goals, "
                "in a professional and focused tone."
            )

        prompt = f"""
You are interviewing a candidate. The following is their CV text:

{cv_text}

{interviewer_role}

Good question examples:
- "Can you describe a time when you optimized a {tech_summary.split(',')[0]} project?"
- "How have you applied {tech_summary.split(',')[0]} in a real-world scenario?"
Bad question examples:
- "What are your goals?"
- "Tell me about yourself."

Now, generate ONE clear question appropriate for an actual interview.
It must end with a '?' and contain NO placeholders, brackets, or vague wording.
"""

        candidate_questions = []
        for attempt in range(3):
            inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(flan_model.device)
            outputs = flan_model.generate(
                **inputs,
                max_new_tokens=80,
                num_beams=4,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                temperature=0.7 + attempt * 0.1,
                eos_token_id=flan_tokenizer.eos_token_id
            )
            q = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
            q = re.sub(r"[\[\]\{\}]", "", q).strip()
            if "?" in q and len(q) > 10:
                candidate_questions.append(q)

        # Pick most relevant question using similarity to CV text
        if candidate_questions:
            cv_emb = embedder.encode(cv_text, convert_to_tensor=True)
            q_embs = embedder.encode(candidate_questions, convert_to_tensor=True)
            sims = util.cos_sim(q_embs, cv_emb).squeeze(1)
            best_idx = torch.argmax(sims).item()
            question = candidate_questions[best_idx]
        else:
            question = "Can you describe a recent technical challenge you faced and how you solved it?"

        if question in asked_questions:
            continue
        asked_questions.add(question)

        print(f"\n--- Question {i}/3 ({q_type.capitalize()}) ---")
        print(question)
        user_answer = input("\nYour answer (in Hungarian): ").strip()

        # Generate professional answer in English, then translate to Hungarian
        ideal_prompt = (
            f"Provide a professional, specific English answer to this interview question (1-2 short paragraphs):\n"
            f"{question}\n\n"
            "The answer should mention specific actions, results, and examples from real projects."
        )

        try:
            inputs = flan_tokenizer(ideal_prompt, return_tensors="pt", truncation=True, max_length=1024).to(flan_model.device)
            outputs = flan_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.6,
                top_p=0.9,
                do_sample=True,
                num_beams=4
            )
            ideal_english = flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            translate_prompt = f"Translate this English answer into Hungarian, keeping it natural and professional:\n\n{ideal_english}"
            t_inputs = gpt_tokenizer(translate_prompt, return_tensors="pt", truncation=True, max_length=1024).to(gpt_model.device)
            t_outputs = gpt_model.generate(**t_inputs, max_new_tokens=300, num_beams=4)
            ideal_answer = gpt_tokenizer.decode(t_outputs[0], skip_special_tokens=True).strip()

        except Exception as e:
            print(f"Error generating ideal answer: {e}")
            ideal_answer = "(Failed to generate ideal answer.)"

        # Similarity scoring
        embeddings = embedder.encode([user_answer, ideal_answer], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

        all_answers.append({
            "question": question,
            "user_answer": user_answer,
            "ideal_answer": ideal_answer,
            "score": similarity
        })

        print("\n=== Immediate Feedback ===")
        print(f"Question: {question}")
        print(f"Your answer: {user_answer}")
        print(f"Ideal answer: {ideal_answer}")
        print(f"Similarity score: {similarity:.2%}")
        if similarity > 0.8:
            print("Excellent answer!")
        elif similarity > 0.6:
            print("Good answer, but could use more detail.")
        else:
            print("Try providing more concrete examples next time.")

        if i < 3:
            input("\nPress Enter to continue to the next question...")

    # Final evaluation
    print("\n=== Final Interview Evaluation ===")
    if not all_answers:
        print("No answers were generated.")
        return

    final_score = sum(a["score"] for a in all_answers) / len(all_answers)
    for idx, ans in enumerate(all_answers, 1):
        print(f"\nQ{idx}: {ans['question']}")
        print(f"Score: {ans['score']:.2%}")
    print(f"\nOverall Interview Performance: {final_score:.2%}")


if __name__ == "__main__":
    main()
