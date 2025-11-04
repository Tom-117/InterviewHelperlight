import os
import random
import re
import sys
from collections import Counter

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)


def load_hf_model(model_id, local_path, model_class, tokenizer_class, torch_dtype):
    try:
        if not os.path.exists(local_path):
            print(f"Downloading {model_id} ...")
            tokenizer = tokenizer_class.from_pretrained(model_id)
            model = model_class.from_pretrained(
                model_id, dtype=torch_dtype, device_map="auto"
            )
            tokenizer.save_pretrained(local_path)
            model.save_pretrained(local_path)
        else:
            print(f"Loading {model_id} from {local_path} ...")
            tokenizer = tokenizer_class.from_pretrained(local_path)
            model = model_class.from_pretrained(
                local_path, dtype=torch_dtype, device_map="auto"
            )
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def load_local_models():
    try:
        # Clear GPU memory and print available memory
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(
                f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB"
            )
            print(
                f"Available GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB used"
            )
        else:
            print("No GPU available, using CPU")

        # Determine dtype based on hardware
        torch_dtype = (
            torch.float16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float32
        )
        print(f"Using dtype: {torch_dtype}")

        print("Starting to load FLAN-T5 Large model...")
        try:
            # 1. FLAN-T5 Large
            flan_tokenizer, flan_model = load_hf_model(
                "google/flan-t5-large",
                "./models/flan-t5-large",
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
                torch_dtype,
            )
            print("FLAN-T5 Large loaded successfully")
        except Exception as e:
            print(f"Failed to load FLAN-T5 Large: {str(e)}")
            print("If you're seeing an out of memory error, try:")
            print("1. Close other applications")
            print("2. Ensure you have enough disk space")
            print("3. If using GPU, ensure you have enough VRAM")
            raise

        # 2. Sentence Embedder
        embedder_path = "./models/all-MiniLM-L6-v2"
        if not os.path.exists(embedder_path):
            print("Downloading Sentence Transformer...")
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embedder.save(embedder_path)
        else:
            print("Loading Sentence Transformer...")
            embedder = SentenceTransformer(embedder_path)

        # 3. Text Analysis Model
        kw_path = "./models/roberta-qa"
        kw_tokenizer, kw_model = load_hf_model(
            "deepset/roberta-base-squad2",
            kw_path,
            AutoModelForQuestionAnswering,
            AutoTokenizer,
            torch_dtype,
        )

        # Initialize the pipeline for text analysis
        keyword_extractor = pipeline(
            "question-answering",
            model=kw_model,
            tokenizer=kw_tokenizer,
            handle_long_generation=True,
        )

        print("All models loaded successfully!\n")
        return (flan_tokenizer, flan_model, embedder, keyword_extractor)

    except Exception as e:
        print(f"Model loading failed: {e}")
        sys.exit(1)
