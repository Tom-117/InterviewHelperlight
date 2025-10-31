import os
import sys
import torch
import re
import random
from collections import Counter
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    pipeline,
)
from sentence_transformers import SentenceTransformer, util


def load_hf_model(model_id, local_path, model_class, tokenizer_class, torch_dtype):

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


def load_local_models():
    try:
        torch.cuda.empty_cache()
        torch_dtype = (
            torch.float16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float32
        )

        # 1. FLAN-T5 Large
        flan_tokenizer, flan_model = load_hf_model(
            "google/flan-t5-large",
            "./models/flan-t5-large",
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            torch_dtype,
        )

        # 2. Sentence Embedder
        embedder_path = "./models/all-MiniLM-L6-v2"
        if not os.path.exists(embedder_path):
            print("Downloading Sentence Transformer...")
            embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embedder.save(embedder_path)
        else:
            print("Loading Sentence Transformer...")
            embedder = SentenceTransformer(embedder_path)

        # 3. Keyword Extraction Model
        kw_path = "./models/valurank-MiniLM-L6-Keyword-Extraction"
        kw_tokenizer, kw_model = load_hf_model(
            "valurank/MiniLM-L6-Keyword-Extraction",
            kw_path,
            AutoModelForTokenClassification,
            AutoTokenizer,
            torch_dtype,
        )

        keyword_extractor = pipeline(
            "token-classification",
            model=kw_model,
            tokenizer=kw_tokenizer,
            aggregation_strategy="simple",
        )

        print("All models loaded successfully!\n")
        return (flan_tokenizer, flan_model, embedder, keyword_extractor)

    except Exception as e:
        print(f"Model loading failed: {e}")
        sys.exit(1)
