# InterviewHelperLight

**Generate personalized technical, behavioral, and motivation interview questions from your CV — with answer evaluation.**

---

## Overview

**InterviewHelperLight** is a **CV-based Interview Preparation Assistant** that helps you:

- **Upload your CV** (PDF, DOCX, TXT)  
- **Extract technical skills** using NLP + embedding similarity  
- **Generate targeted interview questions** (technical, behavioral, motivation)  
- **Evaluate your answers** with STAR method scoring, technical depth, and semantic similarity  

---

### Using Local AI Models

| Model | Purpose |
|------|---------|
| **FLAN-T5 Large** | Question generation & scoring |
| **all-MiniLM-L6-v2** | Semantic similarity & answer evaluation |
| **RoBERTa QA**  | Skill extraction from CV |

---

## Features

| Feature | Description |
|--------|-------------|
| **Smart Skill Extraction** | Detects real skills in context using QA + embeddings |
| **3 Question Types** | Technical , Behavioral (STAR) , Motivation |
| **Answer Evaluation** | STAR structure, technical depth, completeness |
| **Dual Interface** | Web UI (Flask) + CLI |
| **Docker Ready** | Run anywhere with zero setup |

---

## Quick Start

### Option 1: Docker

bash
git clone https://github.com/Tom-117/InterviewHelperlight.git
cd InterviewHelperLight

chmod +x run.sh
./run.sh


### Option 2: Local

# Clone and enter directory
git clone https://github.com/yourusername/InterviewHelperLight.git
cd InterviewHelperLight

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
python app.py for webui or main.py for cli version