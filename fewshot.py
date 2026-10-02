import json
import random
from pathlib import Path
from typing import Any

DATA_DIR= Path(__file__).parent/"data"
print(DATA_DIR)


def generat_qa_pairs(n:int=30, seed: int=42)->list[dict[str,Any]]:
    random.seed(seed)
    topics = [
        ("RAG", "Retrieval-Augmented Generation combines search with LLMs."),
        ("DSPy", "DSPy is a framework for programming language models."),
        ("Prompt Engineering", "Prompt engineering is the art of writing effective prompts."),
        ("Fine-tuning", "Fine-tuning adapts a pre-trained model to a specific task."),
        ("Vector DB", "Vector databases store embeddings for similarity search."),
        ("Chain-of-Thought", "CoT prompting asks the model to show its reasoning."),
        ("Attention", "Attention mechanisms let models focus on relevant tokens."),
        ("Transformers", "Transformers are neural networks based on self-attention."),
    ]
    qa_pairs =[]
    for i in range(n):
        topic, context = random.choice(topics)
        qa_pairs.append({
            "id":i,
            "question" : f"what is {topic}?",
            "context": context,
            "expected": context,
            "category": topic.upper().replace(" ", "_"),
            "difficulty": random.choice(['easy', "medium", "hard"])
        })

    return qa_pairs

qa_data=generat_qa_pairs(40)
# [print(ques) for ques in qa_data]

