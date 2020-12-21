import logging
import csv
import pickle
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import tensorflow_hub as hub

from config import DIALO_GPT_MODEL_SIZE, CUSTOM_QA_PAIRS_PATH


logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG,
    )
logger = logging.getLogger("utils")

logger.info("\nLoading Universal Sentence Encoder...\n")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
logger.info("\nDone\n")


def save_to_pickle(obj, filename):
    with open(filename, "wb") as fin:
        pickle.dump(obj, fin)


def load_from_pickle(filename):
    with open(filename, "rb") as fin:
        obj = pickle.load(fin)
    return obj


def load_keywords_from_csv(filename):
    keywords = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            keywords = [w.strip().lower() for w in row]
    return keywords


def load_and_prepare_custom_qa_from_csv(filename):
    questions = []
    answers = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            q = row[0].strip()
            a = row[1].strip()
            questions.append(q)
            answers.append(a)

    embeddings = embed(questions).numpy()

    return embeddings, answers


logger.info("Loading custom question-answer pairs and preparing embeddings...")

embeddings, answers = load_and_prepare_custom_qa_from_csv(CUSTOM_QA_PAIRS_PATH)

logger.info("Done\n")


def find_custom_answer(user_text, threshold):
    vector = embed([user_text]).numpy()
    scores = np.dot(vector, embeddings.T)
    idx = np.argmax(scores)
    max_score = np.max(scores)
    if max_score > threshold:
        return answers[idx]
    return None


logger.info("Loading %s size DialoGPT...\n", DIALO_GPT_MODEL_SIZE)
tokenizer = AutoTokenizer.from_pretrained(f"microsoft/DialoGPT-{DIALO_GPT_MODEL_SIZE}")
model = AutoModelForCausalLM.from_pretrained(f"microsoft/DialoGPT-{DIALO_GPT_MODEL_SIZE}")
logger.info("Done")


def dialog_gpt(user_text, chat_history_ids, restart_dialogue):
    user_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, user_input_ids], dim=-1) if not restart_dialogue else user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    reply_text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply_text, chat_history_ids
