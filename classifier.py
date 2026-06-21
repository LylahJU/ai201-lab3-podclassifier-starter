import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().
    """
    task_instruction = (
        "You are classifying podcast episodes by their format. "
        "Classify the episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often debating "
        "or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, archival "
        "audio, reporting — with a clear narrative arc\n\n"
        "Return only the label and your reasoning. Do not explain the taxonomy."
    )

    sections = [task_instruction]

    if not labeled_examples:
        sections.append("This episode cannot be labeled.")
    else:
        for ex in labeled_examples:
            sections.append(
                f"Title: {ex['title']}\nDescription: {ex['description']}\nLabel: {ex['label']}"
            )

    sections.append(
        f"Description: {description}\nLabel: ?\n\n"
        "Classify the episode above. Return your answer in the format below:\n"
        "Label: <label>\n"
        "<1-2 sentences of reasoning>"
    )

    return "\n\n---\n\n".join(sections)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.

    Steps:
      1. Call build_few_shot_prompt() to construct the prompt
      2. Send it to the LLM via _client.chat.completions.create()
      3. Parse the response to extract a label and reasoning
      4. Validate the label — if it's not in VALID_LABELS, set it to "unknown"
      5. Return a dict with "label" and "reasoning" keys

    Handle the case where the LLM returns something unparseable gracefully —
    don't let a bad response crash the whole evaluation.
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )

        response_text = response.choices[0].message.content.strip()
        first_line, _, reasoning_text = response_text.partition("\n")

        if ":" not in first_line:
            return {"label": "unknown", "reasoning": response_text}

        extracted_label = first_line.split(":", 1)[1].strip().lower()

        if not extracted_label:
            return {"label": "unknown", "reasoning": response_text}

        label = extracted_label if extracted_label in VALID_LABELS else "unknown"
        return {"label": label, "reasoning": reasoning_text.strip()}

    except Exception as e:
        return {
            "label": "unknown",
            "reasoning": f"Error during classification: {e}",
        }
