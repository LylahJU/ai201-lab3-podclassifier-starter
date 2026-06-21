# Classifier Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 2.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `build_few_shot_prompt()` and
`classify_episode()` in `classifier.py`.

---

## build_few_shot_prompt(labeled_examples, description)

### What it does
Constructs a prompt string for the LLM that includes the task instructions,
all labeled training examples, and the new episode description to classify.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `labeled_examples` | `list[dict]` | Each dict has `"title"`, `"description"`, `"label"` (and others). These are the examples you labeled in Milestone 1. |
| `description` | `str` | The episode description to classify. |

### Output

| Return value | Type | Description |
|---|---|---|
| prompt | `str` | A complete prompt string ready to send to the LLM. |

---

### Spec fields — fill these in before writing code

**Task instruction (what should the LLM know about the task?):**

```
You are classifying podcast episodes by their format. Classify the episode
into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests,
  no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or
  discussing a topic together
- narrative: a story assembled from external sources — interviews, archival
  audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy.
```

---

**How should labeled examples be formatted in the prompt?**

```
Each example should include the episode title, a brief excerpt or the full
description, and the correct label. Separate examples with a blank line or
a delimiter like "---". Include all fields that help the model see why the
label was applied — title and description are both useful; other fields
(like episode ID) are not needed.
```

---

**Example block sketch (write one concrete example):**

```
Title: {title}
Description: {description}
Label: {label}
```

---

**How should the new episode (to be classified) be presented?**

```
Present it in the same format as the labeled examples, but omit the Label
line and replace it with an instruction to classify. For example:

Title: {title}
Description: {description}
Label: ?

Then add a line like: "Classify the episode above. Return your answer in
the format below:" followed by the output format you chose.
```

---

**What output format should you request from the LLM?**

```
[blank — you need to parse the response in classify_episode(). What format
makes parsing reliable? Think about: a single label on its own line?
A structured format like "Label: X / Reasoning: Y"? JSON?
What are the tradeoffs?]
```
`<Label>: <output>`
1-2 sentences on reasoning, why that label was chosen.

---

**Edge cases to handle in the prompt:**

```
[blank — what if labeled_examples is empty? What if the description is very
short? How does your prompt handle these?]
```
If it's empty, give a message "This episode cannot be labeled."
If it's too short, add to the reasoning and explain that. 

If there is a collision of two or more labels, choose on and then explain why that on is chosen over the others. Also include what the other options were. 

---

## classify_episode(description, labeled_examples)

### What it does
Classifies a single podcast episode description using the few-shot LLM classifier.
Returns a dict with a label and reasoning.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | The episode description to classify. |
| `labeled_examples` | `list[dict]` | Labeled training examples from `load_labeled_examples()`. |

### Output

| Return value | Type | Description |
|---|---|---|
| result | `dict` | Must have keys `"label"` and `"reasoning"`. `"label"` must be one of `VALID_LABELS` or `"unknown"`. |

---

### Spec fields — fill these in before writing code

**Step 1 — Build the prompt:**

```
Call build_few_shot_prompt(labeled_examples, description) and store the
returned string in a variable (e.g., prompt). Pass through both arguments
exactly as received — no modification needed before calling.
```

---

**Step 2 — Send to the LLM:**

```
Call _client.chat.completions.create() with:
  - model: the model name from config (LLM_MODEL)
  - messages: a list with one dict — {"role": "user", "content": prompt}
    (system-design.md shows an optional system message too — either shape works)
  - max_tokens: a reasonable limit (e.g., 200–300) to keep responses concise

Extract the response text from:
  response.choices[0].message.content
```

---

**Step 3 — Parse the response:**

```
The LLM is asked to return:
  Line 1: "Label: <label>"
  Line 2+: 1–2 sentences of reasoning

Parsing steps:
1. Strip leading/trailing whitespace from the full response text.
2. Split on the first newline to separate the label line from the reasoning:
     first_line, _, reasoning_text = response_text.strip().partition("\n")
3. Extract the label value from the first line:
     - Strip whitespace, then split on ":" to get the part after the colon.
     - Strip and lowercase the result.
     Example: "Label: Interview" → "interview"
4. Strip whitespace from reasoning_text to use as the reasoning string.
5. If the first line doesn't contain ":" or the extracted value is empty,
   treat the label as "unknown" and use the full response as the reasoning
   (see Step 4 and Step 5).
```

---

**Step 4 — Validate the label:**

```
After extracting the label string in Step 3, check whether it is in VALID_LABELS:
  VALID_LABELS = {"interview", "solo", "panel", "narrative"}

  if extracted_label in VALID_LABELS:
      label = extracted_label
  else:
      label = "unknown"

The reasoning string is kept as-is regardless — it is still useful for
debugging even when the label is "unknown".

Do not attempt to fuzzy-match or correct a near-miss label (e.g. "interviews"
or "Interview format"). Silently coercing an invalid value hides prompt
problems. Set it to "unknown" and let the caller see the raw reasoning.
```

---

**Step 5 — Handle errors gracefully:**

```
Wrap the entire body of classify_episode() in a try/except block.

Things that can go wrong:
  - Network or API error (timeout, rate limit, authentication failure)
    → the _client.chat.completions.create() call raises an exception
  - Empty or None response content
    → response.choices[0].message.content is None or ""
  - Unparseable response (no colon on first line, model ignored the format)
    → Step 3 produces an empty label string, Step 4 sets label = "unknown"

For all of these, return the same fallback dict so the evaluation loop
can continue:

  except Exception as e:
      return {
          "label": "unknown",
          "reasoning": f"Error during classification: {e}",
      }

Do not re-raise or print/log the exception separately — storing it in
"reasoning" is enough for debugging without crashing the loop.
The evaluation loop treats "unknown" as a wrong answer, so failures are
scored but don't stop execution.
```

---

### Return value structure

```python
{
    "label": str,      # one of VALID_LABELS, or "unknown" if invalid/error
    "reasoning": str,  # brief explanation from the LLM
}
```

---

## Notes on label quality

The classifier is only as good as your labels. If your training examples have
inconsistent or ambiguous labels, the LLM will learn the wrong pattern.

Before implementing the classifier, re-read `data/taxonomy.md` and double-check
any labels you're unsure about. Annotation quality is part of the lab.

---

## Implementation Notes

*Fill this in after implementing and testing both functions.*

**Test: what does the raw LLM response look like for one episode?**

```
Episode tested: The Case for Four-Day Workweeks
Raw response text: I've been thinking about the four-day workweek for months, and I want to lay out the case for it as clearly as I can. I'm going to cover the productivity research, the companies that have tried it, the objections I find compelling versus the ones I don't, and what I think it would actually take for this to become mainstream. This is a topic I have a real view on, and I want to share it.
```

**How did you parse the label out of the response?**

```
1. Strip leading/trailing whitespace from the full response:
     response_text = response.choices[0].message.content.strip()

2. Split on the first newline to separate the label line from the reasoning:
     first_line, _, reasoning_text = response_text.partition("\n")

3. Guard: if ":" is not in first_line, return label="unknown" and reasoning=full response.

4. Extract the label value from the first line:
     extracted_label = first_line.split(":", 1)[1].strip().lower()
   split(":", 1) limits splitting to the first colon so values with colons are safe.
   .strip() removes whitespace, .lower() normalizes capitalization.
   Example: "Label: Interview" → "interview"

5. Guard: if extracted_label is empty (e.g. response was "Label:"), return "unknown".

6. Validate: if extracted_label is in VALID_LABELS, use it; otherwise label = "unknown".
   reasoning_text is returned as-is in both cases.
```

**Did any episodes return `"unknown"`? If so, why?**

```
No
```

**One thing about the output format that surprised you:**

```
[your answer here]
```
