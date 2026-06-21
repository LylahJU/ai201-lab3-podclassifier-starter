# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
accuracy = number of correct predictions / total predictions

A prediction is correct when predictions[i] == ground_truth[i] (exact string
match). Divide by len(predictions) to get a float between 0.0 and 1.0.
```

---

**Step-by-step logic:**

```
1. If both lists are empty, return 0.0 (see edge case below).
2. Count the number of indices i where predictions[i] == ground_truth[i].
3. Divide that count by len(predictions).
4. Return the result as a float.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0.

There are no predictions to score, so the result is undefined. 0.0 is a safe
sentinel that won't crash downstream code and signals no evaluation took place.
Returning 1.0 would be misleading — the classifier wasn't tested at all.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Compare position by position:
  index 0: "interview" == "interview" ✓
  index 1: "solo"      == "solo"      ✓
  index 2: "panel"     != "solo"      ✗
  index 3: "interview" != "narrative" ✗

correct  = 2
total    = 4
accuracy = 2 / 4 = 0.5
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correctly classified for class C when:
  ground_truth[i] == C   (the episode actually belongs to this class)
  AND
  predictions[i] == C    (the classifier also predicted this class)

Both conditions must hold. An episode predicted as "interview" when the truth
is "panel" does not count as correct for either class.
```

---

**What does "total" mean for a given class?**

```
"total" is the number of episodes whose ground-truth label is C — not the
total number of predictions. It counts how many test episodes actually belong
to class C, regardless of what the classifier predicted for them.
```

---

**Step-by-step logic:**

```
1. Initialize a dict for each label in VALID_LABELS with correct=0, total=0.
2. Loop over each (predicted, truth) pair simultaneously (zip).
3. For each pair:
     - Increment counts[truth]["total"] by 1 (this episode belongs to class truth).
     - If predicted == truth, also increment counts[truth]["correct"] by 1.
4. After the loop, compute accuracy for each label:
     - If total == 0: accuracy = 0.0
     - Otherwise: accuracy = correct / total
5. Return the dict.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set accuracy = 0.0.

Division by zero is undefined. 0.0 matches the docstring contract and is
safe for downstream display code (the bar in format_evaluation_report will
simply be empty). It also makes clear that the class was absent from the
test set, not that the classifier failed on it.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

Trace each pair (predicted → truth):
  i=0: pred=interview, truth=interview → interview: correct+1, total+1
  i=1: pred=interview, truth=solo      → solo: total+1  (wrong prediction)
  i=2: pred=solo,      truth=solo      → solo: correct+1, total+1
  i=3: pred=panel,     truth=panel     → panel: correct+1, total+1
  i=4: pred=panel,     truth=narrative → narrative: total+1  (wrong prediction)

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
