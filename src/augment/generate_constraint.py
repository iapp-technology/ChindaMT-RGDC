import jsonlines
import os
from tqdm import tqdm
import json
import re


prompt_template = """You are an expert in generating instruction constraints for a given simplified query.

Definition of Constraint: The smallest unit of restriction or requirement that can be added to an instruction to make the task more specific or challenging.

Your Task: Given a Simplified Query and its Answer, generate NEW constraints that could be added to the query. The generated constraints should be relevant to the query and the answer, and should make the task more specific without changing its fundamental goal.

Simplified Query: {QUERY}
Answer: {ANSWER}

Follow these steps:
1. Understand the basic goal of the simplified query.
2. Analyze the answer to understand what aspects can be constrained.
3. Generate NEW constraints that are:
   - Relevant to the query and answer
   - Verifiable in the response
   - Not already explicitly stated in the query
4. Categorize constraints into the following types:

Constraint Types:
- Content Constraints:
    - Specific Terms or Symbols: Mandatory use of certain terms or symbols (e.g., must include the word 'beautiful').
    - Required Elements or Concepts: Specific elements or concepts that must be included (e.g., must mention the Great Wall).
    - Thematic Directives: Instructions about thematic content, perspective, or focus (e.g., focus on environmental impact).

- Numerical Constraints:
    - Quantities related to the content: number of points, sentences, paragraphs, word count, or examples (e.g., exactly three sentences, at least 5 examples).

- Stylistic Constraints:
    - Tone and style requirements (e.g., formal, informal, conversational, humorous).
    - Specific language or terminology preferences (e.g., encyclopedic style, avoid jargon).

- Format Constraints:
    - Structure or format requirements (e.g., list, JSON, bullet points, table).
    - Programming language specifications (e.g., Java, Python).
    - Presentation styles (e.g., markdown, code block format).

- Linguistic Constraints:
    - Language specifications (e.g., in English, in Spanish).
    - Sentence structure requirements (e.g., use only simple sentences, imperative form).
    - Word-level requirements (e.g., lowercase, single-rhyme, alliteration).
 
Response Format:
- Generate constraints that are appropriate for the given query.
- Ensure generated constraints do not overlap with what is already in the query.
- Only include constraint types that are relevant.
- Present each constraint as a dictionary with a 'constraint' key.

```json
{{
    "Content Constraints": [
        {{"constraint": "..."}},
        {{"constraint": "..."}}
    ],
    "Numerical Constraints": [...],
    "Stylistic Constraints": [...],
    "Format Constraints": [...],
    "Linguistic Constraints": [...]
}}
```

Examples:

---
Example 1:
Query: "How do I check if a user pressed the cancel button on a prompt in JS?"
Answer: "You can check if a user pressed cancel by comparing the prompt result to null. When the user clicks cancel, the prompt() function returns null instead of a string. So you simply use: if (result === null) {{ /* user cancelled */ }}"

Response:
```json
{{
    "Numerical Constraints": [
        {{"constraint": "Answer with exactly three sentences."}}
    ]
}}
```
Explanation: The query asks for a JavaScript explanation. A numerical constraint "exactly three sentences" is generated to add structure to the response. This constraint is verifiable by counting the sentences in the answer (which has exactly 3 sentences).

---
Example 2:
Query: "Explain quantum computing"
Answer: "Quantum computing is like having a super-powered calculator that can try many answers at once instead of one at a time. Regular computers use bits (0s and 1s), but quantum computers use 'qubits' that can be both 0 and 1 at the same time. This lets them solve certain complex problems much faster than regular computers."

Response:
```json
{{
    "Content Constraints": [
        {{"constraint": "Explain in simple terms using analogies."}}
    ]
}}
```
Explanation: The query asks for an explanation of quantum computing. A content constraint "explain in simple terms using analogies" is generated because the answer uses accessible language and analogies (like "super-powered calculator"). This makes the explanation requirement more specific without changing the core goal.

---
Example 3:
Query: "I want to replace string like 'This is a new {{object}} at {{place}}' with a Map, {{object: 'student', place: 'point 3, 4'}}, and get a result 'This is a new student at point 3, 4'. How can I do this?"
Answer: "public String replaceWithMap(String template, Map<String, String> values) {{\n    String result = template;\n    for (Map.Entry<String, String> entry : values.entrySet()) {{\n        result = result.replace('{{{{' + entry.getKey() + '}}}}', entry.getValue());\n    }}\n    return result;\n}}"

Response:
```json
{{
    "Format Constraints": [
        {{"constraint": "Answer using Java programming language."}}
    ]
}}
```
Explanation: The query asks how to perform string replacement. The answer is provided in Java code. A format constraint "Answer using Java" is generated to specify the programming language requirement, making the query more precise about the expected output format.

---
Example 4:
Query: "Write a program which accepts a command line parameter as a question and sends it to a server via HTTP GET method."
Answer: "import sys\nimport requests\n\nif len(sys.argv) < 2:\n    print('Usage: python script.py <question>')\n    sys.exit(1)\n\nquestion = sys.argv[1]\nresponse = requests.get('http://example.com/api', params={{'question': question}})\nprint(response.text)"

Response:
```json
{{
    "Format Constraints": [
        {{"constraint": "Write as a Python program."}}
    ]
}}
```
Explanation: The query asks for a program without specifying the language. Since the answer is in Python, a format constraint "Write as a Python program" is generated to make the language requirement explicit. This ensures the response will be in Python.

---
Example 5:
Query: "Recommend books"
Answer: "• 'Dream of the Red Chamber' by Cao Xueqin - A classic tale of noble families\n• 'Journey to the West' by Wu Cheng'en - Adventures of a monk and his companions\n• 'Romance of the Three Kingdoms' by Luo Guanzhong - Epic historical warfare\n• 'Water Margin' by Shi Nai'an - Tales of outlaws and heroes\n• 'The Scholar' by Wu Jingzi - Satire of the imperial examination system\n• 'Strange Tales from a Chinese Studio' by Pu Songling - Collection of supernatural stories\n• 'Border Town' by Shen Congwen - A love story set in rural China\n• 'Family' by Ba Jin - Story of generational conflict\n• 'Rickshaw Boy' by Lao She - Tale of a rickshaw puller's struggles\n• 'To Live' by Yu Hua - A man's journey through tumultuous times"

Response:
```json
{{
    "Stylistic Constraints": [
        {{"constraint": "Use Shakespeare's tone."}}
    ],
    "Format Constraints": [
        {{"constraint": "Use bullet points in the answer."}}
    ],
    "Numerical Constraints": [
        {{"constraint": "Recommend exactly ten books."}}
    ],
    "Content Constraints": [
        {{"constraint": "Focus on Chinese literature."}}
    ]
}}
```
Explanation: The query simply asks to "recommend books" which is very broad. Multiple constraints are generated based on the answer: (1) Stylistic constraint for "Shakespeare's tone" adds a creative challenge; (2) Format constraint for "bullet points" matches the answer's structure; (3) Numerical constraint for "ten books" matches the count in the answer; (4) Content constraint for "Chinese literature" narrows the scope to match the answer's focus.

---

Please only provide the response in JSON format.
""".strip()


def packing(data_path, output_path, model="Qwen/Qwen3.5-35B-A3B-FP8"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_count = sum(1 for _ in f)
        if existing_count > 0:
            print(f"Resuming from record {existing_count}")

    with jsonlines.open(output_path, "a") as writer:
        for idx, d in enumerate(tqdm(data, desc="Generating constraint prompts")):
            if idx < existing_count:
                continue
            simplified_instruction = "\n".join([d["instruction"], d["input"]])
            query = prompt_template.format(
                QUERY=simplified_instruction, ANSWER=d["output"]
            )
            messages = [{"role": "user", "content": query}]
            writer.write(
                {
                    "custom_id": f"request-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                }
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate constraint prompts")
    parser.add_argument("--input", required=True, help="Input data file (JSON)")
    parser.add_argument("--output", required=True, help="Output prompts file (JSONL)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B-FP8")
    args = parser.parse_args()

    packing(args.input, args.output, model=args.model)
