import os
import json
import torch
import argparse
from tqdm import tqdm

import torch.nn as nn

log_softmax = nn.LogSoftmax(dim=-1)
nll_loss = nn.NLLLoss(reduction="none")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="wiz", help="wiz, alpaca")
    parser.add_argument("--mod", type=str, default="pre", help="pre, cherry")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--use_compile", action="store_true", help="Use torch.compile for faster inference")
    args = parser.parse_args()
    return args


# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):
    """
    Calculate perplexity and sentence embedding for the entire input text.

    This function evaluates the model's prediction performance on the whole text
    by computing the cross-entropy loss averaged over all tokens, then converting
    it to perplexity. It also extracts the sentence embedding by mean pooling
    the hidden states from the last layer.

    Args:
        tokenizer: The tokenizer to encode the input text into token IDs.
        model: The language model (must support output_hidden_states=True).
        text (str): The input text to evaluate.
        max_length (int): Maximum sequence length for tokenization. Text will be
                         truncated if longer than this value.

    Returns:
        tuple: A tuple containing:
            - perplexity (torch.Tensor): Perplexity score (exp(loss)) of the text.
                                        Lower values indicate the model is more
                                        confident about the text.
            - sentence_embedding (torch.Tensor): Mean-pooled embedding vector of
                                                shape [1, hidden_dim] representing
                                                the semantic representation of
                                                the entire text.

    Note:
        - Both tensors are moved to CPU before returning.
        - The function uses torch.no_grad() for inference efficiency.
        - Used when mod='pre' to evaluate instruction alone.

    Example:
        >>> ppl, emb = get_perplexity_and_embedding_whole_text(
        ...     tokenizer, model, "### Instruction: Translate", max_length=512
        ... )
        >>> print(f"Perplexity: {ppl.item()}, Embedding shape: {emb.shape}")
    """

    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids.contiguous(), output_hidden_states=True)
    # Cross-entropy loss of every token vs label
    loss = outputs.loss
    # perplexity is the exponential of the loss
    perplexity = torch.exp(loss)

    # get the hidden states of last layer of the model
    hidden_states = outputs.hidden_states
    embeddings = hidden_states[-1]
    # mean pooling all the tokens in the input sequence
    sentence_embedding = embeddings.mean(dim=1)

    return perplexity.to("cpu"), sentence_embedding.to("cpu")


# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(
    tokenizer, model, text, target_span, max_length
):
    """
    Calculate perplexity and token-wise loss for a specific span within the text.

    This function evaluates the model's prediction performance only on the target
    span (e.g., the output/response part) by masking out the rest of the text.
    It computes both the overall perplexity and detailed token-wise losses for
    fine-grained analysis.

    Args:
        tokenizer: The tokenizer to encode the input text into token IDs.
        model: The language model (must support output_hidden_states=True).
        text (str): The full input text containing both context and target span.
        target_span (str): The substring to evaluate. The function will find the
                          last occurrence of this span in the text and only
                          compute loss for tokens in this span.
        max_length (int): Maximum sequence length for tokenization. Text will be
                         truncated if longer than this value.

    Returns:
        tuple: A tuple containing:
            - perplexity (torch.Tensor): Perplexity score (exp(loss)) computed
                                        only for the target span. Tokens before
                                        the target span are masked (set to -100).
            - placeholder (int): Always returns 0 as a placeholder for
                                compatibility with the calling code.
            - losses (list): List of token-wise loss values for each token in
                           the target span. Each element is the negative log
                           likelihood loss for predicting that token.
                           Length equals (end_token - 1).

    Note:
        - The function masks tokens before the target_span by setting labels to -100,
          which tells the loss function to ignore those tokens.
        - Token-wise losses are computed manually using log_softmax and NLLLoss
          for detailed analysis.
        - Both perplexity tensor and losses list are moved to CPU before returning.
        - Used when mod='cherry' to evaluate output with/without instruction context.

    Example:
    >>> text = "### Instruction: Translate\n\n### Response: Hello"
        >>> ppl, _, losses = get_perplexity_and_embedding_part_text(
        ...     tokenizer, model, text, "Hello", max_length=512
        ... )
        >>> print(f"Perplexity: {ppl.item()}, Token losses: {losses}")
    """
    # find only target text for the loss calculation
    input_ids = tokenizer.encode(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)

    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    # mask token ที่ไม่ต้องการคำนวณ loss
    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    logits = outputs.logits
    log_probs = log_softmax(logits[0, :-1])
    targets = input_ids[0, 1:]
    losses = nll_loss(log_probs, targets).tolist()

    return perplexity.to("cpu"), 0, losses


def prepare_text_for_cherry(data_item, prompt_type):
    """Prepare text for cherry mode processing."""
    instruct_i = data_item["instruction"]
    output_i = data_item["output"]

    direct_answer_text = "### Response:" + output_i

    if prompt_type == "wiz":
        whole_text = instruct_i + "\n\n### Response:" + output_i
        input_i = data_item.get("input", "")
        if input_i != "":
            whole_text = instruct_i + "\nInput:" + input_i + "\n\n### Response:" + output_i
    elif prompt_type == "alpaca":
        input_i = data_item.get("input", "")
        if input_i == "":
            temp_dict = {"instruction": instruct_i}
            promt_to_use = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
            whole_text = promt_to_use + output_i
            instruct_i = promt_to_use
        else:
            temp_dict = {"instruction": instruct_i, "input": input_i}
            promt_to_use = PROMPT_DICT["prompt_input"].format_map(temp_dict)
            whole_text = promt_to_use + output_i
            instruct_i = promt_to_use

    return instruct_i, output_i, direct_answer_text, whole_text


def _batched_forward(tokenizer, model, texts, target_spans, max_lengths):
    """Batched forward pass: left-pad, single model call, per-sample loss extraction."""
    encodings = []
    for text, ml in zip(texts, max_lengths):
        ids = tokenizer.encode(text, truncation=True, max_length=max(ml, 1))
        encodings.append(ids)

    batch_max_len = max(len(e) for e in encodings)
    pad_id = tokenizer.pad_token_id

    input_ids_list = []
    attention_mask_list = []
    pad_lens = []

    for enc in encodings:
        pad_len = batch_max_len - len(enc)
        pad_lens.append(pad_len)
        input_ids_list.append([pad_id] * pad_len + enc)
        attention_mask_list.append([0] * pad_len + [1] * len(enc))

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits

    results = []
    for i in range(len(texts)):
        pad_len = pad_lens[i]
        enc = encodings[i]

        s_logits = logits[i, pad_len:]
        s_ids = torch.tensor(enc, dtype=torch.long, device=device)

        log_probs = log_softmax(s_logits[:-1])
        targets = s_ids[1:]
        token_losses_t = nll_loss(log_probs, targets)

        start_index = texts[i].rfind(target_spans[i])
        start_token = len(tokenizer.encode(texts[i][:start_index]))

        target_losses = token_losses_t[start_token - 1:]
        ppl = torch.exp(target_losses.mean()).cpu()

        losses = token_losses_t.tolist()
        results.append((ppl, losses))

    del logits, outputs
    torch.cuda.empty_cache()

    return results


def _batched_forward_safe(tokenizer, model, texts, target_spans, max_lengths):
    """OOM-safe wrapper: splits batch in half on CUDA OOM and retries."""
    try:
        return _batched_forward(tokenizer, model, texts, target_spans, max_lengths)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(texts) <= 1:
            raise
        mid = len(texts) // 2
        left = _batched_forward_safe(tokenizer, model, texts[:mid], target_spans[:mid], max_lengths[:mid])
        right = _batched_forward_safe(tokenizer, model, texts[mid:], target_spans[mid:], max_lengths[mid:])
        return left + right


def process_batch_cherry(tokenizer, model, batch_data, max_length, prompt_type):
    """Process a batch with true batched inference (2 forward passes per batch)."""
    direct_texts = []
    whole_texts = []
    output_spans = []
    direct_max_lens = []

    for data_item in batch_data:
        instruct_i, output_i, direct_answer_text, whole_text = prepare_text_for_cherry(
            data_item, prompt_type
        )
        instruct_ids = tokenizer.encode(
            instruct_i, truncation=True, max_length=max_length
        )
        instruct_len = len(instruct_ids)

        direct_texts.append(direct_answer_text)
        whole_texts.append(whole_text)
        output_spans.append(output_i)
        direct_max_lens.append(max_length - instruct_len + 4)

    direct_results = _batched_forward_safe(
        tokenizer, model, direct_texts, output_spans, direct_max_lens
    )
    whole_results = _batched_forward_safe(
        tokenizer, model, whole_texts, output_spans, [max_length] * len(whole_texts)
    )

    batch_results = []
    for i in range(len(batch_data)):
        d_ppl, d_losses = direct_results[i]
        w_ppl, w_losses = whole_results[i]
        batch_results.append({
            "ppl": [0, d_ppl, w_ppl],
            "token_loss": [[], d_losses, w_losses],
        })

    return batch_results


def main():

    args = parse_args()
    print(args)

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    model_kwargs = {
        "device_map": "auto",
        "cache_dir": "../cache",
        "output_hidden_states": (args.mod == "pre"),
        "dtype": torch.float16,
    }

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
        print("Loaded with flash_attention_2")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )
        print("Loaded with default attention")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir="../cache"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model.eval()

    if args.save_path[-3:] != ".pt":
        args.save_path += ".pt"
    if os.path.exists(args.save_path):
        print("save_path exists!")
        # raise Exception

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    import time

    strat_time = time.time()
    new_data = []
    batch_size = args.batch_size

    if args.mod == "cherry":
        # Sort by text length so batches have similar-length sequences (less padding waste)
        indexed_data = list(enumerate(sampled_data))
        indexed_data.sort(
            key=lambda x: len(x[1].get("instruction", "")) + len(x[1].get("output", ""))
        )
        sorted_indices = [idx for idx, _ in indexed_data]
        sorted_data = [item for _, item in indexed_data]

        # Dynamic batch sizing: scale inversely with sequence length to keep VRAM stable
        max_tokens_budget = batch_size * 512
        batches = []
        i = 0
        while i < len(sorted_data):
            est_chars = len(sorted_data[i].get("instruction", "")) + len(sorted_data[i].get("output", ""))
            est_tokens = max(est_chars // 2, 16)
            dynamic_bs = max(1, min(batch_size, max_tokens_budget // est_tokens))
            batches.append(sorted_data[i:i + dynamic_bs])
            i += dynamic_bs

        print(f"Processing {len(sorted_data)} samples in {len(batches)} batches (dynamic size, budget={max_tokens_budget})")
        sorted_results = []

        for batch_data in tqdm(batches, desc="Processing batches"):
            batch_results = process_batch_cherry(
                tokenizer, model, batch_data, args.max_length, args.prompt
            )
            sorted_results.extend(batch_results)

        # Restore original order so .pt indices match the input JSON
        new_data = [None] * len(sorted_results)
        for sorted_idx, orig_idx in enumerate(sorted_indices):
            new_data[orig_idx] = sorted_results[sorted_idx]

    else:
        # Pre mode - process individually (needs hidden states)
        for i in tqdm(range(len(sampled_data)), desc="Processing samples"):
            data_i = sampled_data[i]
            instruct_i = data_i["instruction"]

            if args.prompt == "wiz":
                input_i = data_i.get("input", "")
                if input_i != "":
                    instruct_i = instruct_i + "\nInput:" + input_i
            elif args.prompt == "alpaca":
                input_i = data_i.get("input", "")
                if input_i == "":
                    temp_dict = {"instruction": instruct_i}
                    instruct_i = PROMPT_DICT["prompt_no_input"].format_map(temp_dict)
                else:
                    temp_dict = {"instruction": instruct_i, "input": input_i}
                    instruct_i = PROMPT_DICT["prompt_input"].format_map(temp_dict)

            temp_data_i = {}
            ppl_ins_alone, emb_ins_alone = get_perplexity_and_embedding_whole_text(
                tokenizer, model, instruct_i, args.max_length
            )
            temp_data_i["ppl"] = [ppl_ins_alone, 0, 0]
            temp_data_i["sent_emb"] = [emb_ins_alone, 0, 0]
            new_data.append(temp_data_i)

    print("New data len:", len(new_data))
    torch.save(new_data, args.save_path)

    print("Time Used:", (time.time() - strat_time) / 60, "(min)")


if __name__ == "__main__":
    main()
