#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理 + 评估脚本。两种用法：

  1) 评估测试集:  python3 inference.py --adapter ./output
  2) 单句推理:    python3 inference.py --adapter ./output --interactive "帮我订明晚7点3个人包间"
"""
import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def extract_json(text):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", default="./output")
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--interactive", default=None)
    args = p.parse_args()

    with open("schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    system_prompt = schema["system_prompt"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    def generate(utt, max_new_tokens=128):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    if args.interactive:
        raw = generate(args.interactive)
        obj = extract_json(raw)
        print("原始输出:", raw)
        print("解析结果:", json.dumps(obj, ensure_ascii=False) if obj else "(解析失败)")
        return

    ds = load_dataset("json", data_files=args.data, split="train")
    intent_correct = 0
    tp = fp = fn = total = 0
    intent_errs = []
    slot_errs = []

    for ex in ds:
        gold = json.loads(ex["messages"][-1]["content"])
        utt = ex["messages"][1]["content"]
        pred = extract_json(generate(utt))
        total += 1
        if pred is not None and pred.get("intent") == gold.get("intent"):
            intent_correct += 1
        ps = (pred.get("slots") or {}) if pred else {}
        gs = gold.get("slots") or {}
        p_set = {(k, str(v)) for k, v in ps.items()}
        g_set = {(k, str(v)) for k, v in gs.items()}
        tp += len(p_set & g_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
        if pred is None or pred.get("intent") != gold.get("intent"):
            if len(intent_errs) < 20:
                intent_errs.append({"utterance": utt, "gold": gold, "pred": pred})
        elif p_set != g_set and len(slot_errs) < 30:
            slot_errs.append({"utterance": utt, "gold": gs, "pred": ps})

    intent_acc = intent_correct / total if total else 0
    sp = tp / (tp + fp) if (tp + fp) else 0
    sr = tp / (tp + fn) if (tp + fn) else 0
    sf = 2 * sp * sr / (sp + sr) if (sp + sr) else 0

    print(f"样本数: {total}")
    print(f"意图准确率: {intent_acc:.4f}")
    print(f"槽位 F1: {sf:.4f} (Precision={sp:.4f}, Recall={sr:.4f})")
    if intent_errs:
        print("\n意图错误样例:")
        for e in intent_errs:
            print(f"  - {e['utterance']}")
            print(f"      gold: {json.dumps(e['gold'], ensure_ascii=False)}")
            print(f"      pred: {json.dumps(e['pred'], ensure_ascii=False) if e['pred'] else '(空)'}")
    if slot_errs:
        print("\n槽位有出入的样例（gold vs pred）:")
        for e in slot_errs:
            print(f"  - {e['utterance']}")
            print(f"      gold: {json.dumps(e['gold'], ensure_ascii=False)}")
            print(f"      pred: {json.dumps(e['pred'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()