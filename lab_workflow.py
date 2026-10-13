"""Student-owned Day 10 Qwen3-VL training and evaluation workflow.

Run each GPU command in a fresh process on Kaggle. The examiner-owned scripts
and tests are imported, never modified. No metric is synthesized here.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = (
    "Locate road damage in this image. Detect only pothole, crack, and manhole. "
    "Return only a JSON list of objects with label and bbox_2d, using integer "
    "coordinates normalized to 0-1000 relative to image width and height. "
    'Example: [{"label":"pothole","bbox_2d":[100,200,300,400]}]. '
    "If there is no road damage, return []."
)
LLM_TARGET = re.compile(
    r"(?:^|\.)language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)
VIT_TARGET = re.compile(
    r"(?:^|\.)visual\.blocks\.\d+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])$"
)
PROJECTOR = re.compile(r"(?:^|\.)visual\.(?:merger|deepstack_merger_list)\.")
PROJECTOR_MODULE = re.compile(r"(?:^|\.)visual\.(?:merger|deepstack_merger_list\.\d+)$")
SETUPS = {"e1": (False, False), "e2": (False, True), "e3": (True, True)}


def yolo_to_box(label, yolo):
    """Convert center/size YOLO coordinates to clipped 0–1000 xyxy."""
    if label not in CLASSES or len(yolo) != 4:
        raise ValueError("invalid YOLO object")
    xc, yc, w, h = map(float, yolo)
    values = [(xc - w / 2) * 1000, (yc - h / 2) * 1000,
              (xc + w / 2) * 1000, (yc + h / 2) * 1000]
    box = [round(max(0, min(1000, value))) for value in values]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"degenerate box after rounding: {box}")
    return {"label": label, "bbox_2d": box}


def answer_for(objects):
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda obj: (obj["bbox_2d"][1], obj["bbox_2d"][0], obj["label"]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def check_data(data_dir="data"):
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "splits/meta.json").read_text(encoding="utf-8"))
    expected = {"train": 1422, "val": 276, "test": 311}
    sessions, summary = {}, {}
    for split in expected:
        rows = read_jsonl(data_dir / "splits" / f"{split}.jsonl")
        if len(rows) != expected[split]:
            raise AssertionError(f"{split}: {len(rows)} images, expected {expected[split]}")
        classes = Counter(obj["label"] for row in rows for obj in row["objects"])
        for row in rows:
            previous = sessions.setdefault(row["session"], split)
            if previous != split:
                raise AssertionError(f"session leakage: {row['session']}")
            if not (data_dir / row["image_path"]).is_file():
                raise FileNotFoundError(data_dir / row["image_path"])
        summary[split] = {"images": len(rows), "boxes": dict(classes)}
        if summary[split] != meta["counts"][split]:
            raise AssertionError(f"{split}: counts differ from meta.json")
    if not str(meta["dataset_version"]).startswith("d10-"):
        raise AssertionError("unexpected dataset_version")
    print(json.dumps({"dataset_version": meta["dataset_version"], "splits": summary}, indent=2))
    return summary


def _messages(image, answer=None):
    conversation = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}
    ]}]
    if answer is not None:
        conversation.append({"role": "assistant", "content": answer})
    return conversation


def encode_example(processor, row, data_dir="data"):
    from PIL import Image
    with Image.open(Path(data_dir) / row["image_path"]) as opened:
        image = opened.convert("RGB")
        prompt = processor.apply_chat_template(
            _messages(image), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        full = processor.apply_chat_template(
            _messages(image, answer_for(row["objects"])),
            add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt")
    for value in (prompt, full):
        value.pop("token_type_ids", None)
    prompt_len = prompt["input_ids"].shape[1]
    if not (full["input_ids"][0, :prompt_len] == prompt["input_ids"][0]).all().item():
        raise AssertionError("assistant answer does not follow the exact prompt prefix")
    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100
    if not (labels != -100).any().item():
        raise AssertionError("no supervised answer tokens")
    full["labels"] = labels
    return {key: value.squeeze(0) if key in
            ("input_ids", "attention_mask", "mm_token_type_ids", "labels") else value
            for key, value in full.items()}


def collate(samples, pad_token_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    if any(set(item) != set(samples[0]) for item in samples):
        raise ValueError("inconsistent processor fields in batch")
    batch = {}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
        if key in samples[0]:
            padding = -100 if key == "labels" else pad_token_id if key == "input_ids" else 0
            batch[key] = pad_sequence([item[key] for item in samples],
                                      batch_first=True, padding_value=padding)
    for key in ("pixel_values", "image_grid_thw"):
        if key in samples[0]:
            batch[key] = torch.cat([item[key] for item in samples], dim=0)
    return batch


def quantized_model():
    import torch
    from bitsandbytes.nn import Linear4bit
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle NVIDIA GPU is required for E0–E5")
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.float16,
        device_map={"": torch.cuda.current_device()})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    # Both modules must remain floating point for E2/E3 and stable output logits.
    for name, module in model.named_modules():
        if (".visual." in f".{name}." or name.endswith("lm_head")) and isinstance(module, Linear4bit):
            raise RuntimeError(f"unexpected 4-bit conversion: {name}")
    return model, processor


def model_preflight(data_dir="data"):
    model, processor = quantized_model()
    row = read_jsonl(Path(data_dir) / "splits/train.jsonl")[0]
    sample = encode_example(processor, row, data_dir)
    answer_ids = sample["labels"][sample["labels"] != -100]
    decoded = processor.tokenizer.decode(answer_ids, skip_special_tokens=False)
    expected = answer_for(row["objects"])
    if expected not in decoded or "<|im_end|>" not in decoded:
        raise AssertionError(f"masked answer mismatch: {decoded[:300]}")
    grid = sample["image_grid_thw"]
    vision_tokens = int(grid.prod(dim=1).sum().item() // processor.image_processor.merge_size ** 2)
    print(f"Masked answer: {decoded[:300]}")
    print(f"Vision tokens: {vision_tokens}; image_grid_thw={grid.tolist()}")
    if vision_tokens != 220:
        raise AssertionError("expected 220 vision tokens; inspect processor resizing")
    del model
    gc.collect()


def configure_lora(base, *, vit=False, projector=False, rank=16):
    import torch
    from peft import LoraConfig, get_peft_model
    if rank < 1:
        raise ValueError("rank must be positive")
    for param in base.parameters():
        param.requires_grad_(False)
    matched = [name for name, _ in base.named_modules()
               if LLM_TARGET.search(name) or (vit and VIT_TARGET.search(name))]
    required = 348 if vit else 252
    if len(matched) != required:
        raise RuntimeError(f"expected {required} LoRA modules, found {len(matched)}: {matched[:5]}")
    if projector:
        modules = [module for name, module in base.named_modules()
                   if PROJECTOR_MODULE.search(name)]
        if not modules:
            raise RuntimeError("projector modules not found")
        for module in modules:
            module.float()
    model = get_peft_model(base, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules=matched, bias="none"))
    if projector:
        changed = 0
        for name, param in model.named_parameters():
            if PROJECTOR.search(name):
                if param.dtype != torch.float32:
                    raise RuntimeError(f"projector not float32: {name}")
                param.requires_grad_(True)
                changed += param.numel()
        if not 100_000_000 < changed < 120_000_000:
            raise RuntimeError(f"unexpected projector parameter count: {changed}")
    model.enable_input_require_grads()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, {"matched_modules": len(matched), "trainable_params": count}


def projector_state(model):
    return {name: param.detach().cpu().clone()
            for name, param in model.named_parameters() if PROJECTOR.search(name)}


def save_model_parts(model, directory, has_projector, metadata=None):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter")
    if has_projector:
        state = projector_state(model)
        if not state:
            raise RuntimeError("cannot save empty projector")
        torch.save(state, directory / "projector.pt")
    if metadata is not None:
        (directory / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_sft(directory, *, trainable=False):
    import torch
    from peft import PeftModel
    base, processor = quantized_model()
    directory = Path(directory)
    projector_file = directory / "projector.pt"
    if projector_file.is_file():
        modules = [module for name, module in base.named_modules()
                   if PROJECTOR_MODULE.search(name)]
        if not modules:
            raise RuntimeError("projector checkpoint exists but modules not found")
        for module in modules:
            module.float()
    model = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=trainable)
    if projector_file.is_file():
        saved = torch.load(projector_file, map_location="cpu", weights_only=True)
        parameters = dict(model.named_parameters())
        if set(saved) != set(projector_state(model)):
            raise RuntimeError("projector checkpoint keys do not match model")
        with torch.no_grad():
            for name, tensor in saved.items():
                parameters[name].copy_(tensor.to(parameters[name].device, parameters[name].dtype))
                if trainable:
                    parameters[name].requires_grad_(True)
    if trainable:
        model.enable_input_require_grads()
    model.train(mode=trainable)
    return model, processor


def train_sft(name, *, base_setup=None, fraction=1.0, data_dir="data",
              batch_size=2, grad_accum=4, max_steps=300):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_cosine_schedule_with_warmup
    if not (0 < fraction <= 1) or min(batch_size, grad_accum, max_steps) < 1:
        raise ValueError("invalid training size or steps")
    key = (base_setup or name).lower()
    vit, has_projector = SETUPS[key]
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    base, processor = quantized_model()
    processor.tokenizer.padding_side = "right"
    model, meta = configure_lora(base, vit=vit, projector=has_projector)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    rows = read_jsonl(Path(data_dir) / "splits/train.jsonl")
    if fraction < 1:
        rows = random.Random(42).sample(rows, round(len(rows) * fraction))
    class Examples(Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            return encode_example(processor, rows[index], data_dir)
    loader = DataLoader(Examples(), batch_size=batch_size, shuffle=True,
                        collate_fn=lambda items: collate(items, processor.tokenizer.pad_token_id))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(.05 * max_steps)), max_steps)
    scaler = torch.amp.GradScaler("cuda")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    iterator = iter(loader)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, max_steps + 1):
        step_losses = []
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"{name}: nonfinite loss at step {step}")
            scaler.scale(loss).backward()
            step_losses.append(float(loss.detach()) * grad_accum)
            del batch, loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= before:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 10 == 0:
            mean_loss = sum(step_losses) / len(step_losses)
            losses.append({"step": step, "loss": mean_loss})
            print(f"{name} {step}/{max_steps}: loss={mean_loss:.4f}", flush=True)
        if step % 100 == 0 or step == max_steps:
            record = {**meta, "name": name, "step": step, "base_setup": key,
                      "train_fraction": fraction, "rank": 16, "lr": 2e-4,
                      "batch_size": batch_size, "grad_accum": grad_accum,
                      "max_steps": max_steps, "seed": 42,
                      "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                      "train_minutes": (time.monotonic() - began) / 60,
                      "loss_history": losses}
            save_model_parts(model, Path("work") / name.lower(), has_projector, record)
    return record


def predict(name, *, checkpoint=None, split="test", data_dir="data", batch_size=2):
    import torch
    from PIL import Image
    model, processor = load_sft(checkpoint) if checkpoint else quantized_model()
    model.eval()
    processor.tokenizer.padding_side = "left"
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    output = []
    device = torch.device("cuda", torch.cuda.current_device())
    for start in range(0, len(rows), batch_size):
        group = rows[start:start + batch_size]
        messages = []
        for row in group:
            with Image.open(Path(data_dir) / row["image_path"]) as opened:
                messages.append(_messages(opened.convert("RGB")))
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True)
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        prompt_length = inputs["input_ids"].shape[1]
        text = processor.batch_decode(generated[:, prompt_length:],
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
        output.extend({"image_id": row["image_id"], "raw": raw}
                      for row, raw in zip(group, text))
        if start == 0 or (start // batch_size + 1) % 20 == 0:
            print(f"{name}: {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    path = Path("work") / f"{name.lower()}_{split}.jsonl"
    write_jsonl(path, output)
    return path


def score(name, predictions, *, metadata=None, split="test", data_dir="data"):
    suffix = "" if split == "test" else f"_{split}"
    out = Path("report/evidence") / f"{name.lower()}{suffix}.json"
    cmd = [sys.executable, "scripts/score.py", "--pred", str(predictions),
           "--name", name.upper(), "--out", str(out), "--split", split,
           "--splits-dir", str(Path(data_dir) / "splits")]
    if metadata:
        for key in ("trainable_params", "peak_vram_gb", "train_minutes"):
            cmd += ["--meta", f"{key}={metadata[key]}"]
    subprocess.run(cmd, check=True)
    return json.loads(out.read_text(encoding="utf-8"))


def best_sft():
    results = {}
    for name in SETUPS:
        path = Path("report/evidence") / f"{name}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["split"] != "test":
            raise ValueError(f"{name} is not test evidence")
        results[name] = result
    if len({value["dataset_version"] for value in results.values()}) != 1:
        raise ValueError("dataset versions disagree")
    return max(SETUPS, key=lambda name: results[name]["micro"]["f1"])


def train_grpo(best, data_dir="data"):
    import torch
    from datasets import Dataset, Image as DatasetImage
    from trl import GRPOConfig, GRPOTrainer
    from bonus_reward import trl_reward
    from scripts.score import _gt_boxes
    if best not in SETUPS:
        raise ValueError(best)
    rows = read_jsonl(Path(data_dir) / "splits/train.jsonl")
    dataset = Dataset.from_dict({
        "prompt": [[{"role": "user", "content": PROMPT}] for _ in rows],
        "image": [str(Path(data_dir) / row["image_path"]) for row in rows],
        "ground_truth": [json.dumps(_gt_boxes(row)) for row in rows],
    }).cast_column("image", DatasetImage())
    model, processor = load_sft(Path("work") / best, trainable=True)
    model.config.use_cache = False
    processor.tokenizer.padding_side = "left"
    record_base = json.loads((Path("work") / best / "run.json").read_text())
    args = GRPOConfig(
        output_dir="work/e5_trainer", max_steps=60, learning_rate=5e-6,
        num_generations=4, max_completion_length=128, beta=0.0,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False, remove_unused_columns=False,
        logging_steps=5, save_steps=20, save_total_limit=2,
        report_to="none", seed=42,
    )
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    trainer = GRPOTrainer(model=model, processing_class=processor, args=args,
                          train_dataset=dataset, reward_funcs=trl_reward)
    trainer.train()
    trained = trainer.accelerator.unwrap_model(trainer.model)
    record = {"name": "E5", "source_setup": best, "max_steps": 60,
              "num_generations": 4, "learning_rate": 5e-6,
              "trainable_params": sum(p.numel() for p in trained.parameters() if p.requires_grad),
              "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
              "train_minutes": (time.monotonic() - began) / 60,
              "sft_baseline_f1": json.loads((Path("report/evidence") / f"{best}.json").read_text())["micro"]["f1"],
              "trainer_log_history": trainer.state.log_history}
    save_model_parts(trained, "work/e5", SETUPS[best][1], record)
    return record


def show_results():
    for path in sorted(Path("report/evidence").glob("e*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        f1 = result["micro"]["f1"]
        print(f"{path.stem:15s} F1={f1:.4f} CI95={result['f1_ci95']} "
              f"valid={result['parse_valid_rate']:.3f} FP/img={result['fp_per_image']:.3f}")


def draw_ground_truth(data_dir="data", count=6):
    """Check six train labels visually before the GPU experiments."""
    from PIL import Image, ImageDraw
    rows = read_jsonl(Path(data_dir) / "splits/train.jsonl")
    selected = [rows[i * len(rows) // count] for i in range(count)]
    target = Path("report/figures")
    target.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(selected, 1):
        with Image.open(Path(data_dir) / row["image_path"]) as opened:
            image = opened.convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for obj in row["objects"]:
            x1, y1, x2, y2 = yolo_to_box(obj["label"], obj["yolo"])["bbox_2d"]
            box = (x1 * width / 1000, y1 * height / 1000,
                   x2 * width / 1000, y2 * height / 1000)
            draw.rectangle(box, outline="lime", width=3)
            draw.text((box[0], box[1]), obj["label"], fill="lime")
        path = target / f"gt_{index:02d}_{row['image_id']}.png"
        image.save(path)
        print(path)


def draw_examples(name, *, data_dir="data", split="test", count=6):
    """Save six paired GT/prediction images for the lab's visual evidence."""
    from PIL import Image, ImageDraw
    from scripts.score import _gt_boxes, _match, parse_output
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    predictions = {row["image_id"]: row["raw"] for row in
                   read_jsonl(Path("work") / f"{name}_{split}.jsonl")}
    candidates = []
    for row in rows:
        parsed = parse_output(predictions[row["image_id"]])
        per, _ = _match(_gt_boxes(row), parsed or [])
        tp, fp, fn = (sum(value[i] for value in per.values()) for i in range(3))
        candidates.append((row, parsed or [], tp, fp, fn))
    correct = [item for item in candidates if item[2] > 0 and item[3] == item[4] == 0]
    incorrect = sorted((item for item in candidates if item[3] or item[4]),
                       key=lambda item: (item[3] + item[4], item[2]), reverse=True)
    chosen = correct[:count // 2] + incorrect[:count - min(len(correct), count // 2)]
    seen = {item[0]["image_id"] for item in chosen}
    chosen += [item for item in candidates if item[0]["image_id"] not in seen][:count - len(chosen)]
    target = Path("report/figures")
    target.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, (row, parsed, tp, fp, fn) in enumerate(chosen, 1):
        with Image.open(Path(data_dir) / row["image_path"]) as opened:
            original = opened.convert("RGB")
        width, height = original.size
        canvas = Image.new("RGB", (width * 2, height + 32), "white")
        canvas.paste(original, (0, 32))
        canvas.paste(original, (width, 32))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"GT | {row['image_id']}", fill="green")
        draw.text((width + 8, 8), f"{name.upper()} prediction | TP={tp} FP={fp} FN={fn}", fill="red")
        for obj in _gt_boxes(row):
            x1, y1, x2, y2 = obj["bbox"]
            box = (x1 * width / 1000, y1 * height / 1000 + 32,
                   x2 * width / 1000, y2 * height / 1000 + 32)
            draw.rectangle(box, outline="green", width=3)
            draw.text((box[0], box[1]), obj["label"], fill="green")
        for obj in parsed:
            x1, y1, x2, y2 = obj["bbox"]
            box = (x1 * width / 1000 + width, y1 * height / 1000 + 32,
                   x2 * width / 1000 + width, y2 * height / 1000 + 32)
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), obj["label"], fill="red")
        path = target / f"{name}_{index:02d}_{row['image_id']}.png"
        canvas.save(path)
        saved.append(str(path))
    print("\n".join(saved))
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-data", "preflight", "predict", "train", "score", "best", "grpo", "results", "figures", "gt-figures"])
    parser.add_argument("--name", help="e0/e1/e2/e3/e4_data_0.1/e4_data_1.0/e5")
    parser.add_argument("--base-setup", choices=list(SETUPS))
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    args = parser.parse_args()
    if args.command == "check-data":
        check_data(args.data_dir)
    elif args.command == "gt-figures":
        draw_ground_truth(args.data_dir)
    elif args.command == "preflight":
        model_preflight(args.data_dir)
    elif args.command == "train":
        if not args.name or (args.name not in SETUPS and not args.base_setup):
            parser.error("train needs --name and, for E4, --base-setup")
        train_sft(args.name, base_setup=args.base_setup, fraction=args.fraction,
                  data_dir=args.data_dir, batch_size=args.batch_size,
                  grad_accum=args.grad_accum)
    elif args.command == "predict":
        if not args.name:
            parser.error("predict needs --name")
        ckpt = None if args.name == "e0" else Path("work") / (args.base_setup or args.name)
        predict(args.name, checkpoint=ckpt, split=args.split,
                data_dir=args.data_dir, batch_size=args.batch_size)
    elif args.command == "score":
        if not args.name:
            parser.error("score needs --name")
        meta_path = Path("work") / (args.base_setup or args.name) / "run.json"
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else None
        score(args.name, Path("work") / f"{args.name}_{args.split}.jsonl",
              metadata=metadata, split=args.split, data_dir=args.data_dir)
    elif args.command == "best":
        print(best_sft())
    elif args.command == "grpo":
        train_grpo(best_sft(), args.data_dir)
    elif args.command == "figures":
        draw_examples(args.name or best_sft(), data_dir=args.data_dir, split=args.split)
    else:
        show_results()


if __name__ == "__main__":
    main()
