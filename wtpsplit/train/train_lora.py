# -*- coding: utf-8 -*-
"""
Train LoRA adapters for SaT (wtpsplit) — single GPU, A100-friendly.

Changements clés vs script d'origine :
- **Pré-calcul des labels** (0/1 = token == '\n') dans le Dataset -> on supprime la
  logique coûteuse du collate d'origine.
- **FastCollator** ultra-lean (pad + attention_mask + to(device)).
- BF16 + AdamW torch fused + SDPA activés.
- torch.compile désactivé par défaut (TORCH_COMPILE=1 pour l'activer).
- Parse JSON tolérant aux clés inconnues (allow_extra_keys) pour éviter les ValueError.
"""

# ------- ENV avant tout import HF/Datasets -------
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_DATASETS_DISABLE_TF_IMPORT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE", "0")   # mets TORCH_COMPILE=1 pour activer torch.compile
# -------------------------------------------------

import copy
import json
import logging
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from glob import glob
from typing import List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import adapters
import datasets
from adapters import AdapterArguments
from adapters.models import MODEL_MIXIN_MAPPING
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin
from tokenizers import AddedToken
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

import wandb
from wtpsplit.models import SubwordXLMConfig, SubwordXLMForTokenClassification
from wtpsplit.train.adaptertrainer import AdapterTrainer
from wtpsplit.train.evaluate import evaluate_sentence
from wtpsplit.train.trainer import Trainer
from wtpsplit.train.utils import Model
from wtpsplit.utils import Constants, LabelArgs

logger = logging.getLogger(__name__)

# CUDA perf
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.benchmark = True

# Adapter shim pour le backbone XLM-R custom de SaT
MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin


# ------------------------- Collate rapide -------------------------
class FastCollator:
    """
    Collate minimaliste : pad input_ids + labels, construit attention_mask.
    Renvoie des tenseurs CPU. Le Trainer les déplace vers le device.
    """
    def __init__(self, pad_id: int = 0, debug: bool = False, move_to_device: bool = False, device: Optional[torch.device] = None):
        self.pad_id = int(pad_id)
        self.debug = debug
        self.move_to_device = bool(move_to_device)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, features):
        if self.debug:
            print(f"[FastCollator] got {len(features)} samples", flush=True)

        input_ids = [torch.as_tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels    = [torch.as_tensor(f["labels"],    dtype=torch.long) for f in features]

        max_len = max(x.size(0) for x in input_ids)

        def pad(stack, pad_val):
            out = torch.full((len(stack), max_len), pad_val, dtype=stack[0].dtype)  # CPU
            for i, t in enumerate(stack):
                n = t.size(0)
                out[i, :n] = t
            return out

        input_ids = pad(input_ids, self.pad_id)          # CPU
        labels    = pad(labels, 0)                        # CPU
        attention = (input_ids != self.pad_id).long()     # CPU

        batch = {
            "input_ids":      input_ids,
            "attention_mask": attention,
            "labels":         labels,
        }

        # En temps normal on NE déplace PAS ici (laisser le Trainer gérer).
        if self.move_to_device:
            for k in batch:
                batch[k] = batch[k].to(self.device, non_blocking=True)
        return batch


# ----------------------------- Args -----------------------------
@dataclass
class Args:
    model_name_or_path: str
    base_model: str = "xlm-roberta-base"
    shuffle: bool = True
    text_path: str = "data/all_data.pth"
    include_languages: Optional[List[str]] = None
    preprocessing_num_workers: int = 1
    block_size: int = 512
    overflow_size: int = 16
    eval_stride: int = 256
    loss_margin: float = 0.5
    pack_samples: bool = False
    one_sample_per_line: bool = False
    use_loss_weights: bool = False
    do_sentence_training: bool = True
    do_auxiliary_training: bool = False
    aux_training_weight: float = 1.0
    ignore_non_hyphen: bool = False
    non_punctuation_sample_ratio: float = None
    adapter_warmup_steps: int = 0
    adapter_lr_multiplier: float = 1.0
    text_column: str = "text"
    num_hidden_layers: int = 0

    # NEW
    use_subwords: bool = True
    freeze_classifier: bool = False
    clf_from_scratch: bool = False
    unfreeze_ln: bool = False
    do_process: bool = False
    meta_clf: bool = False
    wandb_project: str = "sentence"
    eval_every: int = 5
    skip_eval_loss: bool = False
    subsample: Optional[float] = None


# ----------------------------- Main -----------------------------
def main():
    parser = HfArgumentParser([Args, TrainingArguments, LabelArgs, AdapterArguments])

    # Lecture CLI / JSON (tolérante aux clés en trop)
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        try:
            (args, training_args, label_args, adapter_args) = parser.parse_json_file(sys.argv[1], allow_extra_keys=True)
            wandb_name = training_args.output_dir
        except TypeError:
            with open(sys.argv[1], "r", encoding="utf-8") as f:
                raw = json.load(f)
            (args, training_args, label_args, adapter_args) = parser.parse_dict(raw, allow_extra_keys=True)
            wandb_name = getattr(training_args, "output_dir", None)
    else:
        (args, training_args, label_args, adapter_args) = parser.parse_args_into_dataclasses()
        wandb_name = None

    # Single-GPU safe defaults
    training_args.local_rank = -1
    training_args.deepspeed = None
    training_args.ddp_backend = None

    # DataLoader (collate léger => on peut remettre pin_memory)
    training_args.dataloader_num_workers = 0
    training_args.dataloader_persistent_workers = False
    training_args.dataloader_pin_memory = True
    training_args.remove_unused_columns = False
    try:
        training_args.dataloader_drop_last = True
    except Exception:
        pass

    # A100 perf
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        training_args.fp16 = False
        training_args.bf16 = True
        training_args.optim = "adamw_torch_fused"
        if getattr(training_args, "evaluation_strategy", None) in (None, "no"):
            try:
                training_args.evaluation_strategy = "epoch"
            except Exception:
                pass

    # Logging & seed
    from wtpsplit.train.train import setup_logging  # import local pour respecter l'env
    setup_logging(training_args)
    set_seed(training_args.seed)

    # Log device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(
        f"Device: {dev.type} | n_gpu: {torch.cuda.device_count()} | distributed: {training_args.local_rank != -1}"
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.warning(f"GPU: {props.name} | VRAM (GB): {props.total_memory/1024**3:.1f}")

    # --------- Config / labels ---------
    # IMPORTANT : on reste sur la tête **binaire** du checkpoint (1 logit) sauf si l'utilisateur
    # demande explicitement l'entraînement auxiliaire.
    if args.num_hidden_layers:
        config = SubwordXLMConfig.from_pretrained(args.model_name_or_path, num_labels=1, num_hidden_layers=args.num_hidden_layers)
    else:
        config = SubwordXLMConfig.from_pretrained(args.model_name_or_path)  # conserve num_labels du checkpoint (1)

    if args.do_auxiliary_training or args.meta_clf:
        num_labels = Constants.AUX_OFFSET + (1 + len(Constants.PUNCTUATION_CHARS))
        config.num_labels = num_labels  # bascule vers tête multi-classe si demandé

    # ---------- Tokenizer & special token ----------
    logger.warning("[STEP] tokenizer init")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("\n")]})
    newline_id = tokenizer.convert_tokens_to_ids("\n")
    pad_id = tokenizer.pad_token_id or 0
    special_tokens_ids = set(tokenizer.all_special_ids)
    if newline_id in special_tokens_ids:
        special_tokens_ids.remove(newline_id)

    # ---------- Backbone + Adapters ----------
    logger.warning("[STEP] backbone init")
    backbone = SubwordXLMForTokenClassification.from_pretrained(
        args.model_name_or_path, config=copy.deepcopy(config), ignore_mismatched_sizes=True
    )
    backbone.config.base_model = args.base_model

    if num_added and num_added > 0:
        try:
            backbone.resize_token_embeddings(len(tokenizer))
        except AttributeError:
            backbone.base_model.resize_token_embeddings(len(tokenizer))

    logger.warning("[STEP] adapters.init + add/train_adapter(text)")
    orig_model_type = backbone.config.model_type
    backbone.config.model_type = "xlm-roberta"
    adapters.init(backbone)
    backbone.config.model_type = orig_model_type
    backbone.add_adapter("text", config=adapter_args.adapter_config, set_active=True, overwrite_ok=True)
    backbone.train_adapter("text")

    # SDPA (flash attention) si dispo
    try:
        backbone.config._attn_implementation = "sdpa"
    except Exception:
        pass

    # torch.compile optionnel (désactivé par défaut)
    use_compile = os.getenv("TORCH_COMPILE", "0") == "1"
    if use_compile and hasattr(torch, "compile"):
        logger.warning("[compile] enabled -> torch.compile(backbone)")
        try:
            backbone = torch.compile(backbone, fullgraph=False)
        except Exception as e:
            logger.warning(f"[compile] disabled (fallback): {e}")
    else:
        logger.warning("[compile] disabled")

    # Wrapper Model
    model = Model(
        backbone,
        loss_margin=args.loss_margin,
        use_loss_weights=args.use_loss_weights,
        do_sentence_training=args.do_sentence_training,
        do_auxiliary_training=args.do_auxiliary_training,
        aux_training_weight=args.aux_training_weight,
    ).to(dev)

    logger.warning(model.backbone.adapter_summary())
    if args.freeze_classifier:
        for n, p in model.backbone.named_parameters():
            if "classifier" in n:
                p.requires_grad = False
    if args.clf_from_scratch:
        model.backbone.classifier = torch.nn.Linear(model.backbone.config.hidden_size, config.num_labels).to(dev)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.warning(
            f"[GPU] après to(dev) | allocated={torch.cuda.memory_allocated()/2**30:.2f}GB "
            f"reserved={torch.cuda.memory_reserved()/2**30:.2f}GB"
        )

    # ---------- Data ----------
    with training_args.main_process_first():
        data = torch.load(args.text_path)
        data = dict(sorted(data.items()))

    if not args.include_languages:
        args.include_languages = list(data.keys())

    # W&B (optionnel)
    if "wandb" in (training_args.report_to or []) and getattr(training_args, "process_index", 0) == 0:
        wandb.init(name=wandb_name, project=args.wandb_project, group=wandb_name)
        wandb.config.update(args)
        wandb.config.update(training_args)
        wandb.config.update({"newline_id": newline_id, "pad_id": pad_id})
        wandb.config.update(adapter_args)
        for file in glob(os.path.join(os.path.dirname(__file__), "*.py")):
            wandb.save(os.path.abspath(file), policy="now")

    # ---------- helpers dataset ----------
    def prepare_dataset(
        data,
        num_workers=1,
        include_languages=None,
        dataset_name="ud",
        shuffle=False,
        split="train",
        subsample: Optional[float] = None,
        one_sample_per_line: bool = False,
    ):
        with training_args.main_process_first():
            for lang in include_languages:
                if split == "train":
                    dataset = data[lang]["sentence"][dataset_name]["meta"]["train_data"]
                elif split == "valid":
                    dataset = data[lang]["sentence"][dataset_name]["data"]
                else:
                    raise ValueError(f"Unknown split: {split}")
                if dataset is None:
                    return None

                if one_sample_per_line or isinstance(dataset[0], list):
                    processed = []
                    for chunk in dataset:
                        if "\n" in chunk:
                            raise ValueError("Newlines not supported at this stage.")
                        processed.append(
                            {
                                "lang": lang,
                                "ends_with_punctuation": chunk[-1].endswith(tuple(Constants.PUNCTUATION_CHARS)),
                                args.text_column: "\n".join(chunk),
                            }
                        )
                    dataset = datasets.Dataset.from_list(processed)
                else:
                    for sent in dataset:
                        if "\n" in sent:
                            raise ValueError("Newlines not supported at this stage.")
                    dataset = datasets.Dataset.from_list(
                        [
                            {
                                args.text_column: (sent + "\n") if sent and sent[-1] != "\n" else sent,
                                "lang": lang,
                                "ends_with_punctuation": sent.endswith(tuple(Constants.PUNCTUATION_CHARS)),
                            }
                            for sent in dataset
                        ]
                    )
            logger.warning(f"Loaded {len(dataset)} examples for {lang} {dataset_name} {split} dataset.")

        if shuffle:
            dataset = dataset.shuffle(seed=training_args.seed)

        if subsample is not None:
            old_len = len(dataset)
            if subsample >= 1.0:
              n = min(int(subsample), len(dataset))   # interprète comme nombre d’exemples
            elif 0.0 < subsample < 1.0:
              n = max(1, int(subsample * len(dataset)))  # interprète comme ratio
            else:
              n = len(dataset)
            dataset = dataset.select(range(n))
            logger.warning(f"Subsampled {len(dataset)} from {old_len}.")
        if args.ignore_non_hyphen:
            with training_args.main_process_first():
                dataset = dataset.filter(
                    lambda s: any(c in s[args.text_column] for c in label_args.hyphen_chars),
                    num_proc=args.preprocessing_num_workers,
                )
                logger.info(f"Filtered to {len(dataset)} examples (ignore_non_hyphen).")

        if args.non_punctuation_sample_ratio is not None:
            languages_without_punct = {
                lc for lc in Constants.LANGINFO.index if Constants.LANGINFO.loc[lc, "no_punctuation"]
            }

            def drop_some_non_punct(ex):
                include_idx = set(np.where([lc not in languages_without_punct for lc in ex["lang"]])[0])
                punct_idx = {i for i in np.where(ex["ends_with_punctuation"])[0] if i in include_idx}
                target_n_non_punct = int(
                    (len(punct_idx) * args.non_punctuation_sample_ratio) / (1 - args.non_punctuation_sample_ratio)
                )
                n_drop = (len(include_idx) - len(punct_idx)) - target_n_non_punct
                keep = [True] * len(ex["ends_with_punctuation"])
                if n_drop > 0:
                    drop = np.random.choice(list(include_idx - punct_idx), n_drop, replace=False)
                    for i in drop:
                        keep[i] = False
                return keep

            with training_args.main_process_first():
                dataset = dataset.filter(drop_some_non_punct, batched=True, batch_size=1_000_000, num_proc=num_workers)

        # --- tokenize ---
        def tokenize_texts(examples):
            toks = tokenizer(examples[args.text_column], verbose=False)
            # on enlève BOS/EOS pour éviter des '\n' multiples aux bords
            return {"input_ids": [ids[1:-1] for ids in toks["input_ids"]]}

        # --- group into blocks ---
        def group_texts(examples):
            all_blocks, all_block_lengths, all_langs = [], [], []

            def maybe_pad(text):
                if args.pack_samples:
                    pad = config.downsampling_rate - (len(text) % config.downsampling_rate)
                    if pad == config.downsampling_rate:
                        pad = 0
                    text += chr(0) * pad
                return text

            for current_lang in set(examples["lang"]):
                if not args.use_subwords:
                    lang_texts = [
                        maybe_pad(t) for t, lc in zip(examples["input_ids"], examples["lang"]) if lc == current_lang
                    ]
                else:
                    lang_sub = [sw for sw, lc in zip(examples["input_ids"], examples["lang"]) if lc == current_lang]
                    # retire spéciaux sauf '\n' dédié
                    lang_sub = [[tok for tok in sw if tok not in special_tokens_ids] for sw in lang_sub]

                if args.pack_samples:
                    raise NotImplementedError("pack_samples n'est pas supporté pour subwords ici")
                else:
                    if not args.use_subwords:
                        concatenated = "".join(lang_texts)
                        concat_ids = [i for i, t in enumerate(lang_texts) for _ in t]
                    else:
                        concatenated = [x for sub in lang_sub for x in sub]
                        concat_ids = [i for i, sw in enumerate(lang_sub) for _ in sw]

                    total_len = len(concatenated)
                    best_len = math.ceil(total_len / args.block_size) * args.block_size + args.overflow_size
                    while best_len > total_len:
                        best_len -= args.block_size
                    if best_len < args.block_size:
                        best_len = args.block_size + 1
                    if best_len < 0:
                        return {"input_ids": [], "block_lengths": [], "lang": []}

                    concatenated = concatenated[:best_len]
                    concat_ids = concat_ids[:best_len]

                    blocks = [
                        concatenated[i : i + args.block_size + args.overflow_size]
                        for i in range(0, best_len - args.block_size, args.block_size)
                    ]
                    block_ids = [
                        concat_ids[i : i + args.block_size + args.overflow_size]
                        for i in range(0, best_len - args.block_size, args.block_size)
                    ]

                all_blocks.extend(blocks)
                all_block_lengths.extend([list(Counter(ids).values()) for ids in block_ids])
                all_langs.extend([current_lang] * len(blocks))

            return {"input_ids": all_blocks, "block_lengths": all_block_lengths, "lang": all_langs}

        if args.use_subwords:
            with training_args.main_process_first():
                dataset = dataset.map(
                    tokenize_texts, batched=True, num_proc=num_workers, remove_columns=[args.text_column], desc="Tokenizing"
                )
        else:
            with training_args.main_process_first():
                dataset = dataset.rename_column(args.text_column, "input_ids")

        if not one_sample_per_line:
            with training_args.main_process_first():
                dataset = dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=num_workers,
                    remove_columns=["ends_with_punctuation"] if args.text_column == "text" else [],
                    desc="Grouping",
                )

        # ------- PRECOMPUTE LABELS : 0/1 (boundary = token '\n') -------
        def add_labels(examples):
            return {
                "labels": [[1 if t == newline_id else 0 for t in ids] for ids in examples["input_ids"]]
            }

        with training_args.main_process_first():
            dataset = dataset.map(add_labels, batched=True, num_proc=1, desc="Precomputing labels")

        return dataset

    # ---------- Loop ----------
    for lang in tqdm(list(data.keys()), desc="Language"):
        if lang not in args.include_languages:
            continue

        for dataset_name in data[lang]["sentence"].keys():
            print("RUNNING:", dataset_name, lang)

            one_sample_per_line = True if "short" in dataset_name else args.one_sample_per_line

            with training_args.main_process_first():
                valid_dataset = prepare_dataset(
                    data=data,
                    num_workers=1,
                    include_languages=[lang],
                    dataset_name=dataset_name,
                    shuffle=False,
                    split="valid",
                    one_sample_per_line=one_sample_per_line,
                )
                if valid_dataset is None:
                    logger.warning(f"Skipping {lang} {dataset_name} (missing valid).")
                    continue
                logger.warning(f"Valid ds for {lang} {dataset_name} has {len(valid_dataset)} examples.")

                train_dataset = prepare_dataset(
                    data=data,
                    num_workers=args.preprocessing_num_workers,
                    include_languages=[lang],
                    dataset_name=dataset_name,
                    shuffle=args.shuffle,
                    split="train",
                    subsample=args.subsample,
                    one_sample_per_line=one_sample_per_line,
                )
                if train_dataset is None:
                    logger.warning(f"Skipping {lang} {dataset_name} (missing train).")
                    continue
                logger.warning(f"Train ds for {lang} {dataset_name} has {len(train_dataset)} examples.")

            # Debug sample
            idx = random.choice(range(len(train_dataset)))
            sample = train_dataset[idx]
            logger.warning(f"Sample {idx} of train: {sample}.")
            try:
                logger.warning(AutoTokenizer.from_pretrained(args.base_model).decode(sample["input_ids"]))
            except Exception:
                pass

            # ----- metrics (texte concat) -----
            def compute_metrics(trainer):
                metrics = {}
                eval_data = data[lang]["sentence"][dataset_name]["data"]
                model_eval = trainer._wrap_model(trainer.model, training=False)

                with training_args.main_process_first():
                    if one_sample_per_line or isinstance(eval_data[0], list):
                        eval_data = [item for sublist in eval_data for item in sublist]
                    score, info = evaluate_sentence(
                        lang,
                        eval_data,
                        model_eval,
                        stride=args.eval_stride,
                        block_size=args.block_size,
                        batch_size=training_args.per_device_eval_batch_size,
                    )
                    metrics[f"{dataset_name}/{lang}/pr_auc"] = score
                    metrics[f"{dataset_name}/{lang}/f1"] = info["f1"]
                    metrics[f"{dataset_name}/{lang}/f1_best"] = info["f1_best"]
                    metrics[f"{dataset_name}/{lang}/threshold_best"] = info["threshold_best"]
                return metrics

            # Eval à l'époque si possible
            try:
                if getattr(training_args, "evaluation_strategy", None) in (None, "no"):
                    training_args.evaluation_strategy = "epoch"
            except Exception:
                pass

            # --------- Data collator (rapide) ----------
            data_collator = FastCollator(pad_id=pad_id, debug=(os.getenv("DEBUG_COLLATE", "0") == "1"))
            # Trainer (AdapterTrainer quand train_adapter=True)
            trainer_cls = AdapterTrainer if adapter_args.train_adapter else Trainer
            trainer = trainer_cls(
                model,
                training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                logging_prefix=f"{dataset_name}/{lang}/",
                skip_eval_loss=args.skip_eval_loss,
            )

            # Warm-up court (ramène les kernels à chaud)
            if torch.cuda.is_available():
                L = min(256, args.block_size)
                with torch.cuda.amp.autocast(enabled=training_args.bf16):
                    model.backbone.train()
                    fake_input_ids = torch.randint(5, 100, (2, L), device=dev, dtype=torch.long)
                    fake_attn = torch.ones_like(fake_input_ids)
                    fake_labels = torch.zeros((2, L), device=dev, dtype=torch.long)
                    out = model.backbone(
                        input_ids=fake_input_ids, attention_mask=fake_attn, labels=fake_labels
                    )
                    loss = getattr(out, "loss", None)
                    if loss is not None:
                        loss.backward()
                torch.cuda.synchronize()
                logger.warning("[WARMUP] 1 fwd+bwd synthétique OK")
                logger.warning(
                    f"[GPU] before train | alloc={torch.cuda.memory_allocated()/2**30:.2f}GB "
                    f"res={torch.cuda.memory_reserved()/2**30:.2f}GB"
                )

            # --------- Train ----------
            trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                logger.warning(
                    f"[GPU] after train  | alloc={torch.cuda.memory_allocated()/2**30:.2f}GB "
                    f"res={torch.cuda.memory_reserved()/2**30:.2f}GB"
                )

            logger.warning(f"Finished training for {lang} {dataset_name}.")

            # --------- Save ----------
            if getattr(training_args, "local_rank", -1) in (-1, 0):
                out_dir = os.path.join(training_args.output_dir, dataset_name, lang)
                os.makedirs(out_dir, exist_ok=True)
                save_model = copy.deepcopy(model.backbone).to("cpu")
                if adapter_args.train_adapter:
                    save_model.save_adapter(adapter_name="text", save_directory=out_dir, with_head=True)
                else:
                    save_model.save_pretrained(out_dir)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
