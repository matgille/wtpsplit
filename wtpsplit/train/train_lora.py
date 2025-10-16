# -*- coding: utf-8 -*-
"""
SaT LoRA training (wtpsplit) — version "propre"
- Garde le pipeline qui fonctionne chez toi (datasets -> FastCollator -> AdapterTrainer).
- Éval par époque via evaluate_sentence (pas de trainer.evaluate()).
- Early stopping à la patience (sauvegarde du meilleur adapter "text").
- Éval finale détaillée: precision / recall / F1 par seuil + CSV.

Compatible Transformers 4.51.3 et AdapterTrainer de wtpsplit.
"""

# ------- ENV -------
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_DATASETS_DISABLE_TF_IMPORT", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE", "0")

# ------- imports -------
import copy, json, logging, math, random, sys
from collections import Counter
from dataclasses import dataclass
from glob import glob
from typing import List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import datasets

import adapters
from adapters import AdapterArguments
from adapters.models import MODEL_MIXIN_MAPPING
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin

from tokenizers import AddedToken
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments, TrainerCallback, set_seed

import wandb
from wtpsplit import SaT
from wtpsplit.models import SubwordXLMConfig, SubwordXLMForTokenClassification
from wtpsplit.train.adaptertrainer import AdapterTrainer
from wtpsplit.train.evaluate import evaluate_sentence
from wtpsplit.train.trainer import Trainer
from wtpsplit.train.utils import Model
from wtpsplit.train.train import setup_logging
from wtpsplit.utils import Constants, LabelArgs

logger = logging.getLogger(__name__)

# CUDA perf
torch.backends.cuda.matmul.allow_tf32 = True
cudnn.benchmark = True

# Adapter shim pour le backbone XLM-R custom de SaT
MODEL_MIXIN_MAPPING["SubwordXLMRobertaModel"] = BertModelAdaptersMixin


# ------------------------- Collate rapide -------------------------
class FastCollator:
    """Pad CPU + attention_mask; labels déjà pré-calculés dans le Dataset."""
    def __init__(self, pad_id: int = 0, debug: bool = False):
        self.pad_id = int(pad_id)
        self.debug = debug

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

        input_ids = pad(input_ids, self.pad_id)
        labels    = pad(labels, 0)
        attention = (input_ids != self.pad_id).long()

        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


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
    eval_every: int = 1         # eval à chaque époque
    skip_eval_loss: bool = False
    subsample: Optional[float] = None


# ------------------- Early stop + save-best callback -------------------
class EvalSaveAndEarlyStopCallback(TrainerCallback):
    """
    - Évalue via evaluate_sentence (PR_AUC, F1, F1_best, threshold_best)
    - Sauvegarde l’adapter 'text' si meilleure métrique
    - Arrête l’entraînement après 'patience' évals sans amélioration
    """
    def __init__(
        self,
        lang, dataset_name, eval_texts,
        stride, block_size, batch_size,
        out_dir, monitor="f1_best", patience=8, min_delta=1e-4,
        save_name="text", logger=None,
    ):
        self.lang = lang
        self.dataset_name = dataset_name
        self.eval_texts = eval_texts
        self.stride = stride
        self.block_size = block_size
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.save_name = save_name
        self.logger = logger or logging.getLogger(__name__)
        self.best_score = None
        self.best_epoch = -1
        self.no_improve = 0
        os.makedirs(self.out_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Éval rapide via evaluate_sentence
        model.eval()
        with torch.inference_mode():
            pr_auc, info = evaluate_sentence(
                self.lang,
                self.eval_texts,
                model,
                stride=self.stride,
                block_size=self.block_size,
                batch_size=self.batch_size,
            )

        self.logger.warning(
            f"[EVAL] epoch={int(state.epoch)} | {self.dataset_name}/{self.lang} "
            f"| PR_AUC={pr_auc:.4f} | F1={info['f1']:.4f} | "
            f"F1_best={info['f1_best']:.4f} @thr={info['threshold_best']:.4f}"
        )
        watched = {"pr_auc": pr_auc, "f1": info["f1"], "f1_best": info["f1_best"]}[self.monitor]
        improved = (self.best_score is None) or (watched > self.best_score + self.min_delta)

        if improved:
            self.best_score = float(watched)
            self.best_epoch = int(state.epoch)
            self.no_improve = 0
            # Sauvegarde du meilleur adapter
            try:
                model.backbone.save_adapter(adapter_name=self.save_name, save_directory=self.out_dir, with_head=True)
            except Exception:
                model.save_adapter(adapter_name=self.save_name, save_directory=self.out_dir, with_head=True)
            self.logger.warning(
                f"[BEST] new best @{self.best_epoch} ({self.monitor}={self.best_score:.4f}) -> saved to {self.out_dir}"
            )
        else:
            self.no_improve += 1
            self.logger.warning(f"[EARLY-STOP] no improvement {self.no_improve}/{self.patience}")
            if self.no_improve >= self.patience:
                control.should_training_stop = True
                self.logger.warning(
                    f"[EARLY-STOP] patience reached ({self.patience}) -> stop at epoch {int(state.epoch)}; "
                    f"best @{self.best_epoch} ({self.monitor}={self.best_score:.4f})"
                )
        return control


# ----------- métriques classiques (Precision/Recall/F1) à la fin ----------
def _gold_boundaries_from_sentences(sents: List[str]) -> List[int]:
    """Positions char des frontières or (juste après chaque phrase), avec '\n' entre phrases."""
    pos, out = 0, []
    for i, s in enumerate(sents):
        pos += len(s)
        if i < len(sents) - 1:   # pas de frontière après la dernière
            out.append(pos)      # frontière = fin de phrase
            pos += 1             # compte le '\n' inséré entre les phrases
    return out

def _pred_boundaries_from_segments(segments: List[str]) -> List[int]:
    """Positions char cumulées après chaque segment (concaténation exacte des segments)."""
    pos, out = 0, []
    for i, seg in enumerate(segments):
        pos += len(seg)
        if i < len(segments) - 1:
            out.append(pos)
    return out

def _match_tp_fp_fn(gold: List[int], pred: List[int], tol: int = 1):
    """Aligne deux listes triées de frontières; match si |g-p| <= tol."""
    i = j = tp = 0
    while i < len(gold) and j < len(pred):
        if abs(gold[i] - pred[j]) <= tol:
            tp += 1; i += 1; j += 1
        elif pred[j] < gold[i]:
            j += 1  # FP
        else:
            i += 1  # FN
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn

def final_precision_recall_f1_table(backbone_name: str, adapter_dir: str, eval_texts: List[str],
                                    thresholds: List[float], csv_path: str):
    """Charge un SaT(backbone)+adapter puis calcule P/R/F1 par seuil; sauvegarde un CSV."""
    # Long texte d'éval
    long_text = "\n".join(eval_texts)
    gold = _gold_boundaries_from_sentences(eval_texts)
    def _sat_backbone_id(name: str) -> str:
        if name.startswith("segment-any-text/"):
            return name.split("/", 1)[1]
        return name
    # Charge wrapper SaT + adapter best
    bb_for_sat = _sat_backbone_id(backbone_name)
    sat = SaT(bb_for_sat, lora_path=adapter_dir)

    m = sat.model.backbone if hasattr(sat, "model") and hasattr(sat.model, "backbone") else sat.model

# Active l’adapter chargé (SaT le nomme généralement 'sat-lora').
    adapter_names = []
    try:
        adapter_names = list(m.adapters_config.adapters.keys())
    except Exception:
        pass

    name_to_use = "sat-lora" if "sat-lora" in adapter_names else (
        "text" if "text" in adapter_names else (adapter_names[0] if adapter_names else None)
    )

    if name_to_use:
        m.set_active_adapters(name_to_use)
        logger.warning(f"[FINAL] activated adapter: {name_to_use}")
    else:
        logger.warning("[FINAL] WARNING: no adapter active; using base model")
        m = sat.model.backbone if hasattr(sat, "model") and hasattr(sat.model, "backbone") else sat.model
    try:
        m.set_active_adapters("text")
    except Exception:
        pass
    m.eval()

    rows = []
    best = {"f1": -1.0, "thr": None, "p": 0.0, "r": 0.0, "tp": 0, "fp": 0, "fn": 0}
    for thr in thresholds:
        with torch.inference_mode():
            segs = sat.split(long_text, threshold=float(thr))
        pred = _pred_boundaries_from_segments(segs)
        tp, fp, fn = _match_tp_fp_fn(gold, pred, tol=1)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        rows.append((thr, p, r, f1, tp, fp, fn))
        if f1 > best["f1"]:
            best = {"f1": f1, "thr": thr, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn}

    # print court + CSV
    logger.warning("[FINAL CLASSIC METRICS] (thr, P, R, F1, TP, FP, FN):")
    for thr, p, r, f1, tp, fp, fn in rows:
        logger.warning(f"  thr={thr:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (TP={tp}, FP={fp}, FN={fn})")
    logger.warning(f"[FINAL BEST] F1={best['f1']:.4f} @ thr={best['thr']:.2f} | P={best['p']:.3f} R={best['r']:.3f}")

    # CSV
    try:
        import csv
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
            for row in rows:
                w.writerow(row)
    except Exception as e:
        logger.warning(f"[FINAL CLASSIC METRICS] CSV save skipped: {e}")

    return rows, best


# ----------------------------- Main -----------------------------
def main():
    parser = HfArgumentParser([Args, TrainingArguments, LabelArgs, AdapterArguments])

    # --- Parse JSON (et retire les clés HF qui te posaient problème) ---
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Paramètres custom d’early stop (facultatifs dans le JSON)
        early_patience   = int(raw.pop("early_stop_patience", 8))
        early_min_delta  = float(raw.pop("early_stop_min_delta", 1e-4))
        early_monitor    = str(raw.pop("early_stop_monitor", "f1_best"))

        # On NE passe pas ces clés à TrainingArguments (4.51.3 a râlé chez toi)
        for k in ["evaluation_strategy", "save_strategy", "load_best_model_at_end",
                  "metric_for_best_model", "greater_is_better"]:
            raw.pop(k, None)

        # TrainingArguments accepte ce qui suit; le reste va à Args/LabelArgs/AdapterArgs
        (args, training_args, label_args, adapter_args) = parser.parse_dict(raw, allow_extra_keys=True)
        wandb_name = getattr(training_args, "output_dir", None)
    else:
        (args, training_args, label_args, adapter_args) = parser.parse_args_into_dataclasses()
        wandb_name = None
        early_patience, early_min_delta, early_monitor = 8, 1e-4, "f1_best"

    # Single-GPU safe defaults
    training_args.local_rank = -1
    training_args.deepspeed = None
    training_args.ddp_backend = None
    training_args.dataloader_num_workers = 0
    training_args.dataloader_persistent_workers = False
    training_args.dataloader_pin_memory = True
    training_args.remove_unused_columns = False
    try:
        training_args.dataloader_drop_last = True
    except Exception:
        pass

    # Logging & seed
    setup_logging(training_args)
    set_seed(training_args.seed)

    # Log device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.warning(f"Device: {dev.type} | n_gpu: {torch.cuda.device_count()} | distributed: {training_args.local_rank != -1}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.warning(f"GPU: {props.name} | VRAM (GB): {props.total_memory/1024**3:.1f}")

    # --------- Config / labels ---------
    if args.num_hidden_layers:
        config = SubwordXLMConfig.from_pretrained(args.model_name_or_path, num_labels=1, num_hidden_layers=args.num_hidden_layers)
    else:
        config = SubwordXLMConfig.from_pretrained(args.model_name_or_path)

    if args.do_auxiliary_training or args.meta_clf:
        num_labels = Constants.AUX_OFFSET + (1 + len(Constants.PUNCTUATION_CHARS))
        config.num_labels = num_labels

    # ---------- Tokenizer ----------
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

    # SDPA (si dispo)
    try:
        backbone.config._attn_implementation = "sdpa"
    except Exception:
        pass

    # torch.compile optionnel
    if os.getenv("TORCH_COMPILE", "0") == "1" and hasattr(torch, "compile"):
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
            if "classifier" in n: p.requires_grad = False
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
        wandb.config.update(args); wandb.config.update(training_args); wandb.config.update(adapter_args)
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
                        if "\n" in chunk: raise ValueError("Newlines not supported at this stage.")
                        processed.append({"lang": lang, "ends_with_punctuation": chunk[-1].endswith(tuple(Constants.PUNCTUATION_CHARS)), args.text_column: "\n".join(chunk)})
                    dataset = datasets.Dataset.from_list(processed)
                else:
                    for sent in dataset:
                        if "\n" in sent: raise ValueError("Newlines not supported at this stage.")
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
            n = (min(int(subsample), len(dataset)) if subsample >= 1.0
                 else max(1, int(subsample * len(dataset))) if subsample > 0 else len(dataset))
            dataset = dataset.select(range(n))
            logger.warning(f"Subsampled {len(dataset)} from {old_len}.")

        if args.ignore_non_hyphen:
            with training_args.main_process_first():
                dataset = dataset.filter(lambda s: any(c in s[args.text_column] for c in label_args.hyphen_chars), num_proc=args.preprocessing_num_workers)
                logger.info(f"Filtered to {len(dataset)} examples (ignore_non_hyphen).")

        if args.non_punctuation_sample_ratio is not None:
            languages_without_punct = {lc for lc in Constants.LANGINFO.index if Constants.LANGINFO.loc[lc, "no_punctuation"]}
            def drop_some_non_punct(ex):
                include_idx = set(np.where([lc not in languages_without_punct for lc in ex["lang"]])[0])
                punct_idx = {i for i in np.where(ex["ends_with_punctuation"])[0] if i in include_idx}
                target_n_non_punct = int((len(punct_idx) * args.non_punctuation_sample_ratio) / (1 - args.non_punctuation_sample_ratio))
                n_drop = (len(include_idx) - len(punct_idx)) - target_n_non_punct
                keep = [True] * len(ex["ends_with_punctuation"])
                if n_drop > 0:
                    drop = np.random.choice(list(include_idx - punct_idx), n_drop, replace=False)
                    for i in drop: keep[i] = False
                return keep
            with training_args.main_process_first():
                dataset = dataset.filter(drop_some_non_punct, batched=True, batch_size=1_000_000, num_proc=num_workers)

        # --- tokenize ---
        def tokenize_texts(examples):
            toks = tokenizer(examples[args.text_column], verbose=False)
            return {"input_ids": [ids[1:-1] for ids in toks["input_ids"]]}  # enlève BOS/EOS

        # --- group into blocks ---
        def group_texts(examples):
            all_blocks, all_block_lengths, all_langs = [], [], []

            def maybe_pad(text):
                if args.pack_samples:
                    pad = config.downsampling_rate - (len(text) % config.downsampling_rate)
                    if pad == config.downsampling_rate: pad = 0
                    text += chr(0) * pad
                return text

            for current_lang in set(examples["lang"]):
                if not args.use_subwords:
                    lang_texts = [maybe_pad(t) for t, lc in zip(examples["input_ids"], examples["lang"]) if lc == current_lang]
                else:
                    lang_sub = [sw for sw, lc in zip(examples["input_ids"], examples["lang"]) if lc == current_lang]
                    lang_sub = [[tok for tok in sw if tok not in special_tokens_ids] for sw in lang_sub]

                if args.pack_samples:
                    raise NotImplementedError("pack_samples non supporté ici")
                else:
                    if not args.use_subwords:
                        concatenated = "".join(lang_texts)
                        concat_ids = [i for i, t in enumerate(lang_texts) for _ in t]
                    else:
                        concatenated = [x for sub in lang_sub for x in sub]
                        concat_ids = [i for i, sw in enumerate(lang_sub) for _ in sw]

                    total_len = len(concatenated)
                    best_len = math.ceil(total_len / args.block_size) * args.block_size + args.overflow_size
                    while best_len > total_len: best_len -= args.block_size
                    if best_len < args.block_size: best_len = args.block_size + 1
                    if best_len < 0: return {"input_ids": [], "block_lengths": [], "lang": []}

                    concatenated = concatenated[:best_len]; concat_ids = concat_ids[:best_len]
                    blocks = [concatenated[i:i + args.block_size + args.overflow_size] for i in range(0, best_len - args.block_size, args.block_size)]
                    block_ids = [concat_ids[i:i + args.block_size + args.overflow_size] for i in range(0, best_len - args.block_size, args.block_size)]

                all_blocks.extend(blocks)
                all_block_lengths.extend([list(Counter(ids).values()) for ids in block_ids])
                all_langs.extend([current_lang] * len(blocks))

            return {"input_ids": all_blocks, "block_lengths": all_block_lengths, "lang": all_langs}

        if args.use_subwords:
            with training_args.main_process_first():
                dataset = dataset.map(tokenize_texts, batched=True, num_proc=num_workers, remove_columns=[args.text_column], desc="Tokenizing")
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

        # Labels 0/1 (boundary = token '\n')
        def add_labels(examples):
            return {"labels": [[1 if t == newline_id else 0 for t in ids] for ids in examples["input_ids"]]}
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
                valid_dataset = prepare_dataset(data, 1, [lang], dataset_name, False, "valid", None, one_sample_per_line)
                if valid_dataset is None:
                    logger.warning(f"Skipping {lang} {dataset_name} (missing valid)."); continue
                logger.warning(f"Valid ds for {lang} {dataset_name} has {len(valid_dataset)} examples.")

                train_dataset  = prepare_dataset(data, args.preprocessing_num_workers, [lang], dataset_name, args.shuffle, "train", args.subsample, one_sample_per_line)
                if train_dataset is None:
                    logger.warning(f"Skipping {lang} {dataset_name} (missing train)."); continue
                logger.warning(f"Train ds for {lang} {dataset_name} has {len(train_dataset)} examples.")

            # Debug
            idx = random.choice(range(len(train_dataset))); sample = train_dataset[idx]
            logger.warning(f"Sample {idx} of train: {sample}.")
            try:
                logger.warning(AutoTokenizer.from_pretrained(args.base_model).decode(sample["input_ids"]))
            except Exception:
                pass

            # Data collator
            data_collator = FastCollator(pad_id=pad_id, debug=(os.getenv("DEBUG_COLLATE", "0") == "1"))

            # Trainer
            trainer_cls = AdapterTrainer if adapter_args.train_adapter else Trainer
            trainer = trainer_cls(
                model,
                training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,   # juste pour la boucle; on n'appelle pas trainer.evaluate()
                compute_metrics=None,
                data_collator=data_collator,
                logging_prefix=f"{dataset_name}/{lang}/",
                skip_eval_loss=args.skip_eval_loss,
            )

            # Texte d’éval à plat
            eval_data = data[lang]["sentence"][dataset_name]["data"]
            if isinstance(eval_data[0], list):
                eval_data = [s for ch in eval_data for s in ch]

            # Early stop + save-best
            best_dir = os.path.join(training_args.output_dir, dataset_name, lang)
            trainer.add_callback(EvalSaveAndEarlyStopCallback(
                lang=lang,
                dataset_name=dataset_name,
                eval_texts=eval_data,
                stride=args.eval_stride,
                block_size=args.block_size,
                batch_size=training_args.per_device_eval_batch_size,
                out_dir=best_dir,
                monitor=early_monitor,     # "f1_best" par défaut
                patience=early_patience,   # p.ex. 8
                min_delta=early_min_delta, # p.ex. 1e-4
                save_name="text",
                logger=logger,
            ))

            # Warmup rapide (kernels à chaud)
            if torch.cuda.is_available():
                L = min(256, args.block_size)
                with torch.cuda.amp.autocast(enabled=training_args.bf16):
                    model.backbone.train()
                    fake_input_ids = torch.randint(5, 100, (2, L), device=dev, dtype=torch.long)
                    fake_attn = torch.ones_like(fake_input_ids)
                    fake_labels = torch.zeros((2, L), device=dev, dtype=torch.long)
                    out = model.backbone(input_ids=fake_input_ids, attention_mask=fake_attn, labels=fake_labels)
                    loss = getattr(out, "loss", None)
                    if loss is not None: loss.backward()
                torch.cuda.synchronize()
                logger.warning("[WARMUP] 1 fwd+bwd synthétique OK")
                logger.warning(f"[GPU] before train | alloc={torch.cuda.memory_allocated()/2**30:.2f}GB res={torch.cuda.memory_reserved()/2**30:.2f}GB")

            # Train
            trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))

            # Recharge explicitement le meilleur adapter pour la suite
            try:
                bb = getattr(trainer.model, "backbone", trainer.model)
                bb.load_adapter(best_dir, load_as="text", with_head=True)
                bb.set_active_adapters("text")
                logger.warning(f"[BEST-LOAD] Adapter 'text' rechargé depuis {best_dir}")
            except Exception as e:
                logger.warning(f"[BEST-LOAD] skip reload ({e})")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                logger.warning(f"[GPU] after train  | alloc={torch.cuda.memory_allocated()/2**30:.2f}GB res={torch.cuda.memory_reserved()/2**30:.2f}GB")

            logger.warning(f"Finished training for {lang} {dataset_name}.")

            # --------- Save “last” (en plus du best) ----------
            if getattr(training_args, "local_rank", -1) in (-1, 0):
                out_last = os.path.join(training_args.output_dir, dataset_name, lang, "last")
                os.makedirs(out_last, exist_ok=True)
                save_model = copy.deepcopy(model.backbone).to("cpu")
                if adapter_args.train_adapter:
                    save_model.save_adapter(adapter_name="text", save_directory=out_last, with_head=True)
                else:
                    save_model.save_pretrained(out_last)

            # --------- ÉVAL FINALE CLASSIQUE (P/R/F1 par seuil) ----------
            thresholds = [round(x, 2) for x in np.linspace(0.01, 0.50, 50)]
            csv_path   = os.path.join(best_dir, "final_metrics.csv")
            
            final_precision_recall_f1_table(
                backbone_name=args.model_name_or_path,
                adapter_dir=best_dir,
                eval_texts=eval_data,
                thresholds=thresholds,
                csv_path=csv_path,
            )


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
