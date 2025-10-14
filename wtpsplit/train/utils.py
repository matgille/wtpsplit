# wtpsplit/train/utils.py

import logging
import os
import torch
import torch.nn as nn
from wtpsplit.utils import Constants

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(
        self,
        backbone,
        loss_margin=0.5,
        use_loss_weights=False,
        do_sentence_training=True,
        do_auxiliary_training=False,
        aux_training_weight=1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = self.backbone.config

        assert loss_margin <= 0.5

        self.loss_margin = loss_margin
        self.use_loss_weights = use_loss_weights
        self.do_sentence_training = do_sentence_training
        self.do_auxiliary_training = do_auxiliary_training
        self.aux_training_weight = aux_training_weight

    @property
    def device(self):
        return self.backbone.device

    def forward(
        self,
        input_ids,
        language_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        label_weights=None,
        **kwargs,
    ):
        # ---- PATCH: avaler le kwarg ajouté par Transformers.Trainer
        # (sinon SubwordXLMForTokenClassification.forward() plante)
        kwargs.pop("num_items_in_batch", None)
        # ------------------------------------------------------------

        # Masque de réduction pour le calcul de perte (tokens "réels" uniquement)
        if position_ids is not None:
            # XXX: 1 is pad token id pour XLM, 0 sinon
            if "xlm" in self.config.model_type:
                reduced_attention_mask = (input_ids != 1).to(torch.long)
            else:
                reduced_attention_mask = (input_ids != 0).to(torch.long)
        else:
            # fallback si jamais position_ids n'est pas fourni
            if attention_mask is not None:
                reduced_attention_mask = attention_mask.to(torch.long)
            else:
                # on devine via le pad id usuel (0)
                reduced_attention_mask = (input_ids != 0).to(torch.long)

        # Appel backbone : on transmet les kwargs restants (output_hidden_states, etc.)
        output = dict(
            self.backbone.forward(
                input_ids=input_ids,
                language_ids=language_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **kwargs,
            )
        )
        logits = output["logits"]

        if labels is not None:
            # --- Objectif principal (prédiction de saut de ligne) ---
            loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

            if self.do_sentence_training:
                # label smoothing centré sur 0.5 avec marge +/- loss_margin
                sentence_labels = (0.5 - self.loss_margin) + (labels == Constants.NEWLINE_INDEX + 1).to(
                    logits.dtype
                ).view(-1) * self.loss_margin * 2
                sentence_logits = logits[:, :, Constants.NEWLINE_INDEX].view(-1)

                main_loss = (
                    loss_fn(sentence_logits, sentence_labels)
                    * (label_weights.view(-1) if label_weights is not None and self.use_loss_weights else 1)
                    * reduced_attention_mask.view(-1)
                )

                # moyenne sur les tokens valides
                loss = main_loss.sum() / (reduced_attention_mask.sum().clamp_min(1))

            # --- Objectif auxiliaire (prédiction de ponctuation) ---
            if self.do_auxiliary_training:
                ce = nn.CrossEntropyLoss()
                # 0 = "no label", NEWLINE -> maské ; autres labels décalés par AUX_OFFSET
                aux_labels = torch.where(
                    (labels == 0) | (labels == Constants.NEWLINE_INDEX + 1),
                    0,
                    labels - Constants.AUX_OFFSET,
                )
                # mask des positions non valides
                aux_labels = torch.where(reduced_attention_mask == 1, aux_labels, ce.ignore_index)

                aux_logits = logits[:, :, Constants.AUX_OFFSET :].contiguous()
                aux_loss = ce(aux_logits.view(-1, self.config.num_labels - Constants.AUX_OFFSET), aux_labels.view(-1))

                loss = loss + self.aux_training_weight * aux_loss

            output["loss"] = loss

        return output


def cleanup_cache_files(datasets) -> int:
    """Clean up all cache files in the dataset cache directory, except those currently used by any of the provided datasets.

    Args:
        datasets (List[Dataset]): A list of dataset objects.

    Be careful when running this command that no other process is currently using other cache files.

    Returns:
        int: Number of removed files.
    """
    if not datasets:
        return 0

    # Collect all current cache files from the provided datasets
    current_cache_files = set()
    for dataset in datasets:
        dataset_cache_files = [os.path.abspath(cache_file["filename"]) for cache_file in dataset.cache_files]
        current_cache_files.update(dataset_cache_files)
    logger.warning(f"Found {len(current_cache_files)} cache files used by the provided datasets.")

    if not current_cache_files:
        return 0

    # Assuming all datasets have cache files in the same directory
    cache_directory = os.path.dirname(next(iter(current_cache_files)))

    files = os.listdir(cache_directory)
    files_to_remove = []
    for f_name in files:
        full_name = os.path.abspath(os.path.join(cache_directory, f_name))
        if f_name.startswith("cache-") and f_name.endswith(".arrow") and full_name not in current_cache_files:
            files_to_remove.append(full_name)

    for file_path in files_to_remove:
        logger.warning(f"Removing {file_path}")
        os.remove(file_path)

    return len(files_to_remove)
