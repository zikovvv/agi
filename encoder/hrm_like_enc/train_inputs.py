import atexit
from enum import Enum
import functools
import json
from fastapi import FastAPI
from pydantic import BaseModel
import tqdm
import os
import wandb
import time
from dotenv import load_dotenv
from encoder.hrm_like_enc.config import DatasetConfig, TrainConfig, ModelConfig            
from encoder.hrm_like_enc.hrm_lucidrains import HRM
from encoder.hrm_like_enc.inference import augment_colors_batch, augmented_inference_batched, augmented_inference_batched_with_voting
from gen_simple_arc_ds import PuzzleNames
from get_dataloader_for_model_for_task import get_dataloaders_for_flat_seq_cls, get_dataloaders_for_encoder_masked_modeling
from get_ds_for_task import get_arc_puzzle_ds_as_flat_ds, get_ds_1d_seq_for_random_input_with_some_transformation_for_output, get_custom_ds_arc, get_ds_for_masked_modeling_only_answer, get_ds_for_masked_modeling_only_answer_only_foreground_items
from model import *
import math
from dataclasses import dataclass
import sys
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.functional import F # type: ignore
from common import *
from datasets import Dataset
import torch
from torch import nn
import matplotlib.pyplot as plt
from show import plot_batch
import mlflow.pytorch as mlpt

logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
# logger.add(sys.stderr, level="DEBUG") 
logger.add(sys.stderr, level="INFO") 

load_dotenv()


class ManualState(BaseModel):
    lr : float
    t_show_nb_b : int
    v_show_nb_b : int
    stop_pretraining : bool
    skip_train_epoch : bool


def create_command_server(
    tcfg : TrainConfig,
    mcfg : ModelConfig,
    dcfg : DatasetConfig,
) -> ManualState :
    app = FastAPI()
    state = ManualState(
        lr=tcfg.lr,
        t_show_nb_b=tcfg.t_show_nb_b,
        v_show_nb_b=tcfg.v_show_nb_b,
        stop_pretraining=False,
        skip_train_epoch=False
    )

    @app.post("/set_lr")
    async def set_lr(lr: float):
        state.lr = lr
        return 

    @app.post("/visualize_train_batch")
    async def visualize_train_batch(nb_batches: int = 1):
        state.t_show_nb_b = nb_batches
    
    @app.post("/visualize_val_batch")
    async def visualize_val_batch(nb_batches: int = 1):
        state.v_show_nb_b = nb_batches
    
    @app.post("/stop_pretraining")
    async def stop_pretraining():
        state.stop_pretraining = True
    
    @app.post("/skip_train_epoch")
    async def skip_train_epoch():
        state.skip_train_epoch = True
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "state": state}

    # run fastapi app in a separate thread
    import threading
    def run_app():
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8500)

    threading.Thread(target=run_app, daemon=True).start()

    return state

# Extra stability tips

# Temperature anneal: start soft (τ≈1.0–2.0), linearly go to 0.1–0.2.

# Entropy schedule on the input distribution:

# Early: encourage entropy (add -λ·H, λ small like 1e-3).

# Late: penalize entropy (add +λ·H) to push toward one-hots.

# Replay buffer: occasionally train CE on a mix of (a) current optimized inputs and (b) a few recent versions to reduce drift.

# Mask discipline: never let optimized inputs modify immutable/observed cells; enforce with train_mask.

class ModelEncoderBase(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """inputs: [B, ..., D] => outputs: [B, ..., D] returns the same shape as inputs"""
        raise NotImplementedError()

class CNNEncoder2d(ModelEncoderBase):
    def __init__(
        self,
        h, w, mcfg : ModelConfig
    ):
        super().__init__()
        self.cfg = mcfg
        self.h = h
        self.w = w
        self.dim = mcfg.d_model
        self.cfg = mcfg
        self.cnn = nn.Sequential(
            nn.Conv2d(self.dim, self.dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.dim * 2, self.dim * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(self.dim * 2, self.dim, kernel_size=7, padding=3),
            nn.ReLU(),
        )

    def forward(
        self,
        inputs: torch.Tensor, # [B, H*W, D] 
    ) -> torch.Tensor: # [B, H*W, D]
        assert inputs.dim() == 4, f"inputs must be 4 dims [B, H, W, D], got {inputs.shape}"
        B = inputs.shape[0]
        inputs = inputs.reshape(-1, self.h, self.w, self.dim).permute(0, 3, 1, 2) # [B, D, H, W]
        assert inputs.shape == (B, self.dim, self.h, self.w), f"inputs must be [B, {self.dim}, {self.h}, {self.w}], got {inputs.shape}"
        assert inputs.shape[1] == self.h and inputs.shape[2] == self.w, f"inputs must be [B, {self.h}, {self.w}, V], got {inputs.shape}"
        hiddens : torch.Tensor = self.cnn(inputs) # [B, D, H, W]
        hiddens = hiddens.permute(0, 2, 3, 1).reshape(-1, self.h * self.w, self.dim) # [B, H*W, D]
        assert hiddens.shape == (B, self.h * self.w, self.dim), f"hiddens must be [B, {self.h * self.w}, {self.dim}], got {hiddens.shape}"
        return hiddens




class CNNEncoder1d(ModelEncoderBase):
    def __init__(self, mcfg: ModelConfig):
        super().__init__()
        self.cfg = mcfg
        self.dim = self.cfg.d_model
        activation = nn.ReLU()
        self.cnn = nn.Sequential(
            nn.Conv1d(self.dim, self.dim * 2, kernel_size=3, padding=1),
            activation,
            nn.Conv1d(self.dim * 2, self.dim * 2, kernel_size=5, padding=2),
            activation,
            nn.Conv1d(self.dim * 2, self.dim, kernel_size=7, padding=3),
            activation,
        )

    def forward(
        self,
        inputs: torch.Tensor,  # [B, L, D]
    ) -> torch.Tensor:
        assert inputs.dim() == 3, f"inputs must be 3 dims [B, L, D], got {inputs.shape}"
        B = inputs.shape[0]
        L = inputs.shape[1]
        inputs = inputs.permute(0, 2, 1)
        assert inputs.shape == (B, self.dim, L), f"inputs must be [B, {self.dim}, {L}], got {inputs.shape}"
        hiddens: torch.Tensor = self.cnn(inputs)  # [B, D
        hiddens = hiddens.permute(0, 2, 1)
        assert hiddens.shape == (B, L, self.dim), f"hiddens must be [B, {L}, {self.dim}], got {hiddens.shape}"
        return hiddens
    


class ModelClassifier(nn.Module):
    @dataclass
    class Res:
        hiddens: torch.Tensor
        logits: torch.Tensor
        preds: torch.Tensor
        loss: Optional[torch.Tensor] = None
        nb_corr: int = 0
        nb_total: int = 0
    
    def __init__(
        self,
        model_encoder : ModelEncoderBase,
        cfg : ModelConfig,
        nb_cls: int
    ):
        super().__init__()
        self.cfg = cfg
        self.model_encoder : ModelEncoderBase = model_encoder
        self.nb_cls = nb_cls
        self.cls_head = nn.Linear(self.cfg.d_model, nb_cls)
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)

    def forward(
        self,
        inputs: torch.Tensor, # [B, ..., D]
        is_ids: bool, # if true, inputs are ids, else onehots
        labels: Optional[torch.Tensor], # [B, ...]
    ) -> Res:
        if is_ids : # [B, ...]
            embs : torch.Tensor = self.embeddings(inputs)
            inputs = embs
        assert inputs.dim() >= 3, f"inputs must be at least 3 dims [B, ..., D], got {inputs.shape}"
        B = inputs.shape[0]
        D = self.cfg.d_model
        hiddens : torch.Tensor = self.model_encoder(inputs = inputs) # [B, ..., D]
        assert hiddens.shape[0] == B and hiddens.shape[-1] == D, hiddens.shape
        logits: torch.Tensor = self.cls_head(hiddens)  # [B, ..., vocab_size]
        assert logits.shape[0] == B and logits.shape[-1] == self.nb_cls, logits.shape
        preds = logits.argmax(dim=-1)
        if labels is None : 
            return self.Res(
                hiddens=hiddens,
                logits=logits,
                preds=preds,
                loss=None,
                nb_corr=0,
                nb_total=0
            )
            
        loss = F.cross_entropy( 
            logits.view(-1, self.nb_cls),
            labels.view(-1),
            ignore_index=self.cfg.ignore_id
        )
        mask_cls = labels != self.cfg.ignore_id # [B, ...]
        nb_cor = (preds[mask_cls] == labels[mask_cls]).sum().item()
        nb_total = mask_cls.sum().item()
        return self.Res(
            hiddens=hiddens,
            logits=logits,
            preds=preds,
            loss=loss,
            nb_corr=int(nb_cor),
            nb_total=int(nb_total)
        )


class ModelClsCritic(nn.Module):
    @dataclass
    class Res:
        cls_logits: torch.Tensor
        cls_preds: torch.Tensor
        nb_cls: int = 0
        nb_cor_gt: int = 0
        nb_tp: int = 0
        nb_fp: int = 0
        nb_fn: int = 0
        nb_tn: int = 0
        cls_labels: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None

    def __init__(
        self,
        classifier : ModelClassifier,
        mcfg : ModelConfig
    ):
        super().__init__()
        self.cfg = mcfg
        self.dim = self.cfg.d_model
        self.embedding = nn.Embedding(mcfg.vocab_size, mcfg.d_model)
        self.model_classifier : ModelClassifier = classifier
        self.cfg = mcfg

    def get_inputs(self, inputs: torch.Tensor, is_ids: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if is_ids : # [B, ...]
            embs : torch.Tensor = self.embedding(inputs)# [B, ..., D]
            color_input_ids = inputs.clone().detach()
        else: # [B, ..., V]
            V_in = inputs.shape[-1]
            assert V_in <= self.cfg.vocab_size
            embs_weights = self.embedding.weight[:V_in, ...]  # [V, D]
            assert embs_weights.shape == (V_in, self.dim)
            inputs = inputs.softmax(dim=-1) # [B, ..., V]
            color_input_ids = inputs.argmax(dim=-1).detach() # [B, ...]
            embs : torch.Tensor = inputs @ embs_weights  # [B, ..., D]
        return embs, color_input_ids

    def forward(
        self,
        inputs: torch.Tensor,
        is_ids: bool, # if true, inputs are ids, else onehots
        color_labels: Optional[torch.Tensor] # color labels
    ) -> Res:                
        B, *SHAPE, _ = inputs.shape
        D = self.cfg.d_model
        embs, color_input_ids = self.get_inputs(inputs, is_ids = is_ids) # [B, ..., D], [B, ...]
        assert color_input_ids.shape == (B, *SHAPE), f'{SHAPE = }, {color_input_ids.shape = }'
        assert embs.shape == (B, *SHAPE, D), f'{SHAPE = }, {embs.shape = }'
        cls_labels : Optional[torch.Tensor] = None
        if color_labels is not None :
            cls_labels = (color_labels == color_input_ids).long()  # [B, ...], 1 if correct, 0 if wrong, -100 if ignored
            cls_labels[color_labels == self.cfg.ignore_id] = self.cfg.ignore_id
            assert cls_labels.shape == (B, *SHAPE)
        res_cls : ModelClassifier.Res = self.model_classifier(
            embs.reshape(B, -1, D),
            is_ids=False,
            labels=cls_labels.reshape(B, -1) if cls_labels is not None else None
        )
        preds = res_cls.preds.reshape(B, *SHAPE)  # [B, ...]

        mask_classifiable = color_labels != self.cfg.ignore_id if color_labels is not None else torch.ones_like(preds).bool() # mask classifiable does not need to equal mask trainable
        
        # assert (color_labels != self.cfg.ignore_id).shape == (B, *SHAPE)# == color_labels.shape == preds.shape == cls_labels.shape
        # log(f'{mask_classifiable.shape = }')
        # log(f'{mask_classifiable = }')
        # log(f'{color_labels = }')
        nb_actually_right_cells_in_input = (cls_labels[mask_classifiable] == 1).sum().item() if cls_labels is not None else 0
        nb_tp = ((preds[mask_classifiable] == 1) & (cls_labels[mask_classifiable] == 1)).sum().item() if cls_labels is not None else 0
        nb_fp = ((preds[mask_classifiable] == 1) & (cls_labels[mask_classifiable] == 0)).sum().item() if cls_labels is not None else 0
        nb_fn = ((preds[mask_classifiable] == 0) & (cls_labels[mask_classifiable] == 1)).sum().item() if cls_labels is not None else 0
        nb_tn = ((preds[mask_classifiable] == 0) & (cls_labels[mask_classifiable] == 0)).sum().item() if cls_labels is not None else 0 
        nb_classifiable = mask_classifiable.sum().item() if color_labels is not None else 0
        # log(f'{nb_actually_right_cells_in_input = }, {nb_classified_right = }, {nb_classifiable = }')
        return self.Res(
            loss=res_cls.loss,
            cls_logits=res_cls.logits,
            cls_preds=preds,
            nb_tp=int(nb_tp),
            nb_fp=int(nb_fp),
            nb_fn=int(nb_fn),
            nb_tn=int(nb_tn),
            nb_cls=int(nb_classifiable),
            nb_cor_gt=int(nb_actually_right_cells_in_input),
            cls_labels=cls_labels
        )
    

class TrainableInputsWrapper(nn.Module):
    @dataclass
    class Res:
        inner_model_output: ModelClsCritic.Res
        trained_logits: torch.Tensor
        trained_input_loss: torch.Tensor
        
    def __init__(
        self,
        ids : torch.Tensor, # [B, H, W]
        trainable_mask : torch.Tensor, # [B, H, W], 1 if trainable, 0 if not
        inputs_vocab_size : int, # not the same as model, amybe we dont want to predict all tokens
        replace_trainable : bool = True # if true, replace trainable cells with uniform distribution at start  
    ):
        super().__init__()
        self.replace_trainable_and_untrainable = replace_trainable
        self.trainable_mask = trainable_mask  # [B, H, W], 1 if trainable, 0 if not
        self.non_trainable_mask = 1 - trainable_mask # [B, H, W], 1 if not trainable, 0 if trainable
        
        self.vocab_size = inputs_vocab_size
        self.ids = ids

        self.onehots = F.one_hot(ids.clamp(max=inputs_vocab_size - 1), num_classes=inputs_vocab_size).float() # [B, H, W, V]
        self.nb_trainable = trainable_mask.sum().item()
        
        initial_logits = self.onehots.clone().detach() # [B, H, W, V]
        if self.replace_trainable_and_untrainable :
            log('LKDSALDJSADLKJSADLJSA')
            initial_logits[self.trainable_mask == 1] = 1.0 / inputs_vocab_size # uniform distribution for trainable cells
        self.trainable_inputs : nn.Parameter = nn.Parameter(initial_logits) # [B, H, W, V]

    def reset_non_trainable_inputs(self):
        if self.replace_trainable_and_untrainable :
            with torch.no_grad():
                self.trainable_inputs[self.non_trainable_mask == 1] = self.onehots[self.non_trainable_mask == 1]
        
    def forward(
        self,
        model : ModelClsCritic,
        color_labels : Optional[torch.Tensor] = None,
        entropy_reg : float = 0.0 # == 0 => disable, >0  => penalizing entropy => more confident decisions, <0 => encourage entropy => more uniform distribution
    ) -> Res:
        self.reset_non_trainable_inputs()
        res : ModelClsCritic.Res = model(self.trainable_inputs, is_ids = False, color_labels=color_labels)
        p = F.softmax(res.cls_logits, dim=-1)    # [B,H,W,2], porobabilities of each cell to be wrong or right
        p0 = p[..., 0][self.trainable_mask == 1]  # [B,H,W], get only
        input_loss = p0.sum()  # fraction of trainable cells that are likely to be wrong
        
        if entropy_reg != 0.0:
            token_probs = F.softmax(self.trainable_inputs, dim=-1)
            token_ent = -(token_probs * (token_probs.clamp_min(1e-12)).log()).sum(dim=-1).mean()
            input_loss = input_loss + entropy_reg * token_ent
        return self.Res(
            inner_model_output=res,
            trained_logits=self.trainable_inputs,
            trained_input_loss=input_loss
        )



def freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

def unfreeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train()

@dataclass
class ProcessBatchOutput:
    loss: float = 0
    nb_cor: int = 0
    nb_cls: int = 0

    nb_steps: int = 1

    loss_inputs: float = 0.0
    
    trained_logits: torch.Tensor = None # type: ignore
    
    logits : torch.Tensor = None # type: ignore
    hiddens : torch.Tensor = None # type: ignore

class ProcessBatchDo(Enum) :
    train = 'train'
    val = 'val'
    inference = 'inference'

@dataclass
class EpochOutput:
    # set during init
    is_train: bool = True
    batch_size: int = 0
    
    # accumulated
    nb_steps: int = 0
    loss: float = 0.0
    nb_cor: int = 0
    nb_cls: int = 0
    
    # calculated
    mean_acc: float = 0.0
    mean_loss: float = 0.0
    
    
    def calculate(self):
        self.mean_loss = self.loss / max(1, self.nb_steps)
        self.mean_acc = self.nb_cor / max(1, self.nb_cls)





def process_batch_direct(
    bid: int,
    model: nn.Module,
    input_ids: torch.Tensor,
    color_labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mcfg: ModelConfig,
    tcfg: TrainConfig,
    dcfg: DatasetConfig,
    do: ProcessBatchDo
) -> ProcessBatchOutput:    

    input_ids = input_ids.to(tcfg.device)
    color_labels = color_labels.to(tcfg.device)
    res_loss = None
    if do == ProcessBatchDo.train:
        unfreeze(model)
    else:
        freeze(model)

    with torch.set_grad_enabled(do == ProcessBatchDo.train):
        if do == ProcessBatchDo.train:
            optimizer.zero_grad()

        outputs : ModelClassifier.Res = model(input_ids, is_ids =True, labels=color_labels)
        loss = outputs.loss
        assert loss is not None
        res_loss = loss.item()   
        if do == ProcessBatchDo.train:
            loss.backward()
            optimizer.step()
    
    return ProcessBatchOutput(
        loss=res_loss if res_loss is not None else 0.0,
        logits=outputs.logits,
        hiddens=outputs.hiddens,
        nb_cls=outputs.nb_total,
        nb_cor=outputs.nb_corr,
    )



def do_one_epoch_direct(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mcfg : ModelConfig,
    dcfg : DatasetConfig,
    tcfg : TrainConfig,
    do_train : bool
) -> EpochOutput :
    ret = EpochOutput(
        is_train=True,
        batch_size=tcfg.t_batch_size,
    )
    B, H, W = loader.batch_size, mcfg.field_height, mcfg.field_width
    for bid, batch in enumerate(tqdm.tqdm(loader, total=len(loader))):
        input_ids, color_labels = batch['input_ids'], batch['labels']
        
        if tcfg.t_max_nb_aug  > 0 :
            colors_orig, colors_perms, input_ids, color_labels = augment_colors_batch(
                input_ids,
                color_labels,
                max_permutations=tcfg.t_max_nb_aug,
                ignore_label_id=mcfg.ignore_id,
                pad_token_id=mcfg.pad_token_id,
                add_orig=True
            )

        res : ProcessBatchOutput = process_batch_direct(
            bid=bid,
            model=model,
            input_ids=input_ids,
            color_labels=color_labels,
            optimizer=optimizer,
            mcfg=mcfg,
            tcfg=tcfg,
            dcfg=dcfg,
            do=ProcessBatchDo.train if do_train else ProcessBatchDo.val,
        )
        
        ret.nb_steps += 1
        ret.loss += res.loss
        ret.nb_cls += res.nb_cls
        ret.nb_cor += res.nb_cor
    
    ret.calculate()
    return ret



def train_on_direct_objective(
    create_models : Callable,
    create_optimizer : Callable,
    create_dataloaders : Callable,
    save_models : Callable,
    checkpoint_path : str,
    mcfg : ModelConfig,
    dcfg : DatasetConfig,
    tcfg : TrainConfig,
) :
    encoder_path = os.path.join(checkpoint_path, "encoder.pth")
    classifier_path = os.path.join(checkpoint_path, "classifier.pth")
    encoder_model, cls_model = create_models(
        vocab_size=mcfg.vocab_size,
        chkpt_path_encoder=encoder_path,
        chkpt_path_classifier=classifier_path
    )
    optimizer = create_optimizer(
        model=cls_model,
        lr=tcfg.lr
    )
    train_dl, val_dl = create_dataloaders()

    epoch = 0
    try : 
        while 1 :
            tr = do_one_epoch_direct(
                model = cls_model,
                loader = train_dl,
                optimizer = optimizer,
                mcfg = mcfg,
                dcfg = dcfg,
                tcfg = tcfg,
                do_train = True,
                
            )
            l  = f"TRAIN Epoch {epoch:02d} | "
            for k, v in tr.__dict__.items() : 
                l += f"{k} {v:.4f} | "
            log(f"{l}")
            
            # Log training metrics to wandb
            train_metrics = {f"train/{k}": v for k, v in tr.__dict__.items()}
            train_metrics["epoch"] = epoch
            wandb.log(train_metrics)

            l  = f"VAL Epoch {epoch:02d} | "
            va = do_one_epoch_direct(
                model = cls_model,
                loader = val_dl,
                optimizer = optimizer,
                mcfg = mcfg,
                dcfg = dcfg,
                tcfg = tcfg,
                do_train = False,
            )
            for k, v in va.__dict__.items() :
                l += f"{k} {v:.4f} | "
            log(f"{l}")
            
            # Log validation metrics to wandb
            val_metrics = {f"val/{k}": v for k, v in va.__dict__.items()}
            wandb.log(val_metrics)

            epoch += 1
    except KeyboardInterrupt :
        input("Press Enter to confirm saving model, else press ctrl+c...")
        log("Training interrupted by user")
    save_models(encoder_model, encoder_path)
    save_models(cls_model, classifier_path)






def optimize_inputs(
    input_ids: torch.Tensor,
    model: nn.Module,
    nb_steps: int,
    masked_token_id: int,
    expand: bool,
    inputs_vocab_size: int,
) :
    total_loss = 0.0

    trainable_mask = (input_ids == masked_token_id).long() if expand else torch.ones_like(input_ids) # [B, H, W], 1 if trainable, 0 if not
    assert trainable_mask.sum().item() > 0, "No trainable cells in the input!"
    
    input_trainer : TrainableInputsWrapper = TrainableInputsWrapper(
        ids=input_ids,
        trainable_mask=trainable_mask,
        inputs_vocab_size=inputs_vocab_size,
        replace_trainable=expand
    ).to(input_ids.device)
    optimizer_input_trainer = torch.optim.AdamW(input_trainer.parameters(), lr=1e-2, betas=(0.9, 0.98), weight_decay=1e-2)

    cached_intermediate_inputs: List[torch.Tensor] = [input_trainer.trainable_inputs.detach().clone()]
    last_frac_equals = 0.0
    if nb_steps  > 0 :
        freeze(model)
        for step in range(nb_steps):
            optimizer_input_trainer.zero_grad(set_to_none=True)
            res_ingrad : TrainableInputsWrapper.Res = input_trainer(model)
            inp_loss = res_ingrad.trained_input_loss
            inp_loss.backward()
            optimizer_input_trainer.step()
            total_loss += inp_loss.item()
            trained_ids = res_ingrad.trained_logits.argmax(dim=-1)
            assert trained_ids.shape == input_ids.shape
            equals = trained_ids == input_ids
            frac_equals = round(float(equals.sum().item() / equals.numel()), 4)
            if abs(last_frac_equals - frac_equals) > 0.0001 :
                last_frac_equals = frac_equals
                cached_intermediate_inputs.append(input_trainer.trainable_inputs.detach().clone())
                log(f'Step {step + 1}/{nb_steps}, input loss: {inp_loss.item():.4f}, frac equal to original: {frac_equals:.4f}')
    else :
        total_loss = 0.0
    return total_loss, cached_intermediate_inputs
    
     
def process_batch_ctitic_optimize_inputs(
    bid: int,
    model: nn.Module,
    input_ids: torch.Tensor,
    color_labels: torch.Tensor,
    mcfg: ModelConfig,
    tcfg: TrainConfig,
    dcfg: DatasetConfig,
    do_train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> ProcessBatchOutput:
    ret : ProcessBatchOutput = ProcessBatchOutput()
    B, H, W = input_ids.shape[0], mcfg.field_height, mcfg.field_width
    nb_steps_optimize_inputs = tcfg.t_nb_inp_opt_steps if do_train else tcfg.v_nb_inp_opt_steps

    total_inputs_loss, cached_intermediate_inputs  = optimize_inputs(
        input_ids=input_ids,
        model=model,
        nb_steps=nb_steps_optimize_inputs,
        masked_token_id=mcfg.masked_token_id,
        expand=dcfg.expand,
        inputs_vocab_size=mcfg.trained_inputs_vocab_size
    )
    last_trained_input = cached_intermediate_inputs[-1]
    last_pred = last_trained_input.argmax(dim=-1)
    last_pred_acc = (last_pred[color_labels != mcfg.ignore_id] == color_labels[color_labels != mcfg.ignore_id]).sum().item() / (color_labels != mcfg.ignore_id).sum().item()
    # log(f'Batch {bid}: optimized inputs loss: {total_inputs_loss:.4f}, last pred acc: {last_pred_acc:.4f}, nb cached intermediate inputs: {len(cached_intermediate_inputs)}')
    if last_pred_acc != 1.0 :
        log_warn(f'Batch {bid}: optimized last pred acc: {last_pred_acc:.4f}, nb cached intermediate inputs: {len(cached_intermediate_inputs)}')
        log(f'{last_pred[0] = }, {color_labels[0] = }')

    if do_train:
        assert optimizer is not None
        unfreeze(model)
        optimizer.zero_grad(set_to_none=True)
    else:
        freeze(model)
    
    for intermediate_inputs in cached_intermediate_inputs:
        res_cls: ModelClsCritic.Res = model(
            intermediate_inputs,
            is_ids=False,
            color_labels=color_labels
        )

        assert res_cls.loss is not None
        cls_loss = res_cls.loss

        if do_train:
            assert optimizer is not None
            cls_loss.backward()
   
        ret.loss += cls_loss.item()
        ret.nb_cor += res_cls.nb_tp + res_cls.nb_tn
        ret.nb_cls += res_cls.nb_cls

    if do_train:
        assert optimizer is not None
        assert cached_intermediate_inputs[0].argmax(dim=-1).equal(input_ids), "First cached input must be equal to original input"
        optimizer.step()

    return ret


def do_one_epoch_critic(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mcfg : ModelConfig,
    dcfg : DatasetConfig,
    tcfg : TrainConfig,
    do_train : bool
) -> EpochOutput :
    unfreeze(model)
    ret = EpochOutput(
        is_train=True,
        batch_size=tcfg.t_batch_size,
    )
    B, H, W = loader.batch_size, mcfg.field_height, mcfg.field_width
    for bid, batch in enumerate(tqdm.tqdm(loader, total=len(loader))):
        input_ids, color_labels = batch['input_ids'], batch['labels']
        
        if tcfg.t_max_nb_aug  > 0 :
            colors_orig, colors_perms, input_ids, color_labels = augment_colors_batch(
                input_ids,
                color_labels,
                max_permutations=tcfg.t_max_nb_aug,
                ignore_label_id=mcfg.ignore_id,
                pad_token_id=mcfg.pad_token_id,
                add_orig=True
            )

        res : ProcessBatchOutput = process_batch_ctitic_optimize_inputs(
            bid=bid,
            model=model,
            input_ids=input_ids,
            color_labels=color_labels,
            optimizer=optimizer,
            mcfg=mcfg,
            tcfg=tcfg,
            dcfg=dcfg,
            do_train=do_train,
        )
        
        ret.nb_steps += 1
        ret.loss += res.loss
        ret.nb_cls += res.nb_cls
        ret.nb_cor += res.nb_cor
        
    ret.calculate()
    return ret

    




def train_on_critic_objective(
    create_models : Callable,
    create_optimizer : Callable,
    create_dataloaders : Callable,
    save_models : Callable,
    checkpoint_path : str,
    mcfg : ModelConfig,
    dcfg : DatasetConfig,
    tcfg : TrainConfig,
) :

    encoder_model, cls_model, model_critic = create_models()
    optimizer = create_optimizer(
        model=model_critic,
        lr=tcfg.lr
    )
    train_dl, val_dl = create_dataloaders()

    epoch = 0
    try : 
        while 1 :
            tr = do_one_epoch_critic(
                model = model_critic,
                loader = train_dl,
                optimizer = optimizer,
                mcfg = mcfg,
                dcfg = dcfg,
                tcfg = tcfg,
                do_train = True,
                
            )
            l  = f"TRAIN Epoch {epoch:02d} | "
            for k, v in tr.__dict__.items() : 
                l += f"{k} {v:.4f} | "
            log(f"{l}")
            
            # Log training metrics to wandb
            train_metrics = {f"train/{k}": v for k, v in tr.__dict__.items()}
            train_metrics["epoch"] = epoch
            wandb.log(train_metrics)

            l  = f"VAL Epoch {epoch:02d} | "
            va = do_one_epoch_critic(
                model = model_critic,
                loader = val_dl,
                optimizer = optimizer,
                mcfg = mcfg,
                dcfg = dcfg,
                tcfg = tcfg,
                do_train = False,
            )
            for k, v in va.__dict__.items() :
                l += f"{k} {v:.4f} | "
            log(f"{l}")
            # Log validation metrics to wandb
            val_metrics = {f"val/{k}": v for k, v in va.__dict__.items()}
            wandb.log(val_metrics)

            epoch += 1
    except KeyboardInterrupt :
        input("Press Enter to confirm saving model, else press ctrl+c...")
        log("Training interrupted by user")

    save_models(encoder_model, path=os.path.join(checkpoint_path, "encoder.pth"))
    # save_models(cls_model, checkpoint_path, name="classifier.pth")
    save_models(model_critic, path=os.path.join(checkpoint_path, "critic.pth"))



    
def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    
    field_width = 15
    seq_len = 100
    expand = False
    
    dcfg = DatasetConfig(
        seed=123,
        val_frac=0.1,
        test_frac=0.0,
        num_samples=800,
        seq_len=seq_len,
        max_width=field_width,
        expand=expand,
    )

    tcfg = TrainConfig(
        lr=1e-4,

        # train cfg
        t_batch_size=4,
        t_nb_max_self_correction=1,
        t_show_in_window=True,
        t_max_nb_aug=0,
        t_show_nb_b=0,
        t_nb_inp_opt_steps=1,
        

        # val cfg
        v_batch_size=3,
        v_nb_max_self_correction=1,
        v_do_augmented_inference=False,
        v_show_in_window=True,
        v_max_nb_aug=0,
        v_show_nb_b=0,
        v_nb_inp_opt_steps=1
    )
    
    mcfg = ModelConfig(
        d_model=128,
        n_head=8,
        d_head=64,
        num_layers=8,
        dim_feedforward=128,
        vocab_size=200,
        
        nb_max_rope_positions=4000,
        
        nb_refinement_steps=1,
        nb_last_trained_steps=1,
        
        enable_pseudo_diffusion_inner=False,
        enable_pseudo_diffusion_outer=True,
        feed_first_half=False,

        # use_transposed_rope_for_2d_vertical_orientation=False,
        # field_width=field_width,
        # field_height=field_width * 2,

        use_emb_norm = False,
        
        use_axial_rope = True,
        
        # learned pos emb
        use_learned_pos_emb=False,
        use_custom_learned_pos_emb_per_head=True,
        
        # learned pos emb with custo dim
        use_projection_for_learned_pos_embs=False,        
        learned_pos_embs_dim=64,
        
        # if use ready made implementation from x_transformers
        use_x_encoder=False,

        # neccessary for 2d learned pos embedding
        field_width=field_width,
        field_height=field_width * (2 if expand else 1),
        
        # cnn in each attention layer 
        use_cnn=True,
        
        trained_inputs_vocab_size=11,

    )

    wandb.init(
        project="intputs_gradient_optimization",
        config={
            "dataset": dcfg.__dict__,
            "training": tcfg.__dict__,
            "model": mcfg.__dict__,
            "field_width": field_width,
            "expand": expand
        }
    )


    








    
    task_name = 'fill_between_pieces'
    ds_raw = get_custom_ds_arc(
        seq_len=dcfg.seq_len,
        nb_samples=dcfg.num_samples,
        nb_cls=10,
        task=task_name,

        field_width = field_width,


        do_2d = True,
        do_transpose = False,

        nb_missing_min = 20,
        nb_missing_max = 21,
        masked_token_id = mcfg.masked_token_id,
    )

    for i, o in ds_raw :
        print(i)
        print(o)
        print('===============================')
        break
    

    
    
    def create_encoder_model() :
        return CNNEncoder1d(
            mcfg=mcfg
        ).to(tcfg.device)
        
    def create_classifier_model(encoder_model, vocab_size : int) :
        return ModelClassifier(
            model_encoder=encoder_model,
            cfg=mcfg,
            nb_cls=vocab_size
        ).to(tcfg.device)
    
    def create_critic_model(
        cls_model : ModelClassifier
    ) :
        return ModelClsCritic(
            classifier=cls_model,
            mcfg=mcfg
        ).to(tcfg.device)
            
    
    def create_model_optimizer(
        model : nn.Module,
        lr : float,
        betas : Tuple[float, float] = (0.9, 0.98),
        eps : float = 1e-9
    ) -> torch.optim.Optimizer :
        return torch.optim.Adafactor(model.parameters(), lr=lr)

    
    def load_if_path(model: nn.Module, chkpt_path : Optional[str] = None) :
        if chkpt_path is not None :
            if not os.path.exists(chkpt_path) :
                log_warn(f"Checkpoint path {chkpt_path} does not exist, starting from scratch")
            else :
                state_dict = torch.load(chkpt_path, map_location=tcfg.device)
                model.load_state_dict(state_dict)
                log(f"Model loaded from {chkpt_path}")
        return model
    
    def create_encoder_model_with_checkpoint(chkpt_path : Optional[str] = None) :
        encoder_model = create_encoder_model()
        load_if_path(encoder_model, chkpt_path=chkpt_path)
        return encoder_model

    def create_cls_models(
        vocab_size : int,
        chkpt_path_encoder : Optional[str] = None,
        chkpt_path_classifier : Optional[str] = None
    ) :
        encoder_model = create_encoder_model()
        cls_model = create_classifier_model(encoder_model, vocab_size=vocab_size)
        load_if_path(cls_model, chkpt_path=chkpt_path_classifier)
        load_if_path(encoder_model, chkpt_path=chkpt_path_encoder)
        return encoder_model, cls_model 

    def save_models(model : nn.Module, path : str) :
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        log(f"Model saved to {path}")
    
    def create_dataloaders() :
        train_dl, val_dl = get_dataloaders_for_flat_seq_cls(
            ds_raw,
            ignore_label_id = mcfg.ignore_id,
            sep_token_id = mcfg.qa_sep_token_id,
            pad_token_id = mcfg.pad_token_id,
            split_ratio = 1 - dcfg.val_frac,
            batch_size_train = tcfg.t_batch_size,
            batch_size_eval = tcfg.v_batch_size,
            device = tcfg.device,
            add_labels_to_inputs=dcfg.add_labels_to_inputs,
            add_sep=dcfg.add_sep,
            expand=dcfg.expand,
            expand_inputs_token_id=mcfg.masked_token_id,
        )
        return train_dl, val_dl

    train_on_direct_objective(
        create_models=create_cls_models,
        create_optimizer=create_model_optimizer,
        create_dataloaders=create_dataloaders,
        save_models=save_models,
        checkpoint_path='./models/pretrained_cls/',
        mcfg=mcfg,
        dcfg=dcfg,
        tcfg=tcfg,
    )
    

    def init_critic_from_pretrained_cls_or_critic(from_cls: bool = True):
        path = './models/pretrained_cls/' if from_cls else './models/critic/'
        encoder_path = os.path.join(path, "encoder.pth")
        classifier_path = os.path.join(path, "classifier.pth")
        critic_path = os.path.join(path, "critic.pth")
        if from_cls :
            encoder_model, cls_direct_objective = create_cls_models(
                vocab_size=mcfg.vocab_size,
                chkpt_path_encoder=encoder_path,    
                chkpt_path_classifier=classifier_path
            )
            _, cls_critic = create_cls_models(
                vocab_size=2,
            )
            critic_model = create_critic_model(cls_critic)
            critic_model.embedding = cls_direct_objective.embeddings
            return encoder_model, cls_critic, critic_model
        else :
            encoder_model, cls_direct_objective = create_cls_models(
                vocab_size=2,
                chkpt_path_encoder=encoder_path,    
                chkpt_path_classifier=classifier_path
            )
            critic_model = create_critic_model(cls_direct_objective)
            load_if_path(critic_model, chkpt_path=critic_path)
            return encoder_model, cls_direct_objective, critic_model


    
    train_on_critic_objective(
        create_models=functools.partial(init_critic_from_pretrained_cls_or_critic, from_cls=False),
        create_optimizer=create_model_optimizer,
        create_dataloaders=create_dataloaders,
        save_models=save_models,
        checkpoint_path='./models/critic/',
        mcfg=mcfg,
        dcfg=dcfg,
        tcfg=tcfg,
    )









    wandb.finish()
    log("Done.")

if __name__ == "__main__":
    main()
