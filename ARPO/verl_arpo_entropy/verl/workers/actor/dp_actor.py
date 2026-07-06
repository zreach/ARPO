# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False, vopd_topk=None):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        if vopd_topk is not None and self.use_fused_kernels:
            raise RuntimeError("vOPD requires response logits, but actor.use_fused_kernels=True only returns sampled log-probs.")
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            vopd_token_ids = None
            vopd_student_log_probs = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                    if vopd_topk is not None:
                        log_probs_vocab_rmpad = torch.nn.functional.log_softmax(logits_rmpad, dim=-1)
                        if vopd_topk <= 0:
                            vopd_student_log_probs_rmpad = log_probs_vocab_rmpad
                            vopd_token_ids_rmpad = torch.arange(
                                log_probs_vocab_rmpad.size(-1),
                                device=log_probs_vocab_rmpad.device,
                                dtype=torch.long,
                            ).expand(log_probs_vocab_rmpad.size(0), -1)
                        else:
                            topk = min(vopd_topk, log_probs_vocab_rmpad.size(-1))
                            vopd_student_log_probs_rmpad, vopd_token_ids_rmpad = torch.topk(log_probs_vocab_rmpad, k=topk, dim=-1)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if vopd_topk is not None:
                        vopd_student_log_probs_rmpad = gather_outpus_and_unpad(
                            vopd_student_log_probs_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        vopd_token_ids_rmpad = gather_outpus_and_unpad(
                            vopd_token_ids_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if vopd_topk is not None:
                    full_vopd_student_log_probs = pad_input(
                        hidden_states=vopd_student_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_vopd_token_ids = pad_input(
                        hidden_states=vopd_token_ids_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if vopd_topk is not None:
                    vopd_student_log_probs = full_vopd_student_log_probs[:, -response_length - 1 : -1, :]
                    vopd_token_ids = full_vopd_token_ids[:, -response_length - 1 : -1, :].long()

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                    if vopd_topk is not None:
                        log_probs_vocab = torch.nn.functional.log_softmax(logits, dim=-1)
                        if vopd_topk <= 0:
                            vopd_student_log_probs = log_probs_vocab
                            vopd_token_ids = torch.arange(
                                log_probs_vocab.size(-1),
                                device=log_probs_vocab.device,
                                dtype=torch.long,
                            ).view(1, 1, -1).expand(log_probs_vocab.size(0), log_probs_vocab.size(1), -1)
                        else:
                            topk = min(vopd_topk, log_probs_vocab.size(-1))
                            vopd_student_log_probs, vopd_token_ids = torch.topk(log_probs_vocab, k=topk, dim=-1)

            if vopd_topk is not None:
                return entropy, log_probs, vopd_token_ids, vopd_student_log_probs
            return entropy, log_probs

    def _forward_selected_token_log_probs(self, micro_batch, temperature):
        if self.use_fused_kernels:
            raise RuntimeError("vOPD requires teacher logits, but ref.use_fused_kernels=True only returns sampled log-probs.")
        response_token_ids = micro_batch["vopd_token_ids"]
        response_length = response_token_ids.size(1)
        topk = response_token_ids.size(2)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                selected_ids = torch.zeros(
                    batch_size,
                    seqlen,
                    topk,
                    dtype=response_token_ids.dtype,
                    device=response_token_ids.device,
                )
                selected_ids[:, -response_length - 1 : -1, :] = response_token_ids
                selected_ids_rmpad = index_first_axis(rearrange(selected_ids, "b s k -> (b s) k"), indices).long()

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    selected_ids_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                        selected_ids_rmpad.transpose(0, 1),
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    selected_ids_rmpad = selected_ids_rmpad.transpose(0, 1).long()

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)
                logits_rmpad.div_(temperature)
                logsumexp_values = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                selected_log_probs_rmpad = torch.gather(logits_rmpad, dim=-1, index=selected_ids_rmpad) - logsumexp_values

                if self.use_ulysses_sp:
                    selected_log_probs_rmpad = gather_outpus_and_unpad(
                        selected_log_probs_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                full_selected_log_probs = pad_input(
                    hidden_states=selected_log_probs_rmpad,
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                return full_selected_log_probs[:, -response_length - 1 : -1, :]

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]
            logsumexp_values = torch.logsumexp(logits, dim=-1, keepdim=True)
            return torch.gather(logits, dim=-1, index=response_token_ids.long()) - logsumexp_values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        vopd_token_ids_lst = []
        vopd_student_log_probs_lst = []
        export_vopd_topk = self.config.get("export_vopd_topk", False)
        vopd_topk = self.config.get("vopd_topk", 20) if export_vopd_topk else None
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                output = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    vopd_topk=vopd_topk,
                )
                if export_vopd_topk:
                    entropy, log_probs, vopd_token_ids, vopd_student_log_probs = output
                    vopd_token_ids_lst.append(vopd_token_ids)
                    vopd_student_log_probs_lst.append(vopd_student_log_probs)
                else:
                    entropy, log_probs = output
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if export_vopd_topk:
                vopd_token_ids = torch.concat(vopd_token_ids_lst, dim=0)[revert_indices]
                vopd_student_log_probs = torch.concat(vopd_student_log_probs_lst, dim=0)[revert_indices]
        elif export_vopd_topk:
            vopd_token_ids = torch.concat(vopd_token_ids_lst, dim=0)
            vopd_student_log_probs = torch.concat(vopd_student_log_probs_lst, dim=0)

        if export_vopd_topk:
            return log_probs, entropys, vopd_token_ids, vopd_student_log_probs
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_selected_token_log_probs(self, data: DataProto) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["vopd_token_ids", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        selected_log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            selected_log_probs = self._forward_selected_token_log_probs(micro_batch, temperature=temperature)
            selected_log_probs_lst.append(selected_log_probs)

        selected_log_probs = torch.concat(selected_log_probs_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == selected_log_probs.size(0), f"{len(indices)} vs. {selected_log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            selected_log_probs = selected_log_probs[revert_indices]
        return selected_log_probs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)

        use_base_opd = self.config.get("use_base_opd_loss", False)
        use_vopd = self.config.get("use_vopd_loss", False)
        use_myverl_opd = self.config.get("use_myverl_opd_loss", False)
        use_routed_opd = self.config.get("use_routed_opd_loss", False)
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if multi_turn or "loss_mask" in data.batch.keys():
            select_keys.append("loss_mask")
        if self.config.use_kl_loss or use_base_opd or use_vopd or use_myverl_opd:
            select_keys.append("ref_log_prob")
        if use_vopd:
            select_keys.append("vopd_kl_baseline")
        if use_routed_opd:
            for key in ["ref_log_prob", "math_ref_log_prob", "code_ref_log_prob", "math_opd_mask", "code_opd_mask"]:
                if key in data.batch.keys():
                    select_keys.append(key)
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn or "loss_mask" in data.keys():
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if use_routed_opd:
                        loss_mode = self.config.get("routed_opd_loss_mode", "k1")
                        if "math_ref_log_prob" not in data and "code_ref_log_prob" not in data and "ref_log_prob" not in data:
                            raise KeyError("Routed OPD requires math_ref_log_prob/code_ref_log_prob or ref_log_prob.")
                        fallback_teacher_log_prob = data.get("ref_log_prob", None)
                        math_teacher_log_prob = data.get("math_ref_log_prob", fallback_teacher_log_prob)
                        code_teacher_log_prob = data.get("code_ref_log_prob", fallback_teacher_log_prob)
                        if math_teacher_log_prob is None:
                            math_teacher_log_prob = code_teacher_log_prob
                        if code_teacher_log_prob is None:
                            code_teacher_log_prob = math_teacher_log_prob
                        math_mask = data.get("math_opd_mask", torch.zeros_like(response_mask)).to(response_mask.dtype)
                        code_mask = data.get("code_opd_mask", torch.zeros_like(response_mask)).to(response_mask.dtype)
                        math_mask = math_mask * response_mask
                        code_mask = code_mask * response_mask * (1 - math_mask).clamp(min=0, max=1)
                        routed_mask = ((math_mask + code_mask) > 0).to(response_mask.dtype)

                        math_losses = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=math_teacher_log_prob,
                            kl_penalty=loss_mode,
                        )
                        code_losses = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=code_teacher_log_prob,
                            kl_penalty=loss_mode,
                        )
                        distillation_losses = math_losses * math_mask + code_losses * code_mask

                        valid_distill = distillation_losses[routed_mask.bool()]
                        metrics["distillation/routed_math_tokens"] = math_mask.sum().detach().item()
                        metrics["distillation/routed_code_tokens"] = code_mask.sum().detach().item()
                        if valid_distill.numel() > 0:
                            metrics["distillation/routed_abs_loss"] = valid_distill.abs().mean().detach().item()
                            metrics["distillation/routed_loss_min"] = valid_distill.min().detach().item()
                            metrics["distillation/routed_loss_max"] = valid_distill.max().detach().item()

                        loss_max_clamp = self.config.get("routed_opd_loss_max_clamp", None)
                        if loss_max_clamp is not None:
                            distillation_losses = distillation_losses.clamp(
                                min=-loss_max_clamp,
                                max=loss_max_clamp,
                            )

                        if routed_mask.sum() > 0:
                            if self.config.get("routed_opd_use_policy_gradient", True):
                                distill_loss, distill_clipfrac, distill_kl, distill_clipfrac_lower = compute_policy_loss(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=-distillation_losses.detach(),
                                    response_mask=routed_mask,
                                    cliprange=clip_ratio,
                                    cliprange_low=clip_ratio_low,
                                    cliprange_high=clip_ratio_high,
                                    clip_ratio_c=clip_ratio_c,
                                    loss_agg_mode=loss_agg_mode,
                                )
                                metrics["distillation/routed_pg_clipfrac"] = distill_clipfrac.detach().item()
                                metrics["distillation/routed_ppo_kl"] = distill_kl.detach().item()
                                metrics["distillation/routed_pg_clipfrac_lower"] = distill_clipfrac_lower.detach().item()
                            else:
                                distill_loss = agg_loss(
                                    loss_mat=distillation_losses,
                                    loss_mask=routed_mask,
                                    loss_agg_mode=loss_agg_mode,
                                )
                        else:
                            distill_loss = log_prob.new_tensor(0.0)

                        if not self.config.get("routed_opd_use_task_rewards", True):
                            policy_loss = policy_loss.new_tensor(0.0)
                            distill_coef = 1.0
                        else:
                            distill_coef = self.config.get("routed_opd_loss_coef", 1.0)
                        policy_loss = policy_loss + distill_loss * distill_coef
                        metrics["distillation/routed_loss"] = distill_loss.detach().item()
                        metrics["distillation/routed_loss_coef"] = distill_coef

                    if use_base_opd:
                        teacher_log_prob = data["ref_log_prob"]
                        # Base OPD from Eq. (3): r_t = log pi_T(y_t|c_t) - log pi_theta(y_t|c_t),
                        # used as a detached policy-gradient reward on sampled on-policy tokens.
                        opd_rewards = (teacher_log_prob - log_prob).detach()
                        valid_rewards = opd_rewards[response_mask.bool()]
                        if valid_rewards.numel() > 0:
                            metrics["base_opd/reward_mean"] = valid_rewards.mean().detach().item()
                            metrics["base_opd/reward_min"] = valid_rewards.min().detach().item()
                            metrics["base_opd/reward_max"] = valid_rewards.max().detach().item()

                        reward_max_clamp = self.config.get("base_opd_reward_max_clamp", None)
                        if reward_max_clamp is not None:
                            opd_rewards = opd_rewards.clamp(
                                min=-reward_max_clamp,
                                max=reward_max_clamp,
                            )

                        if self.config.get("base_opd_use_policy_gradient", True):
                            base_opd_loss, base_opd_clipfrac, base_opd_kl, base_opd_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=opd_rewards,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                            metrics["base_opd/pg_clipfrac"] = base_opd_clipfrac.detach().item()
                            metrics["base_opd/ppo_kl"] = base_opd_kl.detach().item()
                            metrics["base_opd/pg_clipfrac_lower"] = base_opd_clipfrac_lower.detach().item()
                        else:
                            base_opd_loss = agg_loss(
                                loss_mat=-log_prob * opd_rewards,
                                loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )

                        if not self.config.get("base_opd_use_task_rewards", False):
                            policy_loss = policy_loss.new_tensor(0.0)
                            base_opd_coef = 1.0
                        else:
                            base_opd_coef = self.config.get("base_opd_loss_coef", 1.0)
                        policy_loss = policy_loss + base_opd_loss * base_opd_coef
                        metrics["base_opd/loss"] = base_opd_loss.detach().item()
                        metrics["base_opd/loss_coef"] = base_opd_coef

                    if use_vopd:
                        teacher_log_prob = data["ref_log_prob"]
                        kl_baseline = data["vopd_kl_baseline"].detach()
                        # vOPD from Eq. (10)/(15): advantage = r_t + KL(pi_theta || pi_T),
                        # where the KL baseline is detached and independent of the sampled token.
                        vopd_rewards = (teacher_log_prob - log_prob).detach() + kl_baseline
                        valid_rewards = vopd_rewards[response_mask.bool()]
                        valid_baseline = kl_baseline[response_mask.bool()]
                        if valid_rewards.numel() > 0:
                            metrics["vopd/reward_mean"] = valid_rewards.mean().detach().item()
                            metrics["vopd/reward_min"] = valid_rewards.min().detach().item()
                            metrics["vopd/reward_max"] = valid_rewards.max().detach().item()
                        if valid_baseline.numel() > 0:
                            metrics["vopd/kl_baseline_mean"] = valid_baseline.mean().detach().item()
                            metrics["vopd/kl_baseline_max"] = valid_baseline.max().detach().item()

                        reward_max_clamp = self.config.get("vopd_reward_max_clamp", None)
                        if reward_max_clamp is not None:
                            vopd_rewards = vopd_rewards.clamp(
                                min=-reward_max_clamp,
                                max=reward_max_clamp,
                            )

                        if self.config.get("vopd_use_policy_gradient", True):
                            vopd_loss, vopd_clipfrac, vopd_kl, vopd_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=vopd_rewards,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                            metrics["vopd/pg_clipfrac"] = vopd_clipfrac.detach().item()
                            metrics["vopd/ppo_kl"] = vopd_kl.detach().item()
                            metrics["vopd/pg_clipfrac_lower"] = vopd_clipfrac_lower.detach().item()
                        else:
                            vopd_loss = agg_loss(
                                loss_mat=-log_prob * vopd_rewards,
                                loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )

                        if not self.config.get("vopd_use_task_rewards", False):
                            policy_loss = policy_loss.new_tensor(0.0)
                            vopd_coef = 1.0
                        else:
                            vopd_coef = self.config.get("vopd_loss_coef", 1.0)
                        policy_loss = policy_loss + vopd_loss * vopd_coef
                        metrics["vopd/loss"] = vopd_loss.detach().item()
                        metrics["vopd/loss_coef"] = vopd_coef

                    if use_myverl_opd:
                        teacher_log_prob = data["ref_log_prob"]
                        loss_mode = self.config.get("myverl_opd_loss_mode", "k1")
                        distillation_losses = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=teacher_log_prob,
                            kl_penalty=loss_mode,
                        )
                        valid_distill = distillation_losses[response_mask.bool()]
                        if valid_distill.numel() > 0:
                            metrics["distillation/abs_loss"] = valid_distill.abs().mean().detach().item()
                            metrics["distillation/loss_min"] = valid_distill.min().detach().item()
                            metrics["distillation/loss_max"] = valid_distill.max().detach().item()

                        loss_max_clamp = self.config.get("myverl_opd_loss_max_clamp", None)
                        if loss_max_clamp is not None:
                            distillation_losses = distillation_losses.clamp(
                                min=-loss_max_clamp,
                                max=loss_max_clamp,
                            )

                        if self.config.get("myverl_opd_use_policy_gradient", True):
                            distill_loss, distill_clipfrac, distill_kl, distill_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=-distillation_losses.detach(),
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )
                            metrics["distillation/pg_clipfrac"] = distill_clipfrac.detach().item()
                            metrics["distillation/ppo_kl"] = distill_kl.detach().item()
                            metrics["distillation/pg_clipfrac_lower"] = distill_clipfrac_lower.detach().item()
                        else:
                            distill_loss = agg_loss(
                                loss_mat=distillation_losses,
                                loss_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                            )

                        if not self.config.get("myverl_opd_use_task_rewards", True):
                            policy_loss = policy_loss.new_tensor(0.0)
                            distill_coef = 1.0
                        else:
                            distill_coef = self.config.get("myverl_opd_loss_coef", 1.0)
                        policy_loss = policy_loss + distill_loss * distill_coef
                        metrics["distillation/loss"] = distill_loss.detach().item()
                        metrics["distillation/loss_coef"] = distill_coef

                    if self.config.get("use_opd_loss", False):
                        opd_coef = self.config.get("opd_loss_coef", 0.0)
                        opd_mask = response_mask
                        if self.config.get("opd_positive_only", True):
                            opd_mask = response_mask * (advantages > 0).to(response_mask.dtype)
                        if opd_mask.sum() > 0:
                            opd_loss = agg_loss(loss_mat=-log_prob, loss_mask=opd_mask, loss_agg_mode=loss_agg_mode)
                        else:
                            opd_loss = log_prob.new_tensor(0.0)
                        policy_loss = policy_loss + opd_loss * opd_coef
                        metrics["actor/opd_loss"] = opd_loss.detach().item()
                        metrics["actor/opd_loss_coef"] = opd_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
