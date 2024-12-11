#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import math
import numpy as np
import os
import copy

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

import sys
sys.path.append("./DeepSpeed-Chat")
from dschat.utils.data.data_utils import create_prompt_dataset
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_train_ds_config
from module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from dschat.utils.perf import print_throughput

from mydataset import IGNORE_INDEX, make_data_module

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--train_data',
                        type=str, 
                        default=None,
                        help='Path to the training dataset.')
    parser.add_argument('--eval_data',
                        type=str,
                        default=None,
                        help='Path to the eval dataset for meta-lora algorithm.')
    parser.add_argument("--meta", type=str, default='False')
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument("--group_by_length", type=bool, default=True)
    parser.add_argument('--dataset_format',
                        type=str, 
                        default='alpaca',
                        help='The format of the training dataset.')
    parser.add_argument("--train_input", type=str, default=None, help="x[input]->x['instruction']")
    parser.add_argument("--train_output", type=str, default=None, help="x[output]->x['output']")

    parser.add_argument('--eval_dataset_format', type=str, default="alpaca")
    parser.add_argument("--eval_input", type=str, default=None, help="x[input]->x['instruction']")
    parser.add_argument("--eval_output", type=str, default=None, help="x[output]->x['output']")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="Max steps for each training dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument('--source_max_len', type=int, default=256) # 200
    parser.add_argument('--target_max_len', type=int, default=256)
    parser.add_argument("--train_on_source", type=bool, default=False)
    parser.add_argument("--predict_with_generate", type=bool, default=False)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=10,
        help=
        "Number of steps before updating validation samples used in Meta-LoRA.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
        "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
        'If specified, loss is calculated in fp32.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add <|endoftext|> as additional special token to tokenizer")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)

    train_bs = args.per_device_train_batch_size
    eval_bs = args.per_device_eval_batch_size

    fea_in_list = []
    def hook(module, fea_in, fea_out):
        fea_in_list.append(fea_in[0].data)

    output_grad_list = []
    def hook_backward_function(module, module_input_grad, module_output_grad):
        output_grad_list.append(module_input_grad[0].data)

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()
    print(f"device: {device}")

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    args.end_of_conversation_token = "<|endoftext|>"
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    # tokenizer = load_hf_tokenizer(args.model_name_or_path,
    #                               fast_tokenizer=False,
    #                               add_special_tokens=additional_special_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                            #   cache_dir=args.model_path,
                                            #   token=hf_token,
                                              padding_side="right",
                                              use_fast=False, 
                                              trust_remote_code=True, 
                                              local_files_only=True,
                                              )

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout)
    
    vocab_size = model.config.vocab_size

    if args.compute_fp32_loss:
        print_rank_0(
            f"Using model {model.__class__.__name__} with loss in fp32",
            args.global_rank)
        causal_lm_model_to_fp32_loss(model)
    if args.lora_dim > 0:
        lora_module_name_list = args.lora_module_name.split(',')
        # print("lora_module_name_list: ", lora_module_name_list)
        model = convert_linear_layer_to_lora(model, lora_module_name_list,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)
            model = make_model_gradient_checkpointing_compatible(model)
        
    handle_io_hook = []
    handle_grad_hook = []
            
    if (args.meta == 'True'):
        model.enable_input_require_grads()
        for name, module in model.named_modules(): # add hooks
            if ("lora_left" in name):
                input_hook = module.register_forward_hook(hook=hook)
                handle_io_hook.append(input_hook)
                output_grad_hook = module.register_full_backward_hook(hook_backward_function)
                handle_grad_hook.append(output_grad_hook)
            else:
                pass

    # mydataset
    train_dataset, train_data_collator = make_data_module(tokenizer=tokenizer, args=args, dataset_format=args.dataset_format, is_train=True)
    if (args.meta == 'True'):
        eval_dataset, eval_data_collator = make_data_module(tokenizer=tokenizer, args=args, dataset_format=args.eval_dataset_format, is_train=False)
    
    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = SequentialSampler(train_dataset)
        if (args.meta == 'True'):
            eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        if (args.meta == 'True'):
            eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=train_data_collator,
                                  sampler=train_sampler,
                                  shuffle=False,
                                  batch_size=args.per_device_train_batch_size)
    train_dataloader_list = [train_dataloader]
    if (args.meta == 'True'):
        eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=eval_data_collator,
                                    sampler=eval_sampler,
                                    shuffle=False,
                                    batch_size=args.per_device_eval_batch_size)
        eval_data_iter = iter(eval_dataloader)

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)
    # print(optimizer_grouped_parameters)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_steps if args.max_steps is not None else args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    def get_loss(outputs, labels, bs):
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size), shift_labels.view(-1), reduction='none')
        loss = loss.reshape(bs, -1)
        train_mask = shift_labels.reshape(bs, -1) != IGNORE_INDEX
        num_non_zeros = train_mask.sum(1)
        gathered_loss = (loss * train_mask).sum(1) / num_non_zeros
        return gathered_loss

    loss_save_list = []
    real_step = 0
    for epoch in range(args.num_train_epochs):
        if (real_step >= args.max_steps):
            break
        try:
            train_dataloader = train_dataloader_list[epoch]
        except:
            train_dataloader = train_dataloader_list[0]
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        import time
        for step, train_batch in enumerate(train_dataloader):
            if (real_step >= args.max_steps):
                break
            start = time.time()
            if (args.meta == 'False' or (step == 0)):
                batch = to_device(train_batch, device)
                labels = batch['labels']
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                train_bs = len(outputs.logits)
                if (step % 10 == 0):
                    loss_save_list.append(loss.item())
                if args.print_loss:
                    print(
                        f"\nEpoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                    )
                model.zero_grad()
                model.backward(loss)
                model.step()
                real_step += 1
                fea_in_list.clear()
                output_grad_list.clear()
                model.zero_grad()

            else: # meta == 'True' and step > 0
                if ((step - 1) % args.validation_interval == 0):
                    try:
                        eval_batch = next(eval_data_iter)
                    except StopIteration:
                        eval_data_iter = iter(eval_dataloader) # reload
                        eval_batch = next(eval_data_iter)
                    batch = to_device(eval_batch, device)

                    eval_outputs = model(**batch, use_cache=False)
                    eval_bs = len(eval_outputs.logits)
                    eval_loss = eval_outputs.loss
                    print(f"Step: {step}, eval Loss: {eval_loss}")

                    global val_fea_in_list, val_output_grad_list # For reuse
                    val_fea_in_list = copy.deepcopy(fea_in_list) # Each tensor shape: (val_bs, val_seq_len, input_dim)
                    fea_in_list.clear()

                    with torch.no_grad():
                        for idx in range(len(val_fea_in_list)):
                            val_fea_in = val_fea_in_list[idx] # (val_bs, val_seq_len, input_dim)
                            val_fea_in = val_fea_in.view(eval_bs, -1, val_fea_in.shape[-1])
                            val_fea_in = torch.mean(val_fea_in, dim=1) # shape: (val_bs, val_input_dim)
                            val_fea_in = torch.mean(val_fea_in, dim=0) # shape: (val_input_dim)
                            val_fea_in_list[idx] = val_fea_in # shape: (2 * val_input_dim) 64
                        val_fea_in_list = torch.stack(val_fea_in_list) # shape: (layer_num, 2 * val_input_dim) (32, 64)
                    
                    model.zero_grad()
                    model.backward(eval_loss)
                    output_grad_list = list(reversed(output_grad_list))
                    val_output_grad_list = copy.deepcopy(output_grad_list) 
                    fea_in_list.clear() 
                    output_grad_list.clear()

                    with torch.no_grad():
                        for idx, val_output_grad in enumerate(val_output_grad_list):
                            val_output_grad = val_output_grad.view(eval_bs, -1, val_output_grad.shape[-1])
                            val_output_grad = torch.mean(val_output_grad, dim=1) # shape: (val_bs, val_output_dim)
                            val_output_grad = torch.mean(val_output_grad, dim=0) # shape: (val_output_dim)
                            val_output_grad_list[idx] = val_output_grad # len(list) = layer_num; the shape of each element: (val_output_dim)
                        val_output_grad_list = torch.stack(val_output_grad_list) # shape: (layer_num, val_output_dim) (32, 32)

                    model.step()
                    real_step += 1

                    del eval_outputs, eval_loss
                    torch.cuda.empty_cache()
                    model.zero_grad()

                else:
                    pass # No need to update val_fea_in_list and val_output_grad_list every step.
                    
                batch = to_device(train_batch, device)
                labels = batch['labels']
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                train_bs = len(outputs.logits)
                if (step % 10 == 0):
                    loss_save_list.append(loss.item())
                if args.print_loss:
                    print(
                        f"\nEpoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                    )
                train_gathered_loss = get_loss(outputs, labels, train_bs)
                print("train_loss_of_each_sample: ", train_gathered_loss)
                
                train_fea_in_list = copy.deepcopy(fea_in_list) # Each tensor shape: (train_bs, seq_len, input_dim)
                fea_in_list.clear()

                with torch.no_grad():
                    for idx in range(len(train_fea_in_list)):
                        train_fea_in = train_fea_in_list[idx] # (train_bs, train_seq_len, input_dim)
                        train_fea_in = train_fea_in.view(train_bs, -1, train_fea_in.shape[-1])
                        train_fea_in = torch.mean(train_fea_in, dim=1) # shape: (train_bs, train_input_dim) (4, 32)
                        train_fea_in_list[idx] = train_fea_in # shape: (train_bs, 2 * train_input_dim) (4, 64)
                    train_fea_in_list = torch.stack(train_fea_in_list) # shape: (layer_num, train_bs, 2 * train_input_dim) # (32, 4, 64)
                    train_fea_in_list = train_fea_in_list.transpose(0, 1) # shape: (4, 32, 64) (train_bs, layer_num, 2 * train_input_dim)
                    # print(train_fea_in_list.shape)

                model.zero_grad()
                model.backward(loss, retain_graph=True)
                output_grad_list = list(reversed(output_grad_list))
                train_output_grad_list = copy.deepcopy(output_grad_list)           
                fea_in_list.clear() 
                output_grad_list.clear()
            
                with torch.no_grad():
                    for idx, train_output_grad in enumerate(train_output_grad_list):     
                        train_output_grad = train_output_grad.view(train_bs, -1, train_output_grad.shape[-1])
                        train_output_grad = torch.mean(train_output_grad, dim=1) # shape: (train_bs, train_output_dim) Reduce the dimension of "seq_len" by averaging
                        train_output_grad_list[idx] = train_output_grad # shape: (train_bs, train_output_dim)       
                    train_output_grad_list = torch.stack(train_output_grad_list) # (32, 4, 32)
                    train_output_grad_list = train_output_grad_list.transpose(0, 1) # (4, 32, 32) (train_bs, layer_num, train_output_dim)
                
                # shape:        (4, 32, 64)             (4, 32, 32)            (32, 64)           (32, 32)
                # Now we have train_fea_in_list, train_output_grad_list, val_fea_in_list and val_output_grad_list
                assert train_fea_in_list.shape[1] == val_fea_in_list.shape[0] # Make sure they have equal number of layers
                assert train_output_grad_list.shape[1] == val_output_grad_list.shape[0]
                
                with torch.no_grad():
                    input_sim = (train_fea_in_list * val_fea_in_list).sum(-1) # shape: (4, 32) (train_bs, layer_num)
                    grad_sim = (train_output_grad_list * val_output_grad_list).sum(-1) # shape: (4, 32) (train_bs, layer_num)
                    ex_weight = (input_sim * grad_sim).sum(-1) # shape: 4  (train_bs)

                    print(f'Original ex_weight: {ex_weight}')
                    if (min(ex_weight) < 0.0): # one weight is below 0
                        ex_weight = torch.sub(ex_weight, 2 * min(ex_weight))
                        print(f'Retified ex_weight: {ex_weight}')
                    else:
                        ex_weight = torch.maximum(ex_weight, torch.tensor(0.0, device=device, requires_grad=False))
                        print(f'Retified ex_weight: {ex_weight}')
                    ex_weight_sum = torch.sum(ex_weight)
                    ex_weight_sum += torch.tensor(float(ex_weight_sum == 0.0))
                    ex_weight = ex_weight / ex_weight_sum
                    # ex_weight = ex_weight.to(torch.bfloat16)
                    print(f'Final ex_weight: {ex_weight}') # the weight of each sample
    
                # print("train_loss_of_each_sample: ", train_gathered_loss)

                final_loss = ex_weight @ train_gathered_loss
                print("final_loss: ", final_loss)

                if (final_loss.data == 0.0):
                    fea_in_list.clear()
                    output_grad_list.clear()
                    model.zero_grad()
                    continue
                else:
                    model.zero_grad()
                    model.backward(final_loss)
                    model.step()
                    real_step += 1

                    fea_in_list.clear()
                    output_grad_list.clear()
                    model.zero_grad()

            end = time.time()
            if torch.distributed.get_rank() == 0:
                print_throughput(model.model, args, end - start,
                                 args.global_rank)
            
        model.tput_timer.update_epoch_count()

    for handle in handle_io_hook:
        handle.remove()
    for handle in handle_grad_hook:
        handle.remove()

    if args.output_dir is not None:
        print_rank_0('saving the final model ...', args.global_rank)
        model = convert_lora_to_linear_layer(model)
        
        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args)

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)
            
        print_rank_0('saving the loss list ...', args.global_rank)
        np.savez(os.path.join(args.output_dir, "loss.npz"), loss_save_list)


if __name__ == "__main__":
    main()
