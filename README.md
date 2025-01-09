
# Meta-LoRA

This repository provides the PyTorch implementation of the following paper:
> [**Meta-LoRA: Memory-Efficient Sample Reweighting for Fine-Tuning Large Language Models**]() <br>
> [Weicheng Li]()<sup>1</sup>,
> [Lixin Zou]()<sup>1</sup>,
> [Min Tang]() <sup>2</sup>,
> [Qing Yu]()<sup>1</sup>,
> [Wanli Li]()<sup>3</sup>,
> [Chenliang Li]() <sup>1</sup>,
> <sup>1</sup>Wuhan University, <sup>2</sup>Monash University, <sup>3</sup>Huazhong Agricultural University<br>

## :hammer_and_wrench: Installation

To install, run the following commands to install the required packages:

```
git clone https://github.com/liweicheng-ai/meta-lora.git
cd meta-lora/DeepSpeed-Chat/
pip install -r requirements.txt
pip install -e .
```

We employ [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) as the training framework and make some modifications to tailor it to Meta-LoRA. 

 
## :fire: LLM Backbones

In our work, we adopt Baichuan2-7B-Base model and OPT-1.3B as the backbones:
- Download [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base).
- Download [opt-1.3b](https://huggingface.co/facebook/opt-1.3b).

You can also use other LLMs as the backbones.

### Arguments

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--model_name_or_path`    | `/path/to/model` | Specify the LLM model.|
| `--train_data`     | `/path/to/dataset` | Path to the training dataset file, e.g., `./data/MathInstruct.json`. |
| `--dataset_format`     | `alpaca` | The format of the training dataset file. |
| `--eval_data`     | `/path/to/dataset` | Path to the eval dataset file, e.g., `./data/gsm8k/train.json`. |
| `--eval_dataset_format`     | `gsm8k` | The format of the eval dataset file. |
| `--meta`     | `True` | Whether to perform data weighting, supports `True`, `False`. |
| `--validation_interval`     | `10` | The step interval of updating validation samples. |

#### Additional arguments for datasets

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--train_input`      | `instruction` | Specify the input field of training data when args.dataset_format is set to 'self_defined'. Default: None. |
| `--train_output`      | `response` | Specify the output field of training data when args.dataset_format is set to 'self_defined'. Default: None. |
| `--eval_input`      | `instruction` | Specify the input field of eval data when args.eval_dataset_format is set to 'self_defined'. Default: None. |
| `--eval_output`      | `response` | Specify the output field of eval data when args.eval_dataset_format is set to 'self_defined'. Default: None. |

#### Arguments for LoRA

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--lora_dim`   | `32` | The rank of LoRA modules. |
| `--lora_module_name`      | `down_proj,W_pack,gate_proj,up_proj,o_proj` | Specify which trainable modules to perform Low-Rank Adaptation on. |


## :rocket: Running

#### Single GPU

```
bash train.sh
```

#### Multi GPUs

```
bash train_multi_gpu.sh
```
