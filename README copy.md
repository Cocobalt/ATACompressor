# ATACompressor

## 1 Repository Layout

```
.
├── config/                         # DeepSpeed configs 
│   ├── deepspeed_configurations.json
├── scripts/                        # Helper shell scripts for training/inference
│   ├── pretrain.sh
│   ├── finetune.sh
│   ├── infer.sh
├── train.py                        # Training entrypoint (pretrain & finetune)
├── infer.py                        # Inference entrypoint (regeneration & qa)
├── dataset.py                      # Dataset loading / preprocessing
├── pretrain/
│   └── atacompressor.py            # LoRA module(s) for pretraining
├── finetune/
│   └── atacompressorqa.py          # LoRA module(s) for finetuning
├── inference/                      # (Optional) Saved checkpoints / logs / samples
└── requirements.txt
```

**Directory notes**

- `config/`: Provide one or more DeepSpeed JSON config files (examples shown as `ds_pretrain.json`, `ds_finetune.json`).
- `scripts/`: Optional convenience shell scripts that wrap the CLI invocations shown below.
- `train.py`: Unified training entry point. Use `--task pretrain` or `--task finetune`.
- `infer.py`: Unified inference entry point. Use `--mode regeneration` or `--mode qa`.
- `dataset.py`: Implements dataset loading and tokenization pipelines used by both `train.py` and `infer.py`.
- `pretrain/atacompressor.py` & `finetune/atacompressorqa.py`: LoRA definitions used by the respective stages.
- `inference/`: A suggested folder to store generated outputs, logs, or exported adapters/checkpoints.

------

## 2 Usage

### 2.1 Environment

Create and activate a virtual environment, then install dependencies:

```bash
# Create and activate a Conda environment
conda create -n atacompressor python=3.10 -y
conda activate atacompressor

# Install Python deps (inside the Conda env)
pip install -r requirements.txt
```

### 2.2 Data Preparation

###  Data Preparation

You can preprocess datasets following the procedure described in the paper. 

#### Example A — HotpotQA (preprocessed sample)

```text
Question: "Which writer was from England, Henry Roth or Robert Erskine Childers?"

Context: "<PA> Asgard is a 51 ft gaff rigged yacht. She was owned by the English-born writer and Irish nationalist Erskine Childers and his wife Molly Childers. She is most noted for her use in the Howth gun-running of 1914. </PA> <PA> Henry Roth (February 8, 1906 – October 13, 1995) was an American novelist and short story writer. </PA> <PA> The R509 road, following part of the Childers Road (named after Erskine Childers), is a regional road in Ireland, running through the southeastern side of Limerick City. It forms what is somewhat akin to an inner ring road (albeit mostly two-lane only). </PA> <PA> Mary Alden Osgood Childers, MBE (14 December 1875 – 1 January 1964) was an American-born Irish writer and Irish nationalist. She was the daughter of Dr Hamilton Osgood and Margaret Cushing Osgood of Beacon Hill, Boston, Massachusetts. Her older sister was Gretchen Osgood Warren. Molly married the writer and Irish nationalist, Robert Erskine Childers. Their son, Erskine Hamilton Childers, became the fourth President of Ireland. </PA> <PA> Gretchen Osgood Warren (19 March 1868 – September 1961), the wife of Fiske Warren, was an actress, singer and poet. The daughter of Dr. Hamilton Osgood and Margaret Cushing Osgood of Beacon Hill, Boston, Massachusetts, her younger sister was Mary Alden Childers, the wife of writer and Irish nationalist Robert Erskine Childers. Her nephew Erskine Hamilton Childers served as the fourth President of Ireland from 1973–74. </PA> <PA> Robert Caesar Childers (1838 – 25 July 1876) was a British Orientalist scholar, compiler of the first Pāli-English dictionary. Childers was the husband of Anna Barton of Ireland. He was the father of Irish nationalist Robert Erskine Childers and grandfather to the fourth President of Ireland, Erskine Hamilton Childers. </PA> <PA> Robert Erskine Childers DSC (25 June 1870 – 24 November 1922), universally known as Erskine Childers, was a British writer, whose works included the influential novel "The Riddle of the Sands", and a Fenian revolutionary who smuggled guns to Ireland in his sailing yacht "Asgard". He was executed by the authorities of the nascent Irish Free State during the Irish Civil War. He was the son of British Orientalist scholar Robert Caesar Childers; the cousin of Hugh Childers and Robert Barton; and the father of the fourth President of Ireland, Erskine Hamilton Childers. </PA> <PA> The Irish Bulletin was the official gazette of the government of the Irish Republic. It was produced by the Department of Propaganda during the Irish War of Independence. and its offices were originally located at No. 6 Harcourt Street, Dublin. The paper's first editor was Desmond FitzGerald, until his arrest and replacement by Robert Erskine Childers. "The Bulletin" appeared in weekly editions from 11 November 1919 to 11 July 1921. </PA>"

Gold context: "<PA> Henry Roth (February 8, 1906 – October 13, 1995) was an American novelist and short story writer. </PA> <PA> Robert Erskine Childers DSC (25 June 1870 – 24 November 1922), universally known as Erskine Childers, was a British writer, whose works included the influential novel "The Riddle of the Sands", and a Fenian revolutionary who smuggled guns to Ireland in his sailing yacht "Asgard". He was executed by the authorities of the nascent Irish Free State during the Irish Civil War. He was the son of British Orientalist scholar Robert Caesar Childers; the cousin of Hugh Childers and Robert Barton; and the father of the fourth President of Ireland, Erskine Hamilton Childers. </PA>"

Answer: "Robert Erskine Childers DSC"
```

#### Example B — MS MARCO (preprocessed sample)

```text
Question: "Is Bob Hewitt a citizen of a different country than Ray Ruffels?"

Context: "<PA> The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated. </PA> <PA> The Manhattan Project and its atomic bomb helped bring an end to World War II. Its legacy of peaceful uses of atomic energy continues to have an impact on history and science. </PA> <PA> Essay on The Manhattan Project - The Manhattan Project The Manhattan Project was to see if making an atomic bomb possible. The success of this project would forever change the world forever making it known that something this powerful can be manmade. </PA> <PA> The Manhattan Project was the name for a project conducted during World War II, to develop the first atomic bomb. It refers specifically to the period of the project from 194 … 2-1946 under the control of the U.S. Army Corps of Engineers, under the administration of General Leslie R. Groves. </PA> <PA> versions of each volume as well as complementary websites. The first website–The Manhattan Project: An Interactive History–is available on the Office of History and Heritage Resources website, http://www.cfo.doe.gov/me70/history. The Office of History and Heritage Resources and the National Nuclear Security </PA> <PA> The Manhattan Project. This once classified photograph features the first atomic bomb — a weapon that atomic scientists had nicknamed Gadget.. The nuclear age began on July 16, 1945, when it was detonated in the New Mexico desert. </PA> <PA> Nor will it attempt to substitute for the extraordinarily rich literature on the atomic bombs and the end of World War II. This collection does not attempt to document the origins and development of the Manhattan Project. </PA>"

Gold context: "<PA> Raymond Owen \"Ray\" Ruffels (born 23 March 1946 in Sydney) is an Australian former professional tennis player and coach. </PA> <PA> Robert Anthony John Hewitt (born 12 January 1940) is a former professional tennis player from Australia. In 1967, after marrying a South African, he became a South African citizen. He has won 15 major titles and a career Grand Slam in both men's and mixed doubles. </PA>"

Answer: "yes"
```

#### Example C — SQuAD (preprocessed sample)

```text
Question: "When was the Duchy of Normandy founded?"

Context: "<PA> In the course of the 10th century, the initially destructive incursions of Norse war bands into the rivers of France evolved into more permanent encampments that included local women and personal property. </PA> <PA> The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking ruler Rollo, and was situated in the former Frankish kingdom of Neustria. </PA> <PA> The treaty offered Rollo and his men the French lands between the river Epte and the Atlantic coast in exchange for their protection against further Viking incursions. </PA> <PA> The area corresponded to the northern part of present-day Upper Normandy down to the river Seine, but the Duchy would eventually extend west beyond the Seine. </PA> <PA> The territory was roughly equivalent to the old province of Rouen, and reproduced the Roman administrative structure of Gallia Lugdunensis II (part of the former Gallia Lugdunensis). </PA>"

Gold context: "<PA> The Duchy of Normandy, which began in 911 as a fiefdom, was established by the treaty of Saint-Clair-sur-Epte between King Charles III of West Francia and the famed Viking ruler Rollo, and was situated in the former Frankish kingdom of Neustria. </PA>"

Answer: "911"
```

### 2.3 Training

You can launch training either **directly** with `deepspeed` / `torchrun` or via the provided **scripts**.

#### Pretraining

```bash
# Using DeepSpeed
deepspeed --num_gpus=4 train.py \
  --task pretrain \
  --model_name_or_path <BASE_MODEL> \
  --train_file <PATH/TO/train.jsonl> \
  --validation_file <PATH/TO/valid.jsonl> \
  --output_dir outputs/pretrain \
  --deepspeed config/ds_pretrain.json \
  --max_length 600 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --weight_decay 0.2 \
  --warmup_steps 300 \
  --lr_scheduler_type "constant_with_warmup" \
  --lora_r 64 --lora_alpha 32 --lora_dropout 0.2 --lora_bias "none" \
  --mixed_precision fp16
```

#### Finetuning

```bash
# Using DeepSpeed
deepspeed --num_gpus=4 train.py \
  --task finetune \
  --model_name_or_path <BASE_OR_PRETRAINED> \
  --train_file <PATH/TO/train.jsonl> \
  --validation_file <PATH/TO/valid.jsonl> \
  --output_dir outputs/finetune \
  --deepspeed config/ds_finetune.json \
  --max_length 600 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.2 \
  --warmup_steps 300 \
  --lr_scheduler_type "constant_with_warmup" \
  --lora_r 64 --lora_alpha 32 --lora_dropout 0.2 --lora_bias "none" \
  --mixed_precision fp16
```

Scripted equivalents (if you use the example script names):

```bash
bash scripts/run_pretrain.sh
bash scripts/run_finetune.sh
```

> **Hardware reference:** Examples below assume `4 × A100 40GB`. If you have fewer GPUs or less memory, reduce batch sizes or enable activation checkpointing in your DeepSpeed config.

### 2.4 Inference

Two modes are exposed in `infer.py`:

- `regeneration`: free-form generation
- `qa`: question answering

#### Regeneration

```bash
python infer.py \
  --mode regeneration \
  --model_name_or_path outputs/finetune \
  --input_file <PATH/TO/prompts.jsonl> \
  --output_file inference/regeneration_outputs.jsonl \
  --max_new_tokens 256 \
  --temperature 0.7 \
  --top_p 0.9
```

#### QA

```bash
python infer.py \
  --mode qa \
  --model_name_or_path outputs/finetune \
  --input_file <PATH/TO/qa_inputs.jsonl> \
  --output_file inference/qa_outputs.jsonl \
  --max_new_tokens 256 \
  --temperature 0.0
```

Scripted versions:

```bash
bash scripts/run_infer_regeneration.sh
bash scripts/run_infer_qa.sh
```

## 3 Hyperparameters

### 3.1 Pretraining

| **Hyperparameter**                          | **Assignment**                                      |
| ------------------------------------------- | --------------------------------------------------- |
| learning rate                               | `1e-5`                                              |
| lr scheduler type                           | `constant with warmup`                              |
| warmup steps                                | `300`                                               |
| weight decay                                | `0.2`                                               |
| overall batch size                          | `16`                                                |
| optimizer                                   | `AdamW`                                             |
| epochs                                      | `3`                                                 |
| LoRA layers                                 | `all linear layers`                                 |
| LoRA r                                      | `64`                                                |
| LoRA alpha                                  | `32`                                                |
| LoRA dropout                                | `0.2`                                               |
| LoRA bias                                   | `None`                                              |
| mixed-precision                             | `fp16`                                              |
| GPU                                         | `4 × A100 40GB`                                     |
| max context length                          | `600`                                               |
| λ in Eq. (pretraining objective)            | `1e-4`                                              |
| policy ratio r                              | randomly chosen from `{1, 5, 10, 20, 50}` per batch |
| maximum number of compressed tokens `k_max` | `8`                                                 |

### 3.2 Finetuning

| **Hyperparameter**                          | **Assignment**         |
| ------------------------------------------- | ---------------------- |
| learning rate                               | `1e-6`                 |
| lr scheduler type                           | `constant with warmup` |
| warmup steps                                | `300`                  |
| weight decay                                | `0.2`                  |
| overall batch size                          | `16`                   |
| optimizer                                   | `AdamW`                |
| epochs                                      | `1`                    |
| LoRA layers                                 | `all linear layers`    |
| LoRA r                                      | `64`                   |
| LoRA alpha                                  | `32`                   |
| LoRA dropout                                | `0.2`                  |
| LoRA bias                                   | `None`                 |
| mixed-precision                             | `fp16`                 |
| GPU                                         | `4 × A100 40GB`        |
| max context length                          | `600`                  |
| policy ratio r                              | `10`                   |
| maximum number of compressed tokens `k_max` | `8`                    |

