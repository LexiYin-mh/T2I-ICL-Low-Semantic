

<h1 align="center"> <p>Enhance Multimodal In-Context Learning for Text-to-Image Generation</p></h1>

This final project is based on this github project: https://github.com/UW-Madison-Lee-Lab/CoBSAT. 


# Contents

- [Step 1: Set Up Environment](#step-1-set-up-environment)
- [Step 2: Download Dataset and Model Weights](#step-2-download-dataset-and-model-weights)
- [Step 3: Set up SEED-LLaMA](#step-3-set-up-seed-llama)
  - [Experimented Models](#experimented-models)

- [Step 4: Task 1 Fine-Tuning with CoT Reasoning](#step-4-task-1-fine-tuning-with-cot-reasoning)
  - [Fine-tuning model using CoT augmented dataset](#fine-tuning-model-using-cot-augmented-dataset)
    - [Fine-tuning Stage](#fine-tuning-stage)
    - [Inference using Fine-tuned Model](#inference-using-fine-tuned-model) 
  - [Evaluation Stage](#evaluation-stage)
- [Step 5: Task 2 - Structured Reasoning with Low-Semantic Inputs](#step-5-task-2-structured-reasoning-with-low-semantic-inputs)
  - [Low-Semantic Task Inference](#low-semantic-task-inference)
- [Challenges and Fixes](#challenges-and-fixes)
- [Reference](#reference)

# Step 1: Set Up Environment

To set up the environment for benchmarking MLLMs, please follow the following steps. This works for linux. 

1. Clone this repository and rename it as `cobsat`

   ```bash
   git clone --recurse-submodules https://github.com/LexiYin-mh/T2I-ICL-Low-Semantic
   mv CoBSAT cobsat
   cd cobsat
   ```

2. Install Packages 

  **Linux**

   ```bash
   # create the environment that works for most of the cases
   conda create -n cobsat python=3.8.18
   conda activate cobsat
   pip install torch==2.1.2 torchvision==0.16.2 
   pip install -r conda_env/default_requirements.txt
   
   # create the environment for llava to work (for evaluation)
   conda create -n llava python=3.10.13
   conda activate llava
   pip install --upgrade pip  # enable PEP 660 support
   pip install git+https://github.com/yzeng58/LLaVA/@a61aae093656922fe16ec2152b031dd1de72fe92
   pip install -r conda_env/llava_requirements.txt
   ```

3. Create `environment.py` in the `cobsat` directory. Note that many variables need you to config except `root_dir` on your own

   ```python
   # Configure the environment variables for the project
   
   import os
   root_dir = os.path.dirname(os.path.abspath(__file__))
   
   SEED_PROJECT_ROOT = f'{root_dir}/models/SEED'
   
   ###############
   # NEED UPDATE #
   ###############
   TRANSFORMER_CACHE = '/data/.cache/huggingface/hub' 
   
   # WANDB Logging https://wandb.ai/site
   WANDB_ENTITY = 'your-wandb-entity'
   WANDB_PROJECT = 'your-wandb-project'
   ```

# Step 2: Download Dataset and Model Weights

1. Download the images and their corresponding descriptions of our dataset.

   ```bash
   wget "https://huggingface.co/datasets/yzeng58/CoBSAT/resolve/main/datasets.zip"
   ```
2. Uncompress the `datasets.zip` file via `unzip datasets.zip` and move the `datasets` folder into your `cobsat` folder. 

Up to now, the structure of your `cobsat` folder should look like this.

```
.
├── ...          
├── datasets                # download the dataset in this step
├── load_models
│   ├── call_emu.py
│   ├── call_emu2.py
│   ├── call_gill.py
│   ├── call_gpt.py
│   ├── call_llava.py       # LLaVA-1.5
│   ├── call_llava16.py     # LLaVA-NeXT 
│   ├── call_qwen.py
│   ├── call_seed.py
│   ├── call_gemini.py
│   ├── call_claude.py
│   ├── call_your_model.py  # [optional] create python file to load the model you want to evaluate
│   └── ... 
├── models                  
│   ├── SEED                
│   ├── gill                
│   ├── Emu                 
│   │   └── Emu1 
│   ├── LLaVA               
│   ├── Qwen-VL    
│   ├── Gemini
│   ├── Claude   
│   ├── OwnModel            # [optional] input your own model folder
│   └── ...
├── ...
├── environment.py          # follow the instruction above to create this file
├── load_model.py           # [optional] add your own model                
└── ...
```

# Step 3: Set up SEED-LLaMA

3. Follow the instruction in [SEED-LLaMA](https://github.com/AILab-CVC/SEED), install model weights. 

Besides, this repository also allow testing for other MLLMS listed on the directory below. 

## Experimented Models

- [x] [SEED-LLaMA](https://arxiv.org/abs/2310.01218)
  * Image Generation
  * Text Generation
  * Fine-Tuning
- [x] [Emu](https://arxiv.org/abs/2307.05222)
  * Image Generation
  * Text Generation

# Step 4: Task 1 Fine-Tuning with CoT Reasoning

## Baseline Inference

```bash
conda activate cobsat

python inference_icl.py \
--model seed \
--prompt_type default \
--gen_mode image \
--shot 2 4 \    # change the amount of shots
--seed 123 \
--device cuda \
--task_id 1 2 3 \  # change task id for testing more tasks (total 10)
--overwrite 0 \
--finetuned_model 0 \
--data_mode default \
--ft_mode all \
--eval_task_theme '' \
--low_semantic 0 # change it to be 1 if you are testing task 2
```

**Parameter Descriptions**

- **`model`**: Specifies the model for making the inference.
- **`shot`**: Defines the number of demonstration examples included in each training prompt.
- **`prompt_type`**: Selects the type of prompt to use. Available options include:
  * `default`: The standard prompt design as described in our paper.
- **`gen_mode`**: Determines the output mode of the model, with two options:
  * `image`: The model generates an image output.
  * `text`: The model generates textual descriptions for the next image.
- **`seed`**: An integer used to set the random seed for reproducibility.
- **`device`**: Specifies the computing device for the experiments. The default value is `cuda`, which utilizes a single GPU.
- **`task_id`**: Identifies the task being performed. By default, all ten tasks are executed. Detailed information about each task can be found in `configs.py` under the definition of `task_dataframe`, as well as in our paper.
- **`overwrite`**: Determines whether to reuse existing results or overwrite them. This is applicable when results have already been saved.
- **`finetuned_model`**: Indicates whether to use a finetuned model. If enabled, the finetuned model must be stored beforehand by executing `finetune_icl.py`, and the `data_mode` should be set to `ft_test`. 
- **`data_mode`**: Offers two options: `default` and `ft_test`. In `ft_test` mode, the dataset is divided into training and testing sets, with only the testing set being utilized.
- **`ft_mode`**: The fine-tuning mode used in the experiment, with two options:
  * `all`: fine-tune on subsets of all tasks
  * `leave_one_out`: fine-tune on entire set of other four themed-tasks
- **`eval_task_theme`**: The theme will be evaluated on (the theme that is excludede in fine-tuning). Default is empty string `''`. Only use it when `ft_mode` set to be `leave_one_out`.
- **`low_semantic`**: Indicates whether to use low-semantic input for testing ICL ability of the model. 

The generated outputs will be stored in `results/exps/` by default or `results/ft` if `finetuned_model` is set to `True`.

**Screenshots**


The following image shows the scenario when running the baseline model.
![image](https://github.com/user-attachments/assets/726463ba-3955-4e13-bb44-fc114e4b9dc8)

![image](https://github.com/user-attachments/assets/a7871a95-b030-47a8-990b-389d1c9302ef)

![image](https://github.com/user-attachments/assets/f756e42f-fcc4-4c85-996a-0dcd3e3fc144)



## Fine-tuning model using CoT augmented dataset

### Fine-tuning Stage

```bash
conda activate cobsat

python finetune_icl.py \
--model seed \
--shot 2 \ 							
--prompt_type default \
--gen_mode text \
--ft_mode leave_one_out \
--eval_task_theme color
```

**Parameter Descriptions**

  - **`model`**: Specifies the model for fine-tuning. 
  - **`shot`**: Defines the number of demonstration examples included in each training prompt.
  - **`prompt_type`**: Selects the type of prompt to use. Available options include:
    - `default`: The standard prompt design as described in our paper. Currently only supports 'default' type since it's designed for CoT fine-tuning.
  - **`gen_mode`**: Determines the output mode of the model, with two options:
    - `image`: The model generates an image output.
  - **`ft_mode`**: The fine-tuning mode used in the experiment, with two options:
    - `all`: fine-tune on subsets of all tasks
    - `leave_one_out`: fine-tune on entire set of other four themed-tasks
  - **`eval_task_theme`**: The theme will be evaluated on (the theme that is excludede in fine-tuning). Default is empty string `''`. Only use it when `ft_mode` set to be `leave_one_out`.

The checkpoints of fine-tuned models will be stored in `ft_models/`.

### Inference using Fine-tuned Model
```bash
conda activate cobsat

python inference_icl.py \
--model seed \
--prompt_type default \
--gen_mode image \
--shot 2 \    # change the amount of shots
--seed 123 \
--device cuda \
--task_id 1 2 3 \  # change task id for testing more tasks (total 10)
--overwrite 0 \
--finetuned_model 0 \
--data_mode default \
--ft_mode leave_one_out \
--eval_task_theme color \ # change task theme if required
--low_semantic 0 # change it to be 1 if you are testing task 2
```

**Parameter Descriptions**

- **`model`**: Specifies the model for making the inference.
- **`shot`**: Defines the number of demonstration examples included in each training prompt.
- **`prompt_type`**: Selects the type of prompt to use. Available options include:
  * `default`: The standard prompt design as described in our paper.
- **`gen_mode`**: Determines the output mode of the model, with two options:
  * `image`: The model generates an image output.
  * `text`: The model generates textual descriptions for the next image.
- **`seed`**: An integer used to set the random seed for reproducibility.
- **`device`**: Specifies the computing device for the experiments. The default value is `cuda`, which utilizes a single GPU.
- **`task_id`**: Identifies the task being performed. By default, all ten tasks are executed. Detailed information about each task can be found in `configs.py` under the definition of `task_dataframe`, as well as in our paper.
- **`overwrite`**: Determines whether to reuse existing results or overwrite them. This is applicable when results have already been saved.
- **`finetuned_model`**: Indicates whether to use a finetuned model. If enabled, the finetuned model must be stored beforehand by executing `finetune_icl.py`, and the `data_mode` should be set to `ft_test`. 
- **`data_mode`**: Offers two options: `default` and `ft_test`. In `ft_test` mode, the dataset is divided into training and testing sets, with only the testing set being utilized.
- **`ft_mode`**: The fine-tuning mode used in the experiment, with two options:
  * `all`: fine-tune on subsets of all tasks
  * `leave_one_out`: fine-tune on entire set of other four themed-tasks
- **`eval_task_theme`**: The theme will be evaluated on (the theme that is excludede in fine-tuning). Default is empty string `''`. Only use it when `ft_mode` set to be `leave_one_out`.
- **`low_semantic`**: Indicates whether to use low-semantic input for testing ICL ability of the model. 

The generated outputs will be stored in `results/exps/` by default or `results/ft` if `finetuned_model` is set to `True`.




## Evaluation Stage

```bash
# activate the environment for using LLaVA (since LLaVA is our evaluation model)
conda activate llava

python evaluation_icl.py \
--model seed \
--prompt_type default \
--eval_mode image \
--task_id 1 2 3 \
--shot 2 4 \
--device cuda \
--seed 123 \
--wandb 1 \
--overwrite 0 \
--finetuned_model 0 \
--data_mode default
```

**Parameter Descriptions**

- **`model`**: Specifies the model for making the inference. The supported models include `seed` (SEED-LLaMA), `gill` (GILL), `emu`  (Emu), `emu2` (Emu2), `gpt4v` (GPT-4V), `llava` (LLaVA-1.5), `llava16` (LLaVA-1.6/LLaVA-NeXT), `gemini` (Gemini), `claude` (Claude) and `qwen` (Qwen-VL).  
- **`shot`**: Defines the number of demonstration examples included in each training prompt.
- **`prompt_type`**: Selects the type of prompt to use. Available options include:
  - `default`: The standard prompt design as described in our paper.
  - `misleading`: Introduces misleading information in the textual input of each demonstration, as detailed in the appendix.
  - `cot` (Chain of Thought): Incorporates multi-step inference prompts, prompting the model to generate reasoning steps ("let's think step by step") before the final output.
  - `exact`: Directly provides the ground truth label as the textual input.
  - `caption`: Replaces images in the prompt with their corresponding captions.
  - `instruct`: Adds an additional sentence explicitly stating the relationship between textual input and visual output in each demonstration.
- **`eval_mode`**: Specifies the type of model output to be evaluated. Available options are:
  - `image`: Evaluates the image output generated by the model.
  - `text`: Evaluates the textual descriptions for the subsequent image as generated by the model.
- **`seed`**: An integer used to set the random seed for reproducibility.
- **`device`**: Specifies the computing device for the experiments. The default value is `cuda`, which utilizes a single GPU.
- **`task_id`**: Identifies the task being performed. By default, all ten tasks are executed. Detailed information about each task can be found in `configs.py` under the definition of `task_dataframe`, as well as in our paper.
- **`overwrite`**: Determines whether to reuse existing results or overwrite them. This is applicable when results have already been saved.
- **`finetuned_model`**: Indicates whether to use a finetuned model. If enabled, the finetuned model must be stored beforehand by executing `finetune_icl.py`.
- **`data_mode`**: Offers two options: `default` and `ft_test`. In `ft_test` mode, the dataset is divided into training and testing sets, with only the testing set being utilized.
- **`eval_mllm`**: The multimodal large language model used for evaluating the generated images (descriptions). The supported mllms include `llava`, `gemini`, and `qwen`.
- **`api_key`**: Indicate which key to use. In `environment.py`, you should have already chose the name for your api_key for the model you are going to use.
- **`ft_mode`**: The fine-tuning mode used in the experiment, with two options:
  * `all`: fine-tune on subsets of all tasks
  * `leave_one_out`: fine-tune on entire set of other four themed-tasks
- **`eval_task_theme`**: The theme will be evaluated on (the theme that is excludede in fine-tuning). Default is empty string `''`. Only use it when `ft_mode` set to be `leave_one_out`.


The evaluation results will be stored in `results/evals/` by default or `results/ft` if `finetuned_model` is set to `True`. If `wandb` is `True`, you can also view the evaluation results in your wandb board. 

**Screenshots**


The following image shows the scenario when running the fine-tuned model.
<img width="1477" alt="image" src="https://github.com/user-attachments/assets/8cd9e67b-8e18-4e0b-8a49-6c4cb63fa267" />



# Step 5: Task 2 - Structured Reasoning with Low-Semantic Inputs

## Low-Semantic Task Inference

```bash
conda activate cobsat

python inference_icl.py \
--model seed \
--prompt_type default \
--gen_mode image \
--shot 4 \ # task2-lowsemantic only support 4-shot, 8-shot, 12-shot, 16-shot
--seed 123 \
--device cuda \
--task_id 1 2 3 \  # change task id for testing more tasks (total 10)
--overwrite 0 \
--finetuned_model 1 \
--data_mode default \
--ft_mode all \
--eval_task_theme '' \
--low_semantic 1 # change it to be 1 if you are testing task 2
```

**Parameter Descriptions**

  - **`model`**: Specifies the model for fine-tuning. 
  - **`shot`**: Defines the number of demonstration examples included in each training prompt.
  - **`prompt_type`**: Selects the type of prompt to use. Available options include:
    - `default`: The standard prompt design as described in our paper. Currently only supports 'default' type since it's designed for CoT fine-tuning.
  - **`gen_mode`**: Determines the output mode of the model, with two options:
    - `image`: The model generates an image output.
  - **`ft_mode`**: The fine-tuning mode used in the experiment, with two options:
    - `all`: fine-tune on subsets of all tasks
    - `leave_one_out`: fine-tune on entire set of other four themed-tasks
  - **`eval_task_theme`**: The theme will be evaluated on (the theme that is excludede in fine-tuning). Default is empty string `''`. Only use it when `ft_mode` set to be `leave_one_out`.


The checkpoints of fine-tuned models will be stored in `ft_models/`.


**Screenshots**


The following image shows the scenario when running the fine-tuned model.


![image](https://github.com/user-attachments/assets/781544d1-4128-4278-8037-3326441db1bc)
![image](https://github.com/user-attachments/assets/b378e925-e223-48b8-99aa-a56529aaaff7)


# Challenges and Fixes

During the replication of the CoBSAT project, we faced several challenges that hindered smooth execution. These included:

* **Dependency Issues**: We faced recurring dependency problems while attempting to run the project on various virtual machines (VMs), including Google Colab and the Google Cloud VM without Deep Learning Image. The issues primarily stemmed from mismatches in package versions and missing libraries. Among the tested environments, the Google Colab Deep Learning VM was the most stable and compatible for running the project.

* **GPU Limitations**: Our project required substantial GPU resources, as we were working with the SEED-LLaMA-14b model (and, as a fallback, SEED-LLaMA-8b). However, we had limited access to high-performance GPUs apart from the Nvidia T4, which struggles to load models larger than 7B parameters without additional optimizations such as quantization or model parallelism. To overcome this constraint, we transitioned to using the Lambda Labs VM service, which provided the necessary computational resources.

* **Code Bugs in the Project Repository**: The original CoBSAT repository contained several bugs that further complicated the execution.
    * Model Compatibility: Certain model configurations, such as GPT-4V, were not necessary for our experiments. To address this, we commented out irrelevant code to prevent execution errors.
    * Tensor Shape Mismatches: Some parts of the code produced errors due to incorrect image tensor shapes, which caused downstream operations to fail. We resolved these issues by implementing tensor truncation and padding mechanisms.
    * Unresolvable Low-Frequency Errors: Occasionally, errors occurred that we could not address due to their origins in lower-level library calls. These errors had a frequency of less than 1 in 1,000 cases. To mitigate their impact, we implemented a mechanism to skip problematic operations instead of halting the entire program.

By addressing these challenges systematically, we ensured that the project could run effectively on our chosen infrastructure, enabling successful replication of the baseline experiments.


# Reference

```tex
@article{zeng2024can,
  title={Can MLLMs Perform Text-to-Image In-Context Learning?},
  author={Zeng, Yuchen and Kang, Wonjun and Chen, Yicong and Koo, Hyung Il and Lee, Kangwook},
  journal={arXiv preprint arXiv:2402.01293},
  year={2024}
}
```

