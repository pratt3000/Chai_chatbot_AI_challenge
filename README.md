# Brief overview (of best performing workflow)
I chose the researcher track for my work. My best performing pipeline is a 2 step process. SFT + DPO.

Best performing model `ELO: 1266` <br>
submission link: [link](https://console.chaiverse.com/models/pratt3000-mistral-nemo-_23899_v3) <br>
hf repo: `pratt3000/Mistral-Nemo-2407-LORA-chaidataset-base-dpobase4ep`

All scripts for best performing model are in `finetuning_scripts/best_performing_pipeline_scripts`<br>
More details here: `best_performing_pipeline_config.txt`

## Finetuning Step 1: SFT
This experiment fine-tunes the Mistral-Instruct model using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA), targeting improvements in open-domain conversation tasks, particularly within the roleplay and chitchat domains. The process incorporates structured multi-turn formatting, Flash Attention, and mixed-precision training to optimize both performance and resource utilization.

1. Model and Tokenizer Setup
The base model (Mistral-Instruct) is loaded with torch.bfloat16 precision and Flash Attention 2. This improves memory efficiency and throughput on high-end GPUs (I used A100s). A custom Jinja-based chat template was defined to emulate Mistral’s instruct format.

2. Parameter-Efficient Fine-Tuning (LoRA)
Fine-tuning is performed using LoRA via the peft library. The configuration (r=32, alpha=16, dropout=0.1) targets key transformer modules—q_proj, k_proj, v_proj, o_proj, and MLP layers

3. Training
SFTTrainer (from TRL) configured with:
    - Cosine LR Scheduler for gradual warmup and decay.
    - Gradient Accumulation to simulate larger batch sizes.
    - bfloat16 mixed-precision.
    - Paged AdamW optimizer (32-bit) for improved memory handling.

4. Post-training, the LoRA adapter is merged into the base model, producing a standalone, fully merged model in float16.

5. Lastly I performed hyperparameter tuning during training, produced multiple candidate models even if the offline eval was not as good because I figured ELO rating is a completely different way of evaluation and slightly lower performing models on offline test, could end up getting a good ELO rating for a veriety of reasons. I also considered using something like ChatGPT as a reward function, asking it to pick responses between candidate models and creating my own elo rating for finetuning, but that would have taken more time than I had to do this task.

### Data creation
I'm using multiple datasets for my experiment. All of the datasets are centered around roleplaying since thats the usecase I found most users at using Chai care about.

After going through about 15 datasets I was able to narrow it down to these ones. Somce of the things I filterd by were
1. Multiturn conversations + conversation length.
2. Manually inspected quality/type of responses to make sure they match with what is expected on Chai.
3. Took a sub sample of data from each and evaluated what is the average score for the conversations in the dataset.
    - I trained a reward model based on a roleplay conversation-user feedback data(`IlyaGusev/20231007_chai_prize_model_feedback_all`) that I found. The script can be found here - `finetuning_scripts/reward_model_training.ipynb`. (I found this dataset at the very end so I didnt have much time to work on the reward model, but this I could have probably used this data in many ways to improve overall model performance. )

The datasets I used were (these gave me about 15k conversation samples) (script - `/data_creation/create_15k_roleplay_data_ensemble.ipynb`)
1. "gpt-realm": "AlekseyKorshuk/gpt-roleplay-realm-chatml"
2. "zerofata": "zerofata/Roleplay-Anime-Characters"
3. "erotiquant": "openerotica/erotiquant3"
4. "hieunguyenminh": "hieunguyenminh/roleplay"


Another dataset that I found rather late, but was much better/bigger than these was `tiendung/chai`. This had a `thumbs_up` column which showed user preference for conversations, so I only took the data points with a `thumbs up`.

#### Optional reading (didnt get time to implement)
Some ideas I couldnt implement due to time constraint
1. Training a reward model using the thumbs_up column
    - Use it to rank responses for GRPO finetuning.
    - OR simply for filtering data.
2. Use the base model to generate bad completions
    -  Prompt your base model for alternative responses
    - Keep the human-labeled 'thumbs up' as chosen
    - Use the synthetic response as rejected (can generate multiple synthetic responses and use the above reward model to pick the worst one)


## Finetuning Step 2: DPO
I used the script (`finetuning_scripts/2_DPO_finetuning.ipynb`) to finetune the model using DPO. 
The training builds on a base model previously fine-tuned via supervised fine-tuning (SFT), and leverages LoRA, bfloat16.

### Data
The dataset I used was `chargoddard/chai-dpo` which has rejected and accepted responses from users in a roleplaying based scenario talking with a chatbot. I used those labels as positive and negative. It gave me 100k+ samples which helped quite a bit.

A custom formatting function is applied to transform each record into DPO-compatible input:
`prompt`: Concatenation of history messages, separated with eos_token.
`chosen`: The preferred response.
`rejected`: The less-preferred alternative (handles list or string input formats).



# Learnings
1. DPO, or RLHF in general helped much more than SFT in my experiments, especially for ELO scores. One thing I could have done better was concentrate on RLHF more, experiment more with `SIMPO, GRPO, etc` and construction of semi-synthetic datasets to do the same. I believe that could lead to significant gains. Especially SIMPO since I saw a few papers/articles mentioning it gives better performance when finetuned on smaller datasets.


# To run:
Create a huggingface key and create a .env file in the root folder. Paste it there like so
`hf_key=xxxxxxx`