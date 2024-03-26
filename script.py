import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
import re
from transformers import GenerationConfig
from time import perf_counter

model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def formatted_prompt(question)-> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant:"

def get_model_and_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

def generate_response(model, tokenizer, user_input):

  prompt = formatted_prompt(user_input)

  generation_config = GenerationConfig(penalty_alpha=0.6,do_sample = True,
      top_k=5,temperature=0.5,repetition_penalty=1.2,
      max_new_tokens=2048,pad_token_id=tokenizer.eos_token_id
  )
  start_time = perf_counter()

  inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

  outputs = model.generate(**inputs, generation_config=generation_config)
  response = tokenizer.decode(outputs[0], skip_special_tokens=True)
  assistant_response = response.partition("assistant")[2]
  output_time = perf_counter() - start_time
  return assistant_response, output_time


        
def main() : 
    model, tokenizer = get_model_and_tokenizer(model_id=model_id)
    print("Welcome to the Chatbot ! How can I help ? \n")
    print("Type 'exit' to end the conversation.")
    while True : 
        user_input = input("You : ")
        if user_input.lower() == "exit" : 
            print("Chatbot : Goodbye ! ")
            break
        response, output_time = generate_response(model, tokenizer, user_input)

        print("Chatbot:", response)
        print(f"Time taken for inference: {round(output_time,2)} seconds")
        


if __name__ == "__main__":
    torch.set_warn_always(False)
    main()
