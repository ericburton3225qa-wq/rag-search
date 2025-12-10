from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "microsoft/Phi-3.5-mini-instruct"  # Or "google/gemma-2-2b-it" for even smaller/faster
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
print("Model downloaded and ready!")