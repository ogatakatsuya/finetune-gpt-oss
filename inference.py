import weave
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

weave.init("finetune-gpt-oss")

model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_kwargs = dict(attn_implementation="eager", torch_dtype="auto", use_cache=True, device_map="auto")
base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).cuda()
peft_model_id = "gpt-oss-20b-finetuned-JaFIn"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()

@weave.op
def run_inference(prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    gen_kwargs = {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.6}
    output_ids = model.generate(input_ids, **gen_kwargs)
    response = tokenizer.batch_decode(output_ids)[0]
    return response

prompts = [
    "特例国債（赤字国債）とは何ですか？建設国債とどのように異なるか、分かりやすく説明してください。",
    "NISA（少額投資非課税制度）について、その概要とメリットを簡潔に教えてください。",
]
for prompt in prompts:
    out = run_inference(prompt)
    print(f"Prompt: {prompt}\nResponse: {out}\n")