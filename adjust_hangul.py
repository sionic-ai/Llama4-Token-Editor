import torch
from transformers import (
    AutoTokenizer,
    Llama4ForConditionalGeneration,
    LogitsProcessorList
)
import json

class TokenBiasLogitsProcessor:
    def __init__(self, token_ids, bias_value):
        self.token_ids = token_ids
        self.bias_value = bias_value
    
    def __call__(self, input_ids, scores):
        for token_id in self.token_ids:
            scores[:, token_id] += self.bias_value
        return scores


model_name = "unsloth/Llama-4-Scout-17B-16E-Instruct"

# (1) 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name)

# (2) model.from_pretrained 시점에 device_map, torch_dtype 등 설정
# - device_map="auto": GPU/CPU 메모리를 자동으로 사용
# - torch_dtype=torch.float16: 메모리를 절감하기 위해 FP16 사용
model = Llama4ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="balanced",
    torch_dtype=torch.float16
)

# (3) 한글 토큰 ID 불러오기
with open("token_category_analysis.json", "r", encoding="utf-8") as f:
    categorized_ids = json.load(f)
    hangul_ids = list(set(categorized_ids['hangul_possible'] + categorized_ids['complete_hangul']))


token_bias = 1.2
prompt = "철수와 영희는"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

token_bias_processor = TokenBiasLogitsProcessor(hangul_ids, token_bias)
logits_processor = LogitsProcessorList([token_bias_processor])

# (4) Generate w/ LogitsProcessor
output = model.generate(
    input_ids=input_ids,
    logits_processor=logits_processor,
    max_length=100,
    do_sample=True,
    temperature=0.7
)

# (5) Generate w/o/ LogitsProcessor
output_no_processor = model.generate(
    input_ids=input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7
)
# (6) 결과 출력
print("With LogitsProcessor:")
generated_text_with_processor = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text_with_processor)

print("\nWithout LogitsProcessor:")
generated_text_without_processor = tokenizer.decode(output_no_processor[0], skip_special_tokens=True)
print(generated_text_without_processor)

