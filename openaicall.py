from openai import OpenAI
import json

MODEL_NAME: str = "meta-llama/llama-4-scout"
API_KEY: str = (
    "YOUR_API_KEY"
)
# (3) 한글 토큰 ID 불러오기
with open("token_category_analysis.json", "r", encoding="utf-8") as f:
    categorized_ids = json.load(f)['token_ids']
    hangul_ids = list(set(categorized_ids['hangul_possible'] + categorized_ids['complete_hangul']))
    # hangul_

logit_bias_list = hangul_ids
logit_bias = {str(token) :150 for token in logit_bias_list}

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# (4) Generate w/ LogitsProcessor
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "철수와 영희는 "}
    ],
    logit_bias=logit_bias
)
print("=== Logit Bias 적용 ===")

print(response.choices[0].message.content)

# (5) Generate w/o/ LogitsProcessor
response_no_processor = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "철수와 영희는 "}
    ],
)


print("=== Logit Bias 미적용 ===")
print(response_no_processor.choices[0].message.content)