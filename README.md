# Llama4 Token Editor

LLaMA Token Editor는 대형 언어 모델(LLM)의 토크나이저를 분석하고, 특정 범주의 토큰 가중치를 조정할 수 있는 도구입니다. 이 도구는 주로 Llama, Qwen 계열 모델의 토큰화 방식을 분석하고, 한글 및 영어 토큰을 분석하고 일부 가중치를 조정하는 데 사용됩니다.

다음 모델의 토크나이저를 분석합니다.

한글 표현 가능성을 가진 토큰은 부분 BPE의 표현중 한국어를 표현할 가능성을 가진 바이트 부분 순서열을 다국어와 공유하는 토큰입니다.

완성형 한글 토큰은 온전한 한국어 토큰입니다. 

llama4의 토크나이저 구성이 한국어 표현 관점에서 기존 Llama3.3 과 QWEN 대비 충분히 개선되었음을 알 수 있습니다.


| 분석 항목 | Llama-4-Scout-17B | Llama-3.3-70B-Instruct | Qwen2.5-32B | Mistral-Small-3.1-24B-Base-2503 |
|------------|-------------------|------------------------|-------------|--------------------------------|
| 분석 대상 최대 토큰 ID | 201,135 | 128,256 | 151,665 | 131,072 |
| 분석 대상 토큰 수 | 201,032 | 128,153 | 151,562 | 130,969 |
| 순수 영문 토큰 수 | 36,668 | 28,158 | 27,376 | 22,447 |
| 한글 가능성 토큰 수 | 64,992 | 25,738 | 53,204 | 41,124 |
| 완성형 한글 포함 토큰 수 | 5,490 | 2,281 | 3,498 | 4,492 |
| 특수문자 토큰 수 | 2,509 | 2,588 | 2,579 | 1,500 |
| 미분류 토큰 수 | 97,009 | 71,806 | 68,592 | 65,958 |



## 주요 기능

- **토큰 카테고리 분석**: 모델의 전체 어휘(vocabulary)를 다양한 카테고리로 분류
  - 순수 영문 토큰
  - 한글 가능성 토큰
  - 완성형 한글 포함 토큰
  - 특수문자 토큰
  - 미분류 토큰

- **토큰 가중치 조정**: 분석된 카테고리에 따라 토큰의 가중치를 조정할 수 있음
  - `categorized_token_ids.txt`에 있는 토큰의 가중치 증가
  - `uncategorized_token_ids.txt`에 있는 토큰의 가중치 감소

## 지원 모델

현재 다음 모델들에 대한 분석이 테스트되었습니다:
- Llama-4-Scout-17B
- Llama-3.3-70B-Instruct
- Qwen2.5-32B

## 설치 방법

```bash
git clone https://github.com/yourusername/Llama4-Token-Editor.git
cd Llama4-Token-Editor
pip install -r requirements.txt
```

## 사용 방법

### 토큰 분석 실행

```bash
python token_analyzer.py --model_id "모델_경로_또는_이름"
```

매개변수:
- `--output_file`: 분석 결과를 저장할 JSON 파일 (기본값: token_category_analysis.json)

### 결과 파일

분석이 완료되면 다음 파일들이 생성됩니다:
1. `token_category_analysis.json`: 전체 분석 결과가 담긴 JSON 파일
2. `categorized_token_ids.txt`: 분류된 토큰 ID 목록 (가중치 상향 조정용)
3. `uncategorized_token_ids.txt`: 미분류 토큰 ID 목록 (가중치 하향 조정용)

### 토큰 가중치 조정 예시

토큰 가중치를 조정하려면 생성된 ID 목록 파일을 읽어들여 다음과 같이 사용할 수 있습니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch

# 특정 토큰에 바이어스를 추가하는 프로세서 구현
class TokenBiasLogitsProcessor:
    def __init__(self, token_ids, bias_value):
        self.token_ids = token_ids
        self.bias_value = bias_value
    
    def __call__(self, input_ids, scores):
        # 특정 토큰 ID에 바이어스 적용
        for token_id in self.token_ids:
            scores[:, token_id] += self.bias_value
        return scores

# 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained("your_model_path")
tokenizer = AutoTokenizer.from_pretrained("your_model_path")

# categorized_token_ids.txt에서 토큰 ID 로드
with open("categorized_token_ids.txt", "r") as f:
    content = f.read()
    # "token_bias = [103,104,...]" 형식에서 ID 목록만 추출
    ids_str = content.replace("token_bias = [", "").replace("]", "")
    categorized_ids = [int(id_str) for id_str in ids_str.split(",")]

# 바이어스 값 설정
token_bias = 1.2  # 가중치 상향 조정값

# 입력 프롬프트
prompt = "여기에 프롬프트를 입력하세요"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 토큰 바이어스 프로세서 생성
token_bias_processor = TokenBiasLogitsProcessor(categorized_ids, token_bias)
logits_processor = LogitsProcessorList([token_bias_processor])

# 생성 실행
output = model.generate(
    input_ids,
    logits_processor=logits_processor,
    max_length=100,
    do_sample=True,
    temperature=0.7
)

# 결과 출력
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## 분석 예시

Llama-4-Scout-17B-16E-Instruct 모델 분석 결과:
- 분석 대상 최대 토큰 ID: 201,135
- 분석 대상 토큰 수: 201,032
- 순수 영문 토큰 수: 	36,668
- 한글 가능성 토큰 수: 64,992
- 완성형 한글 포함 토큰 수: 	5,490
- 특수문자 토큰 수: 2,509
- 미분류 토큰 수: 97,009

## 한글 토큰 분석 원리

이 도구는 UTF-8 인코딩과 BPE 알고리즘을 역공학으로 분석하여 한글 문자를 식별합니다:
- `can_be_hangul_utf8()`: 바이트 또는 바이트 시퀀스가 한글의 일부가 될 수 있는지 확인
- `is_complete_hangul_utf8()`: 3바이트 시퀀스가 완성형 한글인지 확인

한글 UTF-8 표현:
- 한글 유니코드 범위: U+AC00 ~ U+D7A3 (가 ~ 힣)
- UTF-8 인코딩에서 첫 바이트: 0xEA ~ 0xED
- 이후 바이트: 0x80 ~ 0xBF

## 기여하기

버그 리포트나 기능 요청은 GitHub 이슈를 통해 제출해주세요. 풀 리퀘스트도 환영합니다!

## 라이선스

MIT License
