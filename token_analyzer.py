import transformers
import json
import argparse
from typing import Dict, List, Any
from tqdm import tqdm

def can_be_hangul_utf8(byte_or_sequence):
    if isinstance(byte_or_sequence, int):
        # 단일 바이트 체크
        return (0xEA <= byte_or_sequence <= 0xED) or (0x80 <= byte_or_sequence <= 0xBF)
    elif len(byte_or_sequence) == 1:
        # 1바이트 시퀀스 체크
        return can_be_hangul_utf8(byte_or_sequence[0])
    elif len(byte_or_sequence) == 2:
        # 2바이트 시퀀스 체크
        return (0x80 <= byte_or_sequence[0] <= 0xBF) and (0x80 <= byte_or_sequence[1] <= 0xBF)
    elif len(byte_or_sequence) >= 3:
        # 3바이트 이상 시퀀스 체크
        return is_complete_hangul_utf8(byte_or_sequence[:3])
    else:
        return False


def is_complete_hangul_utf8(byte_sequence):
    if len(byte_sequence) != 3:
        return False

    first_byte, second_byte, third_byte = byte_sequence

    # 기본 범위 체크
    if not (0xEA <= first_byte <= 0xED):
        return False
    if not (0x80 <= second_byte <= 0xBF):
        return False
    if not (0x80 <= third_byte <= 0xBF):
        return False

    # 세부 범위 체크
    if first_byte == 0xEA:
        return second_byte >= 0xB0
    elif first_byte == 0xED:
        return second_byte <= 0x9F

    return True


def is_special_char_token(token: str) -> bool:
    """특수문자로만 구성된 토큰인지 확인합니다."""
    return all(not (c.isalnum() or c.isspace()) for c in token)


def analyze_token_categories(model_id: str, min_token_id: int = 102) -> Dict[str, Any]:
    """토크나이저의 전체 vocabulary에 대해 각 카테고리별 토큰을 분석합니다."""

    print(model_id)
    # 토크나이저 로드
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()

    max_token_id = len(vocab.values())

    # 각 카테고리별 token_id 집합
    pure_english_ids = set()
    english_containing_ids = set()
    hangul_possible_ids = set()
    complete_hangul_ids = set()
    special_char_ids = set()

    # 102 ~ 128000-1 토큰 ID만 포함
    all_token_ids = {token_id for token_id in vocab.values() if min_token_id < token_id < max_token_id}



    print(f"Analyzing tokens (ID <= {max_token_id})...")
    for token, token_id in tqdm(vocab.items()):
        # max_token_id보다 큰 토큰은 건너뛰기
        if token_id > max_token_id:
            continue

        try:
            # 토큰의 바이트 표현 얻기
            token_bytes = tokenizer.decode([token_id]).encode('utf-8')

            # 1. 영문 토큰 분석
            english_bytes = sum(1 for b in token_bytes if (0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A))
            if english_bytes > 0:
                #english_containing_ids.add(token_id)
                if all((0x41 <= b <= 0x5A) or (0x61 <= b <= 0x7A) for b in token_bytes):
                    pure_english_ids.add(token_id)

            # 2. 한글 분석
            # 완성형 한글 먼저 체크
            for i in range(len(token_bytes) - 2):
                seq = token_bytes[i:i + 3]
                if len(seq) == 3 and is_complete_hangul_utf8(seq):
                    complete_hangul_ids.add(token_id)
                    break

            # 한글 가능성 체크
            for i in range(len(token_bytes)):
                remaining_sequence = token_bytes[i:]
                if can_be_hangul_utf8(remaining_sequence):
                    hangul_possible_ids.add(token_id)
                    break

            # 3. 특수문자 토큰 체크
            if is_special_char_token(token):
                special_char_ids.add(token_id)

        except Exception as e:
            print(f"Error analyzing token {token} (ID: {token_id}): {str(e)}")
            continue

    # 모든 카테고리에 속한 토큰 ID
    categorized_ids = (pure_english_ids | english_containing_ids |
                       hangul_possible_ids | complete_hangul_ids |
                       special_char_ids)

    print(f"categorized_ids {len(categorized_ids)}")
    token_list = []
    for token_id in sorted(categorized_ids):
        token_list.append(str(token_id))
    print(f"len(token_list) {len(token_list)}")
    f = open("categorized_token_ids.txt", "wt")
    ids_string = ",".join(token_list)
    f.write(f"token_bias = [{ids_string}]")
    f.close()


    # 어떤 카테고리에도 속하지 않는 토큰 ID
    uncategorized_ids = all_token_ids - categorized_ids

    return {
        'model_id': model_id,
        'max_token_id': max_token_id,
        'vocab_size': len(all_token_ids),  # max_token_id 이하의 토큰 수
        'statistics': {
            'total_tokens': len(all_token_ids),
            'pure_english': len(pure_english_ids),
            'english_containing': len(english_containing_ids),
            'hangul_possible': len(hangul_possible_ids),
            'complete_hangul': len(complete_hangul_ids),
            'special_char': len(special_char_ids),
            'uncategorized': len(uncategorized_ids)
        },
        'token_ids': {
            'pure_english': sorted(list(pure_english_ids)),
            'english_containing': sorted(list(english_containing_ids)),
            'hangul_possible': sorted(list(hangul_possible_ids)),
            'complete_hangul': sorted(list(complete_hangul_ids)),
            'special_char': sorted(list(special_char_ids)),
            'uncategorized': sorted(list(uncategorized_ids))
        }
    }


def print_uncategorized_tokens(model_id: str, token_ids: List[int], max_tokens: int = 20):
    """미분류 토큰들을 출력합니다."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    print(f"uncategorized_ids {len(token_ids)}")
    token_list = []
    for token_id in sorted(token_ids):
        token_list.append(str(token_id))
    print(f"len(token_list) {len(token_list)}")
    f = open("uncategorized_token_ids.txt", "wt")
    ids_string = ",".join(token_list)
    f.write(f"token_bias = [{ids_string}]")
    f.close()


    print(f"\n=== 미분류 토큰 예시 (최대 {max_tokens}개) ===")
    for token_id in sorted(token_ids)[:max_tokens]:
        token = tokenizer.decode([token_id])
        token_bytes = tokenizer.decode([token_id]).encode('utf-8')
        print(f"Token ID: {token_id:6d} | Token: {token:20s} | Bytes: {' '.join(hex(b) for b in token_bytes)}")


def save_analysis_results(analysis_result: Dict[str, Any], output_file: str = 'token_category_analysis.json'):
    """분석 결과를 JSON 파일로 저장합니다."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)


def print_analysis_summary(analysis_result: Dict[str, Any]):
    """분석 결과의 주요 통계를 출력합니다."""
    stats = analysis_result['statistics']
    print(f"\n=== 토크나이저 카테고리 분석 결과 ===")
    print(f"모델: {analysis_result['model_id']}")
    print(f"분석 대상 최대 토큰 ID: {analysis_result['max_token_id']}")
    print(f"분석 대상 토큰 수: {stats['total_tokens']:,}")
    print(f"순수 영문 토큰 수: {stats['pure_english']:,}")
    #print(f"영문 포함 토큰 수: {stats['english_containing']:,}")
    print(f"한글 가능성 토큰 수: {stats['hangul_possible']:,}")
    print(f"완성형 한글 포함 토큰 수: {stats['complete_hangul']:,}")
    print(f"특수문자 토큰 수: {stats['special_char']:,}")
    print(f"미분류 토큰 수: {stats['uncategorized']:,}")


def token_analysis(model_id: str, output_file: str = 'token_category_analysis.json'):

    # 전체 분석 실행
    analysis_result = analyze_token_categories(model_id)

    # 결과 저장
    save_analysis_results(analysis_result, output_file)

    # 통계 출력
    print_analysis_summary(analysis_result)

    # 미분류 토큰 출력
    print_uncategorized_tokens(model_id, analysis_result['token_ids']['uncategorized'])


def main():
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(
        description="LLaMA Token Editor - 언어 모델 토크나이저 분석 및 가중치 조정 도구")

    parser.add_argument('--model_id', type=str, required=True,
                        help="분석할 모델의 경로 또는 허깅페이스 ID")
    parser.add_argument('--min_token_id', type=int, default=102,
                        help="분석할 최소 토큰 ID (기본값: 102)")
    parser.add_argument('--output_file', type=str, default='token_category_analysis.json',
                        help="분석 결과를 저장할 JSON 파일 경로 (기본값: token_category_analysis.json)")

    # 인자 파싱
    args = parser.parse_args()

    # 토큰 분석 실행
    token_analysis(args.model_id, args.output_file)


if __name__ == "__main__":
    main()
