from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pathlib

# 현재 디렉터리에 캐시 폴더 생성
CACHE_DIR = pathlib.Path(__file__).parent / "model_cache"
CACHE_DIR.mkdir(exist_ok=True)

# 환경 변수에서 토큰 가져오기
token = os.getenv("HUGGING_FACE_TOKEN")
if token is None:
    raise ValueError("Please set the HUGGING_FACE_TOKEN environment variable")

def load_or_download_model():
    model_path = CACHE_DIR / "model"
    tokenizer_path = CACHE_DIR / "tokenizer"
    
    # 캐시된 모델이 있는지 확인
    if model_path.exists() and tokenizer_path.exists():
        print("캐시된 모델을 불러오는 중...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    else:
        print("모델을 다운로드하는 중...")
        model = AutoModelForCausalLM.from_pretrained(
            "Adansonia/internal_audit_16bit",
            torch_dtype="auto",
            device_map="auto",
            token=token
        )
        tokenizer = AutoTokenizer.from_pretrained("Adansonia/internal_audit_16bit", token=token)
        
        # 모델과 토크나이저 저장
        print("모델을 캐시에 저장하는 중...")
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(tokenizer_path))
        
    return model, tokenizer

# 모델과 토크나이저 로드
model, tokenizer = load_or_download_model()

def generate_response(prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 예시 사용 코드
if __name__ == "__main__":
    while True:
        # 사용자로부터 입력 받기
        user_input = input("질문을 입력하세요 (종료하려면 'quit' 입력): ")
        
        # 종료 조건
        if user_input.lower() == 'quit':
            print("프로그램을 종료합니다.")
            break
        
        # 응답 생성
        try:
            response = generate_response(user_input)
            print("\n응답:", response)
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"오류가 발생했습니다: {str(e)}")