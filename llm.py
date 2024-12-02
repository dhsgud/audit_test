from transformers import LlamaForCausalLM, AutoTokenizer
import os
from pathlib import Path
import torch

class AdansoniaModel:
    _instance = None
    _cache_dir = Path.home() / '.cache' / 'adansonia_model'
    _model = None
    _tokenizer = None

    def __init__(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        self._load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        print("Loading model from cache or downloading...")
        self._model = LlamaForCausalLM.from_pretrained(
            "Adansonia/internal_audit_16bit",
            cache_dir=str(self._cache_dir),
            local_files_only=False
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "Adansonia/internal_audit_16bit",
            cache_dir=str(self._cache_dir),
            local_files_only=False
        )
        print("Model loaded successfully!")

    def generate_text(self, prompt, max_length=1024):
        # 입력 텍스트를 토큰화
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # 텍스트 생성
        with torch.no_grad():
            outputs = self._model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.1,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        # 생성된 텍스트 디코딩
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text

def interactive_test():
    model = AdansoniaModel.get_instance()
    
    print("\n=== Adansonia 텍스트 생성 테스트 ===")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    
    while True:
        user_input = input("\n프롬프트를 입력하세요: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("프로그램을 종료합니다.")
            break
        
        if not user_input:
            print("텍스트를 입력해주세요!")
            continue
        
        try:
            generated_text = model.generate_text(user_input)
            print("\n생성된 텍스트:")
            print(generated_text)
        except Exception as e:
            print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    interactive_test()