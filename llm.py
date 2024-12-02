from transformers import AutoModel, AutoTokenizer
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
        self._model = AutoModel.from_pretrained(
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

    def process_text(self, text):
        # 입력 텍스트를 토큰화
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # 모델 추론
        with torch.no_grad():  # 그래디언트 계산 비활성화
            outputs = self._model(**inputs)
        
        # 모델 출력을 디코딩
        last_hidden_state = outputs.last_hidden_state
        # 첫 번째 토큰([CLS])의 임베딩을 사용
        sentence_embedding = last_hidden_state[:, 0, :]
        
        print(f"\n입력 텍스트: {text}")
        print(f"임베딩 shape: {sentence_embedding.shape}")
        print(f"출력 텐서의 일부: {sentence_embedding[0][:5]}")  # 처음 5개 값만 출력
        
        return sentence_embedding

def interactive_test():
    model = AdansoniaModel.get_instance()
    
    print("\n=== Adansonia 모델 테스트 ===")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    
    while True:
        user_input = input("\n텍스트를 입력하세요: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("프로그램을 종료합니다.")
            break
        
        if not user_input:
            print("텍스트를 입력해주세요!")
            continue
        
        try:
            _ = model.process_text(user_input)
        except Exception as e:
            print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    interactive_test()