from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
import os
from pathlib import Path
import torch

class AdansoniaLLM:
    def __init__(self, model_name="Adansonia/internal_audit_16bit", cache_dir=None):
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'adansonia_model'
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading {model_name}...")
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=False
        )
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            local_files_only=False
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print("Model loaded successfully!")

    def __call__(self, prompt, temperature=0.1, max_length=1024):
        """Ollama처럼 직접 호출 가능한 인터페이스"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Ollama처럼 모델 인스턴스 생성
    llm = AdansoniaLLM()
    
    print("\n=== Adansonia LLM 테스트 ===")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    
    while True:
        prompt = input("\n>>> ").strip()
        
        if prompt.lower() in ['quit', 'exit']:
            print("프로그램을 종료합니다.")
            break
        
        if not prompt:
            continue
        
        try:
            # Ollama처럼 직접 호출
            response = llm(prompt)
            print("\n" + response)
        except Exception as e:
            print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    main()