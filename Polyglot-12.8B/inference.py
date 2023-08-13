import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from utils.prompter import Prompter

# MODEL = "./lora-alpaca"

peft_model_id = "./Fine_Tuning_Final_07_19"
config = PeftConfig.from_pretrained(peft_model_id)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map='auto')
model = PeftModel.from_pretrained(model, peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model.eval()

print(model)
    
def gen(x):
    
    with torch.autocast("cuda"):
        inputs = tokenizer(
            f"### 명령어: \n글을 요약해라\n\n### 원본: \n{x}\n\n### 요약:\n",
            return_tensors='pt',
            return_token_type_ids=False,
        )

        inputs = inputs.to('cuda')  # 입력 데이터를 CUDA로 이동
    
        print(len(inputs['input_ids'][0]))

        gened = model.generate(
            **inputs,
            max_new_tokens = 2048,
            early_stopping=True,
            top_k = 50,
            top_p = 0.90,
            temperature = 0.5,
            do_sample=True,
            length_penalty = 2.0,
            num_return_sequences = 1,
            eos_token_id=model.config.eos_token_id,
        )
    
    print(tokenizer.decode(gened[0]))

    
text = '''
앤드류 응은 스탠퍼드대 컴퓨터과학과 교수로 알려져 있으며, 데이터 사이언스와 인공지능 분야에서 세계적으로 인정받고 있는 학자이다. 그는 20일 서울대학교에서 열린 기자회견에서 서울대 데이터사이언스 대학원의 '초거대 AI 모델 및 플랫폼 최적화센터' 개소식에 참석하여 강연을 진행했으며, 21일에는 KBS 별관에서 열린 AI 석학 대담회에 참석하여 인공지능의 다목적 기술로서의 중요성과 AI 산업의 동향에 대해 공유하였다. 또한 B금융그룹에서도 AI에 관한 특별 강연회를 개최하여 앤드류 응 박사를 초청하였다.
'''

text = text.replace("\n", "")

if len(text) > 4000 :
    text = text[:4000]
    
gen(text)