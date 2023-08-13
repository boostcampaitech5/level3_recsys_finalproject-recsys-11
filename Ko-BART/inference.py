import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration, BartConfig

config = BartConfig('/opt/ml/LLM/KoBART/KoBART-summarization/kobart_summary/config.json')

model = BartForConditionalGeneration.from_pretrained('/opt/ml/LLM/KoBART/KoBART-summarization/kobart_summary_07_22')
# model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')


tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')

text = """
1삼성전자가 1분기 반도체 부문에서만 4조6000억원 가까운 적자를 기록했다. 
삼성전자가 반도체 '보릿고개'를 넘고 있는 반면 삼성전자의 파운드리 라이벌인 대만 TSMC는 약 10조원의 영업이익을 거두며 뚜렷한 대조를 보였다.
28일 업계에 따르면 삼성전자가 올해 1분기 6400억원의 영업이익을 기록했다고 27일 공시했다.
지난해 같은 기간보다 95.5%나 떨어진 건데, 삼성전자의 분기 영업이익이 1조원 아래로 내려간 건 지난 2009년 1분기 이후 14년 만에 처음이다. 매출은 지난해 같은 기간보다 18.1% 감소한 63조7454억원으로 집계됐다. 
실적 악화의 주요 원인은 글로벌 경기 침체로 인한 반도체 수요 둔화에 따른 출하 부진과 가격 하락이다. 
삼성전자는 앞서 이런 상황을 고려해 25년 만에 반도체 감산을 선언한 바 있다.
김재준 삼성전자 메모리사업부 부사장은 27일 1분기 실적 발표 후 컨퍼런스콜에서 "고객 수요변동에 대응 가능한 물량을 충분히 확보했다고 판단했기에 생산량 하향 조정을 결정하게 됐다"며 "이번 생산 조정은 충분한 물량을 보유한 레거시(구형) 제품 중심으로 이뤄지고 있으며 선단제품 생산은 조정 없이 유지해 나갈 예정"이라고 말했다. 
반도체 사업을 담당하는 DS 부문에서 올해 1분기 4조5800억원의 적자를 냈다. 삼성전자가 반도체 부문에서 분기 적자를 기록한 것도 글로벌 금융위기를 겪은 2009년 1분기 이후 14년 만이다. 
반도체 부문 매출은 13조7300억원으로 나타났는데, 지난해 같은 기간보다 매출이 절반 수준으로 줄었다. 특히 낸드플래시와 디(D)램 등 주력 제품인 메모리반도체 가격이 급락한 게 실적 부진과 곧바로 연결됐다. 
스마트폰을 담당하는 MX부문은 매출 31조8200억원, 영업이익 3조9400억원을 기록해 반도체 부문 적자를 만회한 것으로 조사됐다. 
삼성전자의 올해 1분기 시설 투자액은 지난해 같은 기간 대비 36% 늘어난 10조7000억원으로, 어려운 환경에서도 미래 대비를 위한 투자는 크게 늘렸다. 
올해 1분기 반도체 수출은 1년 전보다 40%나 줄었다. AI 산업이 성장하면서 하반기부턴 수요가 살아날 거란 긍정적인 시각도 있는 반면, 세계적 경기 침체로 올해 안에 회복기로 접어들긴 어렵다는 의견도 나온다. 
김영건 미래에셋증권 연구원은 삼성전자에 대해 "이미 공시된 바와 같이 가동률 조정중임을 언급했으나, 구체적 규모에 대한 언급은 자제했다. 삼성전자의 높은 메모리 점유율과 이에 따른 규제 리스크 경계에서 비롯된 것으로 판단한다"고 말했다. 
이어 "메모리 가동률 조정이 본격화됨에 따라 재고 정상레벨에 도달한 일부 고객들의 수요를 자극할 수 있을 것으로 전망되며, 하반기 수요 회복과 맞물려 가격 인상 촉진이 예상된다"고 밝혔다. 
출처 : 이코리아(https://www.ekoreanews.co.kr)
"""

text = text.replace('\n', ' ')

raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=1024,  eos_token_id=1, length_penalty = 2.0)

print(tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True))

'1일 0 9시까지 최소 20만3220명이 코로나19에 신규 확진되어 역대 최다 기록을 갈아치웠다.' 
