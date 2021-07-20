# from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
# # from transformers import AutoTokenizer, TFAutoModel
  
# # tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
# tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', cache_dir='D:\\Users\\khann\\.cache\\huggingface\\transformers')

# print(tokenizer)

# # model = TFAutoModelForSeq2SeqLM.from_pretrained('sshleifer/distilbart-cnn-12-6')
# model = TFAutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn', cache_dir='D:\\Users\\khann\\.cache\\huggingface\\transformers')
# # model = TFAutoModel.from_pretrained('facebook/bart-large-cnn')

# file = open('news_short.txt')

# txt = file.read()

# myTokens = tokenizer(txt, max_length=1024, truncation=True, return_tensors='tf')
# # myTokens = tokenizer(txt, max_length=1024, truncation=True)

# # print(myTokens)
# print('*'*80)
# for key, value in myTokens.items():
#     print('{}:\n\t{}'.format(key, value))
# print('*'*80)
# print(myTokens['input_ids'].shape)

# outputs = model.generate(myTokens)
# # just for debugging
# print(outputs)
# print(tokenizer.decode(outputs[0]))

import json

# from transformers import pipeline, TFAutoModelForSeq2SeqLM
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')

model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
# summarizer = pipeline('summarization')
# summarizer = pipeline('summarization', model=model, tokenizer='facebook/bart-large-cnn', framework='tf')

summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)
print('*'*80)
print(summarizer)
print('*'*80)


def summarize_fb(news_item):
	# print(news_item['text_article'])

	tokens = tokenizer.tokenize(news_item['text_article'])
	print(tokens)
	print(len(tokens))

	default_summary = summarizer(news_item['text_article'])
	long_summary = summarizer(news_item['text_article'], max_length=330, min_length=100, truncation=False)
	short_summary = summarizer(news_item['text_article'], max_length=50, min_length=10, truncation=False)
	return {'default_summary': default_summary[0]['summary_text'],
	        'long_summary': long_summary[0]['summary_text'],
	        'short_summary': short_summary[0]['summary_text']}
