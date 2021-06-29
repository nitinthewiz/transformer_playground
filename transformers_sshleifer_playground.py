# from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
# # from transformers import AutoTokenizer, TFAutoModel
  
# # tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir="D:\\Users\\khann\\.cache\\huggingface\\transformers")

# print(tokenizer)

# # model = TFAutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
# model = TFAutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn", cache_dir="D:\\Users\\khann\\.cache\\huggingface\\transformers")
# # model = TFAutoModel.from_pretrained("facebook/bart-large-cnn")

# file = open("news_short.txt")

# txt = file.read()

# myTokens = tokenizer(txt, max_length=1024, truncation=True, return_tensors="tf")
# # myTokens = tokenizer(txt, max_length=1024, truncation=True)

# # print(myTokens)
# print("*"*80)
# for key, value in myTokens.items():
#     print("{}:\n\t{}".format(key, value))
# print("*"*80)
# print(myTokens['input_ids'].shape)

# outputs = model.generate(myTokens)
# # just for debugging
# print(outputs)
# print(tokenizer.decode(outputs[0]))

import json

# from transformers import pipeline, TFAutoModelForSeq2SeqLM
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
# summarizer = pipeline('summarization')
# summarizer = pipeline('summarization', model=model, tokenizer="facebook/bart-large-cnn", framework='tf')
summarizer = pipeline('summarization', model=model, tokenizer=tokenizer)
print("*"*80)
print(summarizer)
print("*"*80)

def summarize_sshleifer(data_file_name):
	with open('data.json') as data_file:
	    data = data_file.read()
	    news_list = json.loads(data)

	for news_item in news_list:
		print("*"*80)
		print(news_item["text_article"])
		print("*"*80)
		print("-"*80)
		long_summary = summarizer(news_item["text_article"], max_length=330, min_length=100, truncation=False)
		print(long_summary[0]['summary_text'])
		print("-"*80)
		short_summary = summarizer(news_item["text_article"], max_length=50, min_length=10, truncation=False)
		print(short_summary[0]['summary_text'])
		print("-"*80)

	return {'long_summary': long_summary[0]['summary_text'], 'short_summary': short_summary[0]['summary_text']}
