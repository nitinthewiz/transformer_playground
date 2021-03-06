import json
from transformers_facebook_playground import summarize_fb
from transformers_sshleifer_playground import summarize_sshleifer
from transformers_t5base_playground import summarize_t5base

with open('data.json') as data_file:
    data = data_file.read()
    news_list = json.loads(data)

for news_item in news_list:
    summary_fb = summarize_fb(news_item)
    summary_sshleifer = summarize_sshleifer(news_item)
    summary_t5base = summarize_t5base(news_item)

    print('The news article is')
    print(news_item['text_article'])
    print('\n')
    print('-' * 80)
    print('\n')
    print('Summary from - facebook/bart-large-cnn')
    print('\n')
    print('Default Settings summary is - ')
    print(summary_fb['default_summary'])
    print('Long summary is - ')
    print(summary_fb['long_summary'])
    print('Short summary is - ')
    print(summary_fb['short_summary'])
    print('-' * 80)
    print('\n')
    print('Summary from - sshleifer/distilbart-cnn-12-6')
    print('\n')
    print('Default Settings summary is - ')
    print(summary_sshleifer['default_summary'])
    print('Long summary is - ')
    print(summary_sshleifer['long_summary'])
    print('Short summary is - ')
    print(summary_sshleifer['short_summary'])
    # print(summary_sshleifer)
    print('-' * 80)
    print('\n')
    print('Summary from - t5-base')
    print('\n')
    print('Default Settings summary is - ')
    print(summary_t5base['default_summary'])
    print('Long summary is - ')
    print(summary_t5base['long_summary'])
    print('Short summary is - ')
    print(summary_t5base['short_summary'])
    # print(summary_t5base)
    print('-' * 80)
    print('*' * 80)
    print('\n')
