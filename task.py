from transformers_facebook_playground import summarize_fb
from transformers_sshleifer_playground import summarize_sshleifer
from transformers_t5base_playground import summarize_t5base

data_file_name = 'data.json'

summary_fb = summarize_fb(data_file_name)
summary_sshleifer = summarize_sshleifer(data_file_name)
summary_t5base = summarize_t5base(data_file_name)


