from summarizer import Summarizer
from transformers import *

import streamlit as st
import re
import gc

@st.cache(allow_output_mutation=True)
def load_model():
    # Load model, model config and tokenizer via Transformers
    custom_config = AutoConfig.from_pretrained('ahmedrachid/FinancialBERT')
    custom_config.output_hidden_states=True
    custom_tokenizer = AutoTokenizer.from_pretrained('ahmedrachid/FinancialBERT')
    custom_model = AutoModel.from_pretrained('ahmedrachid/FinancialBERT', config=custom_config)
    model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
    return model

st.header("ANTPHY Financial Text Summarizer")
input = st.text_area('Enter your text: ', 'Enter your text')
num_sent = st.number_input('Number of sentences', min_value=1, max_value=50, value=5, step=1)

if st.button('Summarize'):
    model = load_model()
    input = re.sub(r'\([^)]*\)', '', input)
    summary = model(input,num_sentences=num_sent)
    st.markdown(summary)
gc.collect()
