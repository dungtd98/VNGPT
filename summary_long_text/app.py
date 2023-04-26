from langchain.chat_models import ChatOpenAI
import openai
from langchain.schema import (
    HumanMessage, SystemMessage
)
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from lib_app.utils import *
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import redis
import asyncio
import threading
import json
SECRET_KEY = os.environ['SECRET_KEY']
REDIS_HOST = os.environ['REDIS_HOST']
REDIS_PORT = os.environ['REDIS_PORT']
redis_db = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0) 
def get_openai_resp(llm, prompt, index):
    """get openAi summary response and save data into redis"""
    resp = llm([HumanMessage(content=prompt)])
    redis_db.set(f'PART_{index}', resp.content)

def process_summarize_parallel(docs, init_prompt, llm):
    thread_list = []
    docs_redis_keys = []
    for index, doc in enumerate(docs):
        docs_redis_keys.append(f'PART_{index}')
        prompt=init_prompt+doc.page_content
        
        thread = threading.Thread(target=get_openai_resp, args=(llm, prompt, index))
        thread.start()
        thread_list.append(thread)
    return thread_list, docs_redis_keys
def return_full_summary(docs_redis_keys):
    full_summary = ''
    for key in docs_redis_keys:
        partial_text = redis_db.get(key).decode()
        full_summary+=' '+partial_text

    return full_summary
def summary_long_text(text, api_key, max_tokens, language_summary, prompts_summary, custom_prompts_summary, n=600):
    if language_summary == 'Tiếng Việt':
        language_summary = 'Vietnamese'
    else:
        language_summary = 'English'
    openai_key = encode("decode", api_key, SECRET_KEY)
    openai.api_key = openai_key
    chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, max_tokens=2048, temperature=0.7, openai_api_key=openai_key)
    
    words = text.split()
    num_words = len(words)
    if num_words > 600:
        text_spliter = CharacterTextSplitter(
            # separator = "\n",
            chunk_size = n,
            chunk_overlap  = 100,
            length_function = len,
        )
        text_chunks = text_spliter.split_text(text)
        docs = [Document(page_content=t) for t in text_chunks]
        total_parts = len(docs)*4
        max_words = int(round(max_tokens / total_parts))
        if prompts_summary == 'Tùy chỉnh prompt':
            prompt = f"Summarize text below into {language_summary} with maximum {max_words} words and {custom_prompts_summary}:"
        elif prompts_summary == 'Rút gọn văn bản bằng ngôn ngữ chọn và xuống gần với giá trị max_tokens':
            prompt = f"Summarize text below into {language_summary} with maximum {max_words} words:"
        else:
            prompt = f"Summarize text below into {language_summary} with maximum {max_words} words:"
        thread_list, docs_redis_keys = process_summarize_parallel(docs, prompt, chat)
        for thread in thread_list:
            thread.join()
        new_paragraph = return_full_summary(docs_redis_keys)  
        yield new_paragraph
    else:
        prompt = ""
        if prompts_summary == 'Tùy chỉnh prompt':
            prompt = f"Summarize text below into {language_summary} with maximum {max_tokens} words and {custom_prompts_summary}: {text}"
        elif prompts_summary == 'Rút gọn văn bản bằng ngôn ngữ chọn và xuống gần với giá trị max_tokens':
            prompt = f"Summarize text below into {language_summary} with maximum {max_tokens} words: {text}"
        else:
            prompt = f"Summarize text below into {language_summary} with maximum {max_tokens} words: {text}"
        resp = chat([HumanMessage(content=prompt)])
        new_paragraph = ''
        new_paragraph += ''.join(resp.content) + ' '
        yield new_paragraph