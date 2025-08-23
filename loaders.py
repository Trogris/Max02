import os
from time import sleep
import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from fake_useragent import UserAgent


def carrega_site(url: str) -> str:
    """
    Tenta carregar o conteúdo de uma página com até 5 tentativas,
    alternando o USER_AGENT para reduzir bloqueios simples.
    """
    documento = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            docs = loader.load()
            documento = '\n\n'.join([doc.page_content for doc in docs])
            break
        except Exception:
            print(f'Erro ao carregar o site (tentativa {i+1})')
            sleep(3)

    if documento == '':
        st.error('Não foi possível carregar o site. Verifique a URL ou tente novamente.')
        st.stop()
    return documento


def carrega_youtube(video_url_or_id: str) -> str:
    """
    Aceita URL completa do YouTube ou ID do vídeo.
    """
    loader = YoutubeLoader(video_url_or_id, add_video_info=False, language=['pt'])
    docs = loader.load()
    return '\n\n'.join([doc.page_content for doc in docs])


def carrega_csv(caminho: str) -> str:
    loader = CSVLoader(caminho)
    docs = loader.load()
    return '\n\n'.join([doc.page_content for doc in docs])


def carrega_pdf(caminho: str) -> str:
    loader = PyPDFLoader(caminho)
    docs = loader.load()
    return '\n\n'.join([doc.page_content for doc in docs])


def carrega_txt(caminho: str) -> str:
    loader = TextLoader(caminho)
    docs = loader.load()
    return '\n\n'.join([doc.page_content for doc in docs])


