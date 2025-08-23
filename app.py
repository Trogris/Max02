import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from loaders import carrega_site, carrega_youtube, carrega_pdf, carrega_csv, carrega_txt

# -----------------------------
# Configurações
# -----------------------------
TIPOS_ARQUIVOS_VALIDOS = ['Site', 'Youtube', 'Pdf', 'Csv', 'Txt']

CONFIG_MODELOS = {
    'Groq': {
        'modelos': ['llama-3.1-70b-versatile', 'gemma2-9b-it', 'mixtral-8x7b-32768'],
        'chat': ChatGroq
    },
    'OpenAI': {
        'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini'],
        'chat': ChatOpenAI
    }
}

# Memória de conversa
MEMORIA = ConversationBufferMemory()


# -----------------------------
# Funções utilitárias
# -----------------------------
def carrega_arquivos(tipo_arquivo: str, arquivo):
    """
    Recebe o tipo de entrada e o valor do widget correspondente.
    Retorna o conteúdo do documento como string.
    """
    if not arquivo:
        st.warning("Nenhum conteúdo informado. Selecione/insira um arquivo, site ou link do YouTube.")
        return None

    # Entrada como URL de site
    if tipo_arquivo == 'Site':
        if not isinstance(arquivo, str):
            st.error("Para 'Site', informe uma URL válida (texto).")
            return None
        return carrega_site(arquivo)

    # Entrada como URL do YouTube
    if tipo_arquivo == 'Youtube':
        if not isinstance(arquivo, str):
            st.error("Para 'Youtube', informe a URL do vídeo.")
            return None
        return carrega_youtube(arquivo)

    # Uploads: Pdf, Csv, Txt
    if tipo_arquivo in ['Pdf', 'Csv', 'Txt']:
        # st.file_uploader retorna UploadedFile com .getbuffer()
        # (compatível com Streamlit 1.38)
        try:
            data = arquivo.getbuffer()
        except Exception:
            # fallback raro
            if hasattr(arquivo, "read"):
                data = arquivo.read()
            else:
                st.error("Arquivo inválido para upload.")
                return None

        sufixos = {'Pdf': '.pdf', 'Csv': '.csv', 'Txt': '.txt'}
        sufixo = sufixos[tipo_arquivo]

        with tempfile.NamedTemporaryFile(suffix=sufixo, delete=False) as temp:
            temp.write(bytes(data))
            caminho_temp = temp.name

        if tipo_arquivo == 'Pdf':
            return carrega_pdf(caminho_temp)
        if tipo_arquivo == 'Csv':
            return carrega_csv(caminho_temp)
        if tipo_arquivo == 'Txt':
            return carrega_txt(caminho_temp)

    st.error(f"Tipo de arquivo não suportado: {tipo_arquivo}")
    return None


def carrega_modelo(provedor: str, modelo: str, api_key: str, tipo_arquivo: str, arquivo):
    """
    Monta o 'chain' com base no documento carregado e salva em session_state.
    """
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    if documento is None:
        return

    system_message = f"""Você é um assistente amigável chamado Max.
Você possui acesso às seguintes informações vindas de um documento {tipo_arquivo}:

####
{documento}
####

Utilize as informações fornecidas para basear as suas respostas.

Sempre que houver $ na sua saída, substitua por S.

Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue"
sugira ao usuário carregar novamente o Oráculo!
"""

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat_cls = CONFIG_MODELOS[provedor]['chat']
    chat = chat_cls(model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state['chain'] = chain
    # Mantém memória entre execuções
    if 'memoria' not in st.session_state:
        st.session_state['memoria'] = MEMORIA


def pagina_chat():
    st.header('🤖 Bem-vindo ao Max', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.info("Inicialize o Oráculo na barra lateral para começar.")
        return

    memoria: ConversationBufferMemory = st.session_state.get('memoria', MEMORIA)

    # Reexibir histórico
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)
        chat.markdown(mensagem.content)

    # Input do usuário + streaming da resposta
    input_usuario = st.chat_input('Fale com o Max...')
    if input_usuario:
        st.chat_message('human').markdown(input_usuario)

        chat_ai = st.chat_message('ai')
        resposta_stream = chain.stream({
            'input': input_usuario,
            'chat_history': memoria.buffer_as_messages
        })
        resposta_final = chat_ai.write_stream(resposta_stream)

        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta_final)
        st.session_state['memoria'] = memoria


def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])

    # ----------- Aba 1: Upload / Entradas -----------
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de entrada', TIPOS_ARQUIVOS_VALIDOS, index=0)
        arquivo = None

        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a URL do site')
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do vídeo do YouTube')
        elif tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo PDF', type=['pdf'])
        elif tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do arquivo CSV', type=['csv'])
        elif tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo TXT', type=['txt'])

    # ----------- Aba 2: Modelos -----------
    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor do modelo', list(CONFIG_MODELOS.keys()))
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a API key do provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}', '')
        )
        st.session_state[f'api_key_{provedor}'] = api_key

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Inicializar Oráculo', use_container_width=True):
            if not api_key:
                st.error("Informe sua API key para prosseguir.")
            else:
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    with col2:
        if st.button('Apagar Histórico de Conversa', use_container_width=True):
            st.session_state['memoria'] = MEMORIA
            st.success("Histórico apagado.")

    return  # nada explícito a retornar


def main():
    # Barra lateral primeiro (escolhas ficam salvas antes do chat renderizar)
    with st.sidebar:
        sidebar()

    # Área principal do chat
    pagina_chat()


if __name__ == '__main__':
    main()


