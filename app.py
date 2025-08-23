import tempfile
import os
from pathlib import Path
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import carrega_site, carrega_youtube, carrega_pdf, carrega_csv, carrega_txt

# ---------------- Configurações ----------------
TIPOS_ARQUIVOS_VALIDOS = ['Site', 'Youtube', 'Pdf', 'Csv', 'Txt', 'Pasta']

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

MEMORIA = ConversationBufferMemory()

# ---------------- Funções auxiliares ----------------
def carrega_arquivos(tipo_arquivo, arquivo):
    if not arquivo:
        st.warning("Nenhum conteúdo informado.")
        return None

    # Site
    if tipo_arquivo == 'Site':
        return carrega_site(arquivo)

    # Youtube
    if tipo_arquivo == 'Youtube':
        return carrega_youtube(arquivo)

    # PDF / CSV / TXT (uploads)
    if tipo_arquivo in ['Pdf', 'Csv', 'Txt']:
        try:
            data = arquivo.getbuffer()
        except Exception:
            data = arquivo.read()

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

    # Pasta local
    if tipo_arquivo == 'Pasta':
        base = Path(arquivo).expanduser().resolve()
        if not base.exists() or not base.is_dir():
            st.error(f"Pasta não encontrada: {base}")
            return None

        exts = st.session_state.get('pasta_exts', ['pdf', 'csv', 'txt', 'md'])
        recursivo = st.session_state.get('pasta_recursivo', True)
        pattern = "**/*" if recursivo else "*"

        arquivos = [p for p in base.glob(pattern)
                    if p.is_file() and p.suffix.lower().lstrip('.') in exts]

        if not arquivos:
            st.warning("Nenhum arquivo encontrado nessa pasta.")
            return None

        documentos = []
        for p in arquivos[:100]:  # limite de 100 arquivos
            try:
                suf = p.suffix.lower()
                if suf == ".pdf":
                    documentos.append(carrega_pdf(str(p)))
                elif suf == ".csv":
                    documentos.append(carrega_csv(str(p)))
                elif suf in (".txt", ".md"):
                    documentos.append(carrega_txt(str(p)))
            except Exception as e:
                st.warning(f"Falha ao ler {p.name}: {e}")

        return "\n\n".join(documentos)

    st.error(f"Tipo de arquivo não suportado: {tipo_arquivo}")
    return None


def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
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
    sugira ao usuário carregar novamente o Max!"""

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    st.session_state['chain'] = template | chat


def pagina_chat():
    st.header('🤖 Bem-vindo ao Max', divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.info("Inicialize o Max na barra lateral para começar.")
        return

    memoria = st.session_state.get('memoria', MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    input_usuario = st.chat_input('Pergunte ao Max...')
    if input_usuario:
        st.chat_message('human').markdown(input_usuario)
        resposta = st.chat_message('ai').write_stream(chain.stream({
            'input': input_usuario,
            'chat_history': memoria.buffer_as_messages
        }))
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria


def sidebar():
    tabs = st.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de entrada', TIPOS_ARQUIVOS_VALIDOS)
        arquivo = None

        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a URL do site')
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do vídeo')
        elif tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do PDF', type=['pdf'])
        elif tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do CSV', type=['csv'])
        elif tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do TXT', type=['txt'])
        elif tipo_arquivo == 'Pasta':
            arquivo = st.text_input('Digite o caminho da pasta (ex.: C:\\dados ou /home/ubuntu/dados)')
            recursivo = st.checkbox('Ler subpastas recursivamente', value=True)
            padrao = st.text_input('Extensões a considerar (separe por vírgula)', value='pdf,csv,txt,md')
            st.session_state['pasta_recursivo'] = recursivo
            st.session_state['pasta_exts'] = [e.strip().lower() for e in padrao.split(',') if e.strip()]

    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor do modelo', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f'Adicione a API key do provedor {provedor}',
            value=st.session_state.get(f'api_key_{provedor}', '')
        )
        st.session_state[f'api_key_{provedor}'] = api_key

    if st.button('Inicializar Max', use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state['memoria'] = MEMORIA


def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == '__main__':
    main()
