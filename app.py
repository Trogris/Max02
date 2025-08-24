import os
import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from loaders import (
    carrega_site,
    carrega_youtube,
    carrega_pdf,
    carrega_csv,
    carrega_txt,
)

# ---------------- Configurações ----------------
TIPOS_ARQUIVOS_VALIDOS = ["Site", "Youtube", "Pdf", "Csv", "Txt"]

CONFIG_MODELOS = {
    "Groq": {
        "modelos": ["llama-3.1-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
        "chat": ChatGroq,  # requer groq_api_key
        "secrets_key": "GROQ_API_KEY",
    },
    "OpenAI": {
        "modelos": ["gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini"],
        "chat": ChatOpenAI,  # aceita api_key
        "secrets_key": "OPENAI_API_KEY",
    },
}

# Memória inicial
MEMORIA_INICIAL = ConversationBufferMemory()


# ---------------- Util: pega API key de forma segura ----------------
def get_api_key(provedor: str, typed_key: str) -> str:
    """
    Prioriza a chave digitada na sidebar.
    Se vazia, tenta st.secrets e depois variável de ambiente.
    """
    prov_info = CONFIG_MODELOS[provedor]
    secret_name = prov_info.get("secrets_key") or ""
    key = (typed_key or "").strip()
    if not key and secret_name:
        key = (st.secrets.get(secret_name) if hasattr(st, "secrets") else None) or os.getenv(secret_name) or ""
        key = key.strip()
    return key


def get_provider_kwargs(provedor: str, api_key: str) -> dict:
    """
    Mapeia o nome do parâmetro de credencial correto por provedor.
    Groq => groq_api_key
    OpenAI => api_key
    """
    if provedor == "Groq":
        return {"groq_api_key": api_key}
    elif provedor == "OpenAI":
        return {"api_key": api_key}
    return {"api_key": api_key}


# ---------------- Funções de carga ----------------
def carrega_arquivos(tipo_arquivo, arquivo):
    if not arquivo:
        st.warning("Nenhum conteúdo informado.")
        return None

    # Site
    if tipo_arquivo == "Site":
        if not (isinstance(arquivo, str) and arquivo.strip()):
            st.error("Informe uma URL válida para Site.")
            return None
        return carrega_site(arquivo.strip())

    # Youtube
    if tipo_arquivo == "Youtube":
        if not (isinstance(arquivo, str) and arquivo.strip()):
            st.error("Informe uma URL válida para Youtube.")
            return None
        try:
            return carrega_youtube(arquivo.strip())
        except Exception as e:
            st.error(
                f"❌ Erro ao carregar YouTube: {e}\n\n"
                "Dicas:\n"
                "- Use link de **vídeo** (ex.: https://www.youtube.com/watch?v=XXXXXXXXXXX ou https://youtu.be/XXXXXXXXXXX).\n"
                "- Evite playlist/canal (&list=...).\n"
                "- Shorts/Live funcionam se o link contiver o ID (11 caracteres).\n"
                "- Alguns vídeos não possuem transcrição (nem automática) ou têm transcrições desativadas."
            )
            return None

    # PDF / CSV / TXT (uploads)
    if tipo_arquivo in ["Pdf", "Csv", "Txt"]:
        try:
            data = arquivo.getbuffer()  # Streamlit UploadedFile
        except Exception:
            data = arquivo.read()

        sufixos = {"Pdf": ".pdf", "Csv": ".csv", "Txt": ".txt"}
        sufixo = sufixos[tipo_arquivo]

        with tempfile.NamedTemporaryFile(suffix=sufixo, delete=False) as temp:
            temp.write(bytes(data))
            caminho_temp = temp.name

        if tipo_arquivo == "Pdf":
            return carrega_pdf(caminho_temp)
        if tipo_arquivo == "Csv":
            return carrega_csv(caminho_temp)
        if tipo_arquivo == "Txt":
            return carrega_txt(caminho_temp)

    st.error(f"Tipo de arquivo não suportado: {tipo_arquivo}")
    return None


# ---------------- Monta cadeia LLM ----------------
def carrega_modelo(provedor, modelo, api_key_digitada, tipo_arquivo, arquivo):
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    if documento is None:
        return

    # Garante uso apenas do texto do Document
    doc_text = getattr(documento, "page_content", str(documento))

    api_key = get_api_key(provedor, api_key_digitada)
    if not api_key:
        st.error(
            f"Informe a chave do provedor {provedor} na barra lateral "
            f"ou configure {CONFIG_MODELOS[provedor]['secrets_key']} em Secrets/variável de ambiente."
        )
        st.stop()

    system_message = f"""Você é um assistente amigável chamado Max.
Você possui acesso às seguintes informações vindas de um documento {tipo_arquivo}:

####
{doc_text}
####

Utilize as informações fornecidas para basear as suas respostas.

Sempre que houver $ na sua saída, substitua por S.

Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue"
sugira ao usuário carregar novamente o Max!
"""

    template = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )

    # Instancia o chat com as credenciais corretas
    chat_cls = CONFIG_MODELOS[provedor]["chat"]
    kwargs = get_provider_kwargs(provedor, api_key)
    chat = chat_cls(model=modelo, **kwargs)

    # cria a cadeia e armazena
    st.session_state["chain"] = template | chat


# ---------------- Página de chat ----------------
def pagina_chat():
    st.header("🤖 Bem-vindo ao Max", divider=True)

    chain = st.session_state.get("chain")
    if chain is None:
        st.info("Inicialize o Max na barra lateral para começar.")
        return

    memoria = st.session_state.get("memoria", ConversationBufferMemory())

    # Re-renderiza o histórico
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    # Entrada do usuário
    input_usuario = st.chat_input("Pergunte ao Max...")
    if input_usuario:
        st.chat_message("human").markdown(input_usuario)

        # Stream da resposta
        resposta = st.chat_message("ai").write_stream(
            chain.stream({"input": input_usuario, "chat_history": memoria.buffer_as_messages})
        )

        # Atualiza memória
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta or "")
        st.session_state["memoria"] = memoria


# ---------------- Sidebar ----------------
def sidebar():
    tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos"])
    with tabs[0]:
        tipo_arquivo = st.selectbox("Selecione o tipo de entrada", TIPOS_ARQUIVOS_VALIDOS)
        arquivo = None

        if tipo_arquivo == "Site":
            arquivo = st.text_input("Digite a URL do site")
        elif tipo_arquivo == "Youtube":
            arquivo = st.text_input("Digite a URL do vídeo")
        elif tipo_arquivo == "Pdf":
            arquivo = st.file_uploader("Faça o upload do PDF", type=["pdf"])
        elif tipo_arquivo == "Csv":
            arquivo = st.file_uploader("Faça o upload do CSV", type=["csv"])
        elif tipo_arquivo == "Txt":
            arquivo = st.file_uploader("Faça o upload do TXT", type=["txt"])

    with tabs[1]:
        provedor = st.selectbox("Selecione o provedor do modelo", list(CONFIG_MODELOS.keys()))
        modelo = st.selectbox("Selecione o modelo", CONFIG_MODELOS[provedor]["modelos"])

        placeholder = f"Adicione a API key do provedor {provedor}"
        default_key = st.session_state.get(f"api_key_{provedor}", "")
        api_key_typed = st.text_input(placeholder, value=default_key, type="password")
        st.session_state[f"api_key_{provedor}"] = api_key_typed

        # Dica visual de onde mais pode estar a chave
        secret_env = CONFIG_MODELOS[provedor]["secrets_key"]
        st.caption(f"Também aceito chave via st.secrets['{secret_env}'] ou variável de ambiente {secret_env}.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Inicializar Max", use_container_width=True):
            carrega_modelo(provedor, modelo, api_key_typed, tipo_arquivo, arquivo)
    with col2:
        if st.button("Apagar Histórico de Conversa", use_container_width=True):
            st.session_state["memoria"] = ConversationBufferMemory()
            st.success("Histórico apagado.")


# ---------------- Main ----------------
def main():
    with st.sidebar:
        sidebar()
    pagina_chat()


if __name__ == "__main__":
    main()

