# loaders.py
import re
import urllib.parse
from typing import Optional

from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import _errors as yte


# =========================
# Util: extrai/valida ID
# =========================
def _extract_youtube_id(url_or_id: str) -> str:
    """
    Retorna o ID (11 chars) de:
      - watch?v=ID
      - youtu.be/ID
      - /shorts/ID
      - /live/ID
      - ou ID puro
    """
    s = (url_or_id or "").strip()

    # Caso já seja um ID puro
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s

    # Tentar parsear como URL
    try:
        u = urllib.parse.urlparse(s)
    except Exception:
        raise ValueError("URL do YouTube inválida.")

    host = (u.netloc or "").lower().replace("www.", "")
    path = u.path or ""
    qs = urllib.parse.parse_qs(u.query or "")

    # youtube.com/watch?v=ID
    if host.endswith("youtube.com"):
        if path == "/watch" and "v" in qs:
            vid = (qs["v"][0] or "").split("&")[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
                return vid
        # shorts/live: /shorts/ID ou /live/ID
        m = re.match(r"^/(shorts|live)/([A-Za-z0-9_-]{11})", path)
        if m:
            return m.group(2)

    # youtu.be/ID
    if host == "youtu.be":
        vid = path.lstrip("/").split("/")[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
            return vid

    raise ValueError("Não consegui extrair um ID de vídeo válido do link informado.")


# =========================
# Loader: YouTube transcript
# =========================
def carrega_youtube(url_ou_id: str) -> Document:
    """
    Carrega a transcrição do YouTube (se existir) e retorna um Document.
    Lança exceções com mensagens claras quando não for possível obter a transcrição.
    """
    video_id = _extract_youtube_id(url_ou_id)

    idiomas = ["pt-BR", "pt", "en"]
    transcript = None
    last_err: Optional[Exception] = None

    for lang in idiomas:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            break
        except Exception as e:
            last_err = e

    if transcript is None:
        if isinstance(last_err, yte.TranscriptsDisabled):
            raise RuntimeError("As transcrições estão desativadas para este vídeo.")
        if isinstance(last_err, (yte.NoTranscriptFound, yte.NoTranscriptAvailable)):
            raise RuntimeError("Este vídeo não possui transcrição disponível (nem automática).")
        if isinstance(last_err, yte.VideoUnavailable):
            raise RuntimeError("Vídeo indisponível (privado, removido ou restrito).")
        if isinstance(last_err, yte.InvalidVideoId):
            raise RuntimeError("ID de vídeo inválido. Verifique se o link aponta para um VÍDEO (não playlist/canal).")
        raise RuntimeError(f"Não foi possível obter a transcrição: {type(last_err).__name__}")

    text = "\n".join([item.get("text", "") for item in transcript]).strip()
    if not text:
        raise RuntimeError("A transcrição retornou vazia.")

    return Document(
        page_content=text,
        metadata={
            "source": f"https://www.youtube.com/watch?v={video_id}",
            "video_id": video_id,
            "loader": "youtube_transcript_api",
        },
    )


# =========================
# Loaders auxiliares (stubs)
# Mantém as assinaturas esperadas no seu app.
# Implemente conforme sua base atual se necessário.
# =========================
from langchain.schema import Document as _Doc

def carrega_site(url: str) -> _Doc:
    # Implemente com seu crawler atual
    return _Doc(page_content=f"[SITE] Conteúdo carregado de {url}")

def carrega_pdf(path: str) -> _Doc:
    # Implemente com seu parser de PDF atual
    return _Doc(page_content=f"[PDF] Conteúdo carregado de {path}")

def carrega_csv(path: str) -> _Doc:
    # Implemente com seu parser de CSV atual
    return _Doc(page_content=f"[CSV] Conteúdo carregado de {path}")

def carrega_txt(path: str) -> _Doc:
    # Implemente com seu parser de TXT atual
    return _Doc(page_content=f"[TXT] Conteúdo carregado de {path}")



