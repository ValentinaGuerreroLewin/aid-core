import os
from typing import Optional, List, Literal

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# ============================================================
#   Cargar variables de entorno desde .env
# ============================================================
load_dotenv()

# ============================================================
#   CONFIGURACIÓN BASE AI.D CORE API
# ============================================================

app = FastAPI(
    title="AI.D Core API",
    version="1.0.0",
    description="Backend mínimo de AI.D para Ad.AI y apps externas.",
)

# CORS: para que luego puedas llamarlo desde otras apps/frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego lo puedes restringir a tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_LANG = os.getenv("AID_DEFAULT_LANG", "es")

# ============================================================
#   MODELOS DE PETICIÓN / RESPUESTA (ENDPOINT /chat)
# ============================================================

class ChatRequest(BaseModel):
    message: str
    mode: str = "adai"      # "adai" (asesor publicidad) o "external"
    language: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    mode: str
    used_language: str
    fallback: bool


# ============================================================
#   ENDPOINT DE SALUD / ESTADO
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "name": "AI.D Core API",
        "version": "1.0.0",
        "modes": ["adai", "external"],
        "default_lang": DEFAULT_LANG,
        "llm_configured": bool(OPENAI_API_KEY),
    }


# ============================================================
#   FUNCIÓN AUXILIAR: LLAMADA AL MODELO DE IA (para /chat)
# ============================================================

async def call_llm(prompt: str, language: str) -> Optional[str]:
    """
    Llama al modelo de OpenAI usando la API HTTP.
    Devuelve el texto generado, o None si algo falla (para usar fallback).
    """
    if not OPENAI_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4.1-mini",  # puedes cambiarlo luego
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres AI.D, la IA que se adapta al contexto donde se instala. "
                    "Responde de forma clara, estratégica y accionable."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        # Si falla la llamada, devolvemos None y el endpoint usará fallback
        return None


# ============================================================
#   ENDPOINT PRINCIPAL /chat  (para pruebas generales)
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    lang = request.language or DEFAULT_LANG or "es"

    # Prompt diferente según modo
    if request.mode == "adai":
        context = (
            "Eres AI.D como asesor de publicidad dentro de Ad.AI. "
            "Ayudas a PYMEs y emprendedores a mejorar sus campañas, "
            "definir presupuestos, audiencias, creatividades y embudos."
        )
    else:
        context = (
            "Eres AI.D en modo externo, asistente genérico instalable en cualquier app. "
            "Te adaptas al tema de la aplicación y das respuestas claras y accionables."
        )

    full_prompt = f"{context}\n\nIdioma de respuesta: {lang}\n\nMensaje del usuario:\n{request.message}"

    llm_reply = await call_llm(full_prompt, lang)

    # Si falla el modelo, usamos fallback básico
    if llm_reply is None:
        fallback_text = (
            "Hola, soy AI.D en modo básico (sin modelo configurado o con error de conexión). "
            "Cuéntame sobre tu contexto y lo que quieres lograr, y te ayudo a ordenar tus "
            "próximos 3 pasos con lo que sé hasta ahora."
        )
        return ChatResponse(
            reply=fallback_text,
            mode=request.mode,
            used_language=lang,
            fallback=True,
        )

    return ChatResponse(
        reply=llm_reply,
        mode=request.mode,
        used_language=lang,
        fallback=False,
    )


# ============================================================
#   WIDGET HTML INCRUSTADO (SERVIDO POR FASTAPI)  (/widget)
# ============================================================

WIDGET_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <title>AI.D Widget Demo</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7fb;
    }

    /* Botón flotante */
    .aid-launcher {
      position: fixed;
      bottom: 20px;
      right: 20px;
      width: 58px;
      height: 58px;
      background: #a8d0e6;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 8px 24px rgba(0,0,0,0.18);
      cursor: pointer;
      z-index: 9999;
      font-size: 24px;
    }

    .aid-launcher:hover {
      transform: translateY(-2px);
    }

    /* Ventana del chat */
    .aid-window {
      position: fixed;
      bottom: 90px;
      right: 20px;
      width: 360px;
      max-width: 95vw;
      height: 480px;
      max-height: 80vh;
      background: #ffffff;
      border-radius: 16px;
      box-shadow: 0 16px 40px rgba(0,0,0,0.18);
      display: none;
      flex-direction: column;
      overflow: hidden;
      z-index: 9998;
    }

    .aid-header {
      background: #a8d0e6;
      padding: 14px;
      font-weight: 700;
      color: #10212b;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .aid-header small {
      font-weight: 400;
      font-size: 11px;
      opacity: 0.8;
    }

    .aid-close {
      border: none;
      background: transparent;
      font-size: 18px;
      cursor: pointer;
      color: #10212b;
    }

    .aid-messages {
      flex: 1;
      padding: 12px;
      overflow-y: auto;
      background: #f7fbff;
      font-size: 13px;
    }

    .aid-msg {
      margin-bottom: 8px;
      line-height: 1.4;
    }

    .aid-msg-user {
      text-align: right;
    }

    .aid-msg-user span {
      display: inline-block;
      background: #10212b;
      color: #ffffff;
      padding: 8px 10px;
      border-radius: 16px 2px 16px 16px;
    }

    .aid-msg-bot span {
      display: inline-block;
      background: #ffffff;
      border-radius: 2px 16px 16px 16px;
      padding: 8px 10px;
      border: 1px solid rgba(0,0,0,0.08);
    }

    .aid-input {
      display: flex;
      padding: 10px;
      border-top: 1px solid rgba(0,0,0,0.08);
      background: #ffffff;
    }

    .aid-input input {
      flex: 1;
      padding: 7px 9px;
      border-radius: 10px;
      border: 1px solid rgba(0,0,0,0.18);
      font-size: 13px;
      outline: none;
    }

    .aid-input button {
      margin-left: 8px;
      background: #ffd700;
      color: #10212b;
      padding: 7px 14px;
      border-radius: 999px;
      border: none;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
    }

    .aid-input button:disabled {
      opacity: 0.6;
      cursor: default;
    }
  </style>
</head>
<body>

  <div style="padding:20px;">
    <h1>Demo local — AI.D Widget</h1>
    <p>Haz clic en el botón flotante para abrir el chat de AI.D.</p>
  </div>

  <!-- Botón flotante -->
  <div class="aid-launcher" onclick="toggleAid()">
    💬
  </div>

  <!-- Ventana del chat -->
  <div class="aid-window" id="aidWindow">
    <div class="aid-header">
      <div>
        AI.D · Asistente
        <br />
        <small>La IA que se adapta a tu app</small>
      </div>
      <button class="aid-close" onclick="toggleAid()">×</button>
    </div>

    <div class="aid-messages" id="aidMessages">
      <div class="aid-msg aid-msg-bot">
        <span>Hola, soy AI.D. Cuéntame brevemente qué quieres mejorar y en qué contexto usas este asistente.</span>
      </div>
    </div>

    <div class="aid-input">
      <input type="text" id="aidInput" placeholder="Escribe tu mensaje…" />
      <button id="aidSendBtn" onclick="sendAidMessage()">Enviar</button>
    </div>
  </div>

  <script>
    function toggleAid() {
      const box = document.getElementById("aidWindow");
      box.style.display = box.style.display === "flex" ? "none" : "flex";
    }

    async function sendAidMessage() {
      const input = document.getElementById("aidInput");
      const messages = document.getElementById("aidMessages");
      const btn = document.getElementById("aidSendBtn");

      const text = input.value.trim();
      if (!text) return;

      messages.innerHTML += '<div class="aid-msg aid-msg-user"><span>' + text + '</span></div>';
      input.value = "";
      messages.scrollTop = messages.scrollHeight;

      btn.disabled = true;
      btn.textContent = "Enviando…";

      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: text,
            mode: "adai",
            language: "es"
          })
        });

        const data = await res.json();
        const reply = data.reply || "No se recibió respuesta de AI.D.";

        messages.innerHTML += '<div class="aid-msg aid-msg-bot"><span>' + reply + '</span></div>';
        messages.scrollTop = messages.scrollHeight;
      } catch (e) {
        messages.innerHTML += '<div class="aid-msg aid-msg-bot"><span>Hubo un error al conectar con AI.D.</span></div>';
      } finally {
        btn.disabled = false;
        btn.textContent = "Enviar";
      }
    }
  </script>
</body>
</html>
"""

@app.get("/widget", response_class=HTMLResponse)
async def widget():
    return HTMLResponse(content=WIDGET_HTML)


# ============================================================
#   ENDPOINT PARA WIDGET / WEBSITE  (/api/aid-chat)
# ============================================================

class WidgetMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class WidgetChatRequest(BaseModel):
    system_prompt: str
    messages: List[WidgetMessage]


class WidgetChatResponse(BaseModel):
    reply: str


@app.post("/api/aid-chat", response_model=WidgetChatResponse)
async def aid_chat(request: WidgetChatRequest):
    """
    Endpoint pensado para el chatbot flotante del sitio o para Ad.AI.

    El front envía:
      - system_prompt: instrucciones del asistente (tono, rol, etc.)
      - messages: historial de conversación (user/assistant)

    Aquí construimos la request a OpenAI usando esos datos.
    """

    if not OPENAI_API_KEY:
        # Fallback rápido si no hay API key configurada
        return WidgetChatResponse(
            reply=(
                "Ahora mismo AI.D no tiene configurada la clave de OpenAI. "
                "Pídele al equipo que revise la variable OPENAI_API_KEY en el servidor."
            )
        )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Construimos la lista de mensajes para OpenAI
    openai_messages = [{"role": "system", "content": request.system_prompt}]
    for m in request.messages:
        openai_messages.append({"role": m.role, "content": m.content})

    payload = {
        "model": "gpt-4.1-mini",
        "messages": openai_messages,
    }

    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"].strip()
        return WidgetChatResponse(reply=reply)
    except Exception:
        return WidgetChatResponse(
            reply=(
                "No pude conectar con el modelo de IA en este momento. "
                "Si necesitas ayuda urgente, te recomiendo contactar directamente con MKT 360."
            )
        )