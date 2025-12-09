import os
import json
from typing import Optional, List, Literal, Tuple

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
    allow_origins=["*"],  # luego lo puedes restringir a tus domininios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_LANG = os.getenv("AID_DEFAULT_LANG", "es")

# ============================================================
#   CONFIGURACIÓN PROVEEDOR LLM (OPENAI / LOCAL)
# ============================================================

LLM_PROVIDER = os.getenv("AID_LLM_PROVIDER", "openai").lower()  # "openai" o "local"
LOCAL_LLM_URL = os.getenv("AID_LOCAL_LLM_URL", "http://localhost:11434/api/chat")
LOCAL_LLM_MODEL = os.getenv("AID_LOCAL_LLM_MODEL", "phi-2")

# ============================================================
#   LISTA DE PALABRAS FUERTES PARA STOP SCROLL (HEURÍSTICA)
# ============================================================

STRONG_WORDS = [
    "error",
    "errores",
    "secreto",
    "secretos",
    "sistema",
    "sistemas",
    "ventas",
    "venta",
    "crecer",
    "growth",
    "escala",
    "escala tu negocio",
    "automatizar",
    "automatización",
    "automatizacion",
    "funnel",
    "embudo",
    "plan",
    "plan claro",
    "resultado",
    "resultados",
    "clientes reales",
    "conversiones",
    "conversión",
    "conversión real",
    "diagnóstico",
    "diagnostico",
    "audit",
    "auditoría",
    "auditoria",
]


def _word_count(text: str) -> int:
    return len([w for w in text.strip().split() if w])


def _count_strong_words(text: str) -> int:
    lower = text.lower()
    count = 0
    for w in STRONG_WORDS:
        if w in lower:
            count += 1
    return count


def _evaluate_scroll_stop(topic: str, platform: str) -> Tuple[int, List[str]]:
    """
    Heurística simple para estimar qué tanto 'stop scroll' tiene un hook/tema
    según longitud, presencia de números, preguntas, palabras fuertes, etc.
    """
    details: List[str] = []
    text = (topic or "").strip()
    words = _word_count(text)

    if words == 0:
        details.append("No hay texto para evaluar. Escribe al menos una frase clara.")
        return 0, details

    has_number = any(ch.isdigit() for ch in text)
    has_question = "?" in text or "¿" in text
    strong_count = _count_strong_words(text)
    lower = text.lower()

    score = 0

    # 1) Longitud del hook
    if 4 <= words <= 10:
        score += 25
        details.append("✅ Longitud clara (4–10 palabras). Buen ritmo para detener scroll.")
    elif 2 <= words <= 14:
        score += 15
        details.append("🟡 Longitud aceptable, pero podría ser más directa.")
    else:
        score += 5
        details.append("⚠️ Hook muy largo o muy corto. Resta fuerza al stop scroll.")

    # 2) Números o datos concretos
    if has_number:
        score += 10
        details.append("✅ Incluye números/datos (ej. '3 pasos', '7 días'), esto suele captar atención.")
    else:
        details.append("🟡 No hay números ni datos concretos. Podrías sumar 'X pasos', 'Y días', etc.")

    # 3) Forma de pregunta
    if has_question:
        score += 10
        details.append("✅ Está formulado como pregunta. Invita a frenar el scroll para responder mentalmente.")
    else:
        details.append("🟡 No es una pregunta. A veces una pregunta directa aumenta el stop scroll.")

    # 4) Palabras fuertes
    if strong_count >= 2:
        score += 18
        details.append("✅ Usa varias palabras fuertes (errores, sistema, resultados, etc.).")
    elif strong_count == 1:
        score += 10
        details.append("🟡 Usa al menos una palabra fuerte. Podrías reforzar con otra más.")
    else:
        score += 4
        details.append("⚠️ Falta vocabulario que active decisión (errores, sistema, resultados...).")

    # 5) Claridad global (tamaño total)
    if words <= 18:
        score += 12
        details.append("✅ Frase relativamente corta. Se lee en un vistazo en el feed.")
    elif words <= 25:
        score += 6
        details.append("🟡 Un poco larga. Podría perder fuerza en pantallas móviles.")
    else:
        score += 2
        details.append("⚠️ Demasiado texto para un primer impacto. Conviene recortar.")

    # 6) CTA suave dentro del texto
    soft_cta = any(
        kw in lower
        for kw in [
            "comenta",
            "escribe",
            "pídeme",
            "pideme",
            "quieres",
            "descubre",
            "aprende",
            "guarda",
            "save",
        ]
    )
    if soft_cta:
        score += 8
        details.append("✅ Hay un llamado suave a la acción (CTA) dentro del hook.")
    else:
        details.append("🟡 No se percibe un CTA claro. Recuerda la mecánica de palabra clave.")

    # 7) Ajuste por plataforma
    if platform in ["instagram", "tiktok", "threads", "youtube"]:
        if words <= 14:
            score += 8
            details.append("✅ Buen tamaño para feeds rápidos (Instagram, TikTok, Threads, Shorts).")
        else:
            score -= 5
            details.append("⚠️ Para Instagram/TikTok/Threads conviene algo más corto y directo.")
    elif platform == "linkedin":
        if 6 <= words <= 18:
            score += 6
            details.append("✅ Para LinkedIn, buen equilibrio entre claridad y contexto.")
        elif words < 6:
            details.append("🟡 En LinkedIn puedes sumar un poco más de contexto estratégico.")

    # Normalizar 0–100
    if score < 0:
        score = 0
    if score > 100:
        score = 100

    return score, details


def _scroll_level(score: int) -> str:
    if score >= 80:
        return "ALTO"
    if score >= 55:
        return "MEDIO"
    return "BAJO"


def _scroll_advice(score: int, platform: str) -> str:
    if platform == "linkedin":
        base = "LinkedIn premia claridad, datos y contexto de negocio. "
    else:
        base = "Esta red premia hooks claros, emocionales y muy directos. "

    if score >= 80:
        return (
            base
            + "Este tema tiene muy buen potencial de detener el scroll. Puedes pasar a estructurar el resto del post "
              "(caption, slides o guion) y definir tu palabra clave del funnel."
        )
    if score >= 55:
        return (
            base
            + "El stop scroll es aceptable, pero con 1–2 ajustes (hacerlo más corto, sumar un número o una promesa "
              "concreta) podrías llevarlo a un nivel alto."
        )
    return (
        base
        + "El hook es débil para frenar el scroll. Conviene reformularlo: hazlo más corto, convierte la idea en "
          "pregunta o agrega una promesa de resultado clara."
    )

# ============================================================
#   CAPA GENÉRICA DE LLM (OPENAI / LOCAL)
# ============================================================


async def _call_openai_chat(messages: List[dict], model: str = "gpt-4.1-mini") -> Optional[str]:
    """
    Llamada genérica a OpenAI Chat Completions.
    Devuelve el 'content' del primer mensaje o None si falla.
    """
    if not OPENAI_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=80.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None


async def _call_local_chat(messages: List[dict], model: Optional[str] = None) -> Optional[str]:
    """
    Llamada genérica a un modelo local (ej. Ollama /phi-2).
    Ajusta el body si tu servidor usa un formato distinto.
    """
    try:
        async with httpx.AsyncClient(timeout=80.0) as client:
            r = await client.post(
                LOCAL_LLM_URL,
                json={
                    "model": model or LOCAL_LLM_MODEL,
                    "messages": messages,
                },
            )
        r.raise_for_status()
        data = r.json()

        # Intentamos distintas estructuras comunes
        if isinstance(data, dict):
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                return data["message"]["content"]
            if "content" in data:
                return data["content"]

        return None
    except Exception:
        return None


async def call_llm_messages(messages: List[dict], model: str = "gpt-4.1-mini") -> Optional[str]:
    """
    Capa unificada: decide según LLM_PROVIDER si usa OpenAI o local.
    """
    if LLM_PROVIDER == "local":
        return await _call_local_chat(messages, model=model)
    # por defecto: openai
    return await _call_openai_chat(messages, model=model)


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
    if LLM_PROVIDER == "openai":
        llm_configured = bool(OPENAI_API_KEY)
    else:
        # Asumimos que el modelo local está corriendo si el proveedor es "local"
        llm_configured = True

    return {
        "status": "ok",
        "name": "AI.D Core API",
        "version": "1.0.0",
        "modes": ["adai", "external"],
        "default_lang": DEFAULT_LANG,
        "llm_configured": llm_configured,
        "llm_provider": LLM_PROVIDER,
        "local_llm_url": LOCAL_LLM_URL if LLM_PROVIDER == "local" else None,
    }


# ============================================================
#   FUNCIÓN AUXILIAR: LLAMADA AL MODELO DE IA (para /chat)
# ============================================================

async def call_llm(prompt: str, language: str) -> Optional[str]:
    """
    Llama al modelo configurado (OpenAI o local) a través de la capa genérica.
    Devuelve el texto generado, o None si algo falla (para usar fallback).
    """
    system_msg = {
        "role": "system",
        "content": (
            "Eres AI.D, la IA que se adapta al contexto donde se instala. "
            "Responde de forma clara, estratégica y accionable."
        ),
    }
    user_msg = {"role": "user", "content": prompt}

    return await call_llm_messages([system_msg, user_msg], model="gpt-4.1-mini")


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
    """

    # Si el proveedor es OpenAI y no hay clave, fallback
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        return WidgetChatResponse(
            reply=(
                "Ahora mismo AI.D no tiene configurada la clave de OpenAI. "
                "Pídele al equipo que revise la variable OPENAI_API_KEY en el servidor."
            )
        )

    # Construimos la lista de mensajes para la capa genérica
    llm_messages = [{"role": "system", "content": request.system_prompt}]
    for m in request.messages:
        llm_messages.append({"role": m.role, "content": m.content})

    reply = await call_llm_messages(llm_messages, model="gpt-4.1-mini")

    if reply is None:
        return WidgetChatResponse(
            reply=(
                "No pude conectar con el modelo de IA en este momento. "
                "Si necesitas ayuda urgente, te recomiendo contactar directamente con MKT 360."
            )
        )

    return WidgetChatResponse(reply=reply.strip())


# ============================================================
#   ENDPOINT ESPECIALIZADO: OPTIMIZADOR DE CAMPAÑAS (/api/ads/optimizer)
# ============================================================

class BudgetSplit(BaseModel):
    platform: str
    percentage: float


class AdsOptimizerRequest(BaseModel):
    business_name: Optional[str] = None
    business_type: str          # ej: "clases de boxeo", "ecommerce de ropa"
    country: str                # ej: "Chile", "Estados Unidos"
    objective: Literal["awareness", "leads", "sales", "traffic", "mixed"]
    monthly_budget: float
    currency: str = "USD"
    platforms: List[str]        # ej: ["Instagram Ads", "Facebook Ads", "Google Ads"]
    notes: Optional[str] = None
    language: Optional[str] = None


class AdsOptimizerResponse(BaseModel):
    summary: str
    recommended_strategy: str
    budget_distribution: List[BudgetSplit]
    audiences: List[str]
    creatives: List[str]
    ctas: List[str]


@app.post("/api/ads/optimizer", response_model=AdsOptimizerResponse)
async def ads_optimizer(request: AdsOptimizerRequest):
    """
    Endpoint especializado para que AI.D actúe como planner de campañas.

    Devuelve:
      - resumen del caso
      - estrategia recomendada
      - distribución de presupuesto por plataforma
      - ideas de audiencias
      - ideas de creatividades
      - CTAs sugeridos
    """

    lang = request.language or DEFAULT_LANG or "es"

    # Fallback cuando proveedor es OpenAI y no hay clave
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        return AdsOptimizerResponse(
            summary=(
                "AI.D no tiene configurada la clave de OpenAI en el servidor. "
                "Configura OPENAI_API_KEY para activar el optimizador de campañas."
            ),
            recommended_strategy="",
            budget_distribution=[],
            audiences=[],
            creatives=[],
            ctas=[],
        )

    business_name = request.business_name or "Sin nombre específico"
    notes = request.notes or "Sin notas adicionales."
    platforms_str = ", ".join(request.platforms)

    system_prompt = (
        "Eres AI.D, planner senior de medios digitales con más de 15 años de experiencia "
        "en campañas para PYMEs y emprendedores. "
        "Tu trabajo es proponer una estrategia clara, realista y accionable."
    )

    user_prompt = f"""
Idioma de respuesta: {lang}.

Datos del negocio:
- Nombre: {business_name}
- Tipo de negocio: {request.business_type}
- País: {request.country}
- Objetivo principal: {request.objective}
- Presupuesto mensual: {request.monthly_budget} {request.currency}
- Plataformas disponibles: {platforms_str}
- Notas extra: {notes}

Devuelve SIEMPRE un JSON válido (sin texto antes ni después) con esta estructura:

{{
  "summary": "Resumen breve del caso y del objetivo en máximo 4 líneas.",
  "recommended_strategy": "Estrategia explicada en 6–10 líneas, en lenguaje simple, realista y accionable.",
  "budget_distribution": [
    {{ "platform": "Instagram Ads", "percentage": 50 }},
    {{ "platform": "Facebook Ads", "percentage": 30 }},
    {{ "platform": "Google Ads", "percentage": 20 }}
  ],
  "audiences": [
    "Descripción de audiencia 1",
    "Descripción de audiencia 2"
  ],
  "creatives": [
    "Idea de creatividad 1",
    "Idea de creatividad 2"
  ],
  "ctas": [
    "Llamado a la acción 1",
    "Llamado a la acción 2"
  ]
}}

Asegúrate de que la suma de los porcentajes de budget_distribution sea aproximadamente 100.
No incluyas comentarios ni explicación fuera del JSON.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw_content = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw_content is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw_content)
        except Exception:
            # Si no se puede parsear el JSON, devolvemos algo básico
            return AdsOptimizerResponse(
                summary="Propuesta de estrategia (texto libre):",
                recommended_strategy=raw_content,
                budget_distribution=[],
                audiences=[],
                creatives=[],
                ctas=[],
            )

        budget_list: List[BudgetSplit] = []
        for item in parsed.get("budget_distribution", []):
            try:
                budget_list.append(
                    BudgetSplit(
                        platform=str(item.get("platform", "Sin plataforma")),
                        percentage=float(item.get("percentage", 0)),
                    )
                )
            except Exception:
                continue

        return AdsOptimizerResponse(
            summary=str(parsed.get("summary", "")),
            recommended_strategy=str(parsed.get("recommended_strategy", "")),
            budget_distribution=budget_list,
            audiences=[str(a) for a in parsed.get("audiences", [])],
            creatives=[str(c) for c in parsed.get("creatives", [])],
            ctas=[str(c) for c in parsed.get("ctas", [])],
        )

    except Exception:
        # Fallback si falla toda la llamada
        return AdsOptimizerResponse(
            summary=(
                "No se pudo conectar con el modelo de IA para generar la estrategia. "
                "Intenta de nuevo en unos minutos o revisa la configuración del modelo."
            ),
            recommended_strategy="",
            budget_distribution=[],
            audiences=[],
            creatives=[],
            ctas=[],
        )


# ============================================================
#   ENDPOINT ESPECIALIZADO: ANALIZADOR DE CONTENIDO (/api/content/analyzer)
# ============================================================

class ContentAnalyzerRequest(BaseModel):
    platform: Literal["instagram", "linkedin", "tiktok", "facebook", "youtube", "generic"]
    content_text: Optional[str] = None          # caption, texto del post, descripción del video
    url: Optional[str] = None                   # enlace al post (opcional)
    keywords: Optional[List[str]] = None        # palabras clave esperadas
    language: Optional[str] = None              # idioma deseado para la respuesta


class ContentAnalyzerResponse(BaseModel):
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    optimized_caption: str
    keyword_gaps: List[str]
    performance_score: int
    ad_recommendation: str


@app.post("/api/content/analyzer", response_model=ContentAnalyzerResponse)
async def content_analyzer(request: ContentAnalyzerRequest):
    """
    Analiza contenido de redes sociales y devuelve:
      - diagnóstico
      - fortalezas
      - debilidades
      - qué mejorar
      - caption optimizado
      - palabras clave faltantes
      - puntaje probable de rendimiento (1–100)
      - recomendación de si sirve para publicidad
    """

    lang = request.language or DEFAULT_LANG or "es"

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        return ContentAnalyzerResponse(
            summary="AI.D no tiene clave de OpenAI configurada.",
            strengths=[],
            weaknesses=[],
            recommendations=[],
            optimized_caption="",
            keyword_gaps=[],
            performance_score=0,
            ad_recommendation="No disponible."
        )

    keywords_str = ", ".join(request.keywords) if request.keywords else "Ninguna"

    user_prompt = f"""
Analiza el siguiente contenido para la plataforma: {request.platform}.
Idioma de respuesta: {lang}.

Contenido del usuario:
{request.content_text or "Sin texto entregado."}

URL asociada (si aplica): {request.url or "No proporcionada"}

Palabras clave esperadas: {keywords_str}

Devuelve SIEMPRE un JSON válido con esta estructura:

{{
  "summary": "Diagnóstico del contenido en máximo 3 líneas.",
  "strengths": ["fortaleza 1", "fortaleza 2"],
  "weaknesses": ["debilidad 1", "debilidad 2"],
  "recommendations": ["recomendación 1", "recomendación 2"],
  "optimized_caption": "Caption reescrito y optimizado para alcance y conversión.",
  "keyword_gaps": ["keyword faltante 1"],
  "performance_score": 0-100,
  "ad_recommendation": "Explica si este contenido sirve para publicidad y por qué."
}}

No agregues texto fuera del JSON.
"""

    messages = [
        {
            "role": "system",
            "content": "Eres AI.D, analista senior de contenido para marketing digital y publicidad."
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            # Si falla el parseo, devolvemos texto libre en campos básicos
            return ContentAnalyzerResponse(
                summary="Resultado sin formato JSON:",
                strengths=[],
                weaknesses=[],
                recommendations=[],
                optimized_caption=raw,
                keyword_gaps=[],
                performance_score=0,
                ad_recommendation="No se pudo evaluar."
            )

        return ContentAnalyzerResponse(
            summary=str(parsed.get("summary", "")),
            strengths=[str(s) for s in parsed.get("strengths", [])],
            weaknesses=[str(s) for s in parsed.get("weaknesses", [])],
            recommendations=[str(r) for r in parsed.get("recommendations", [])],
            optimized_caption=str(parsed.get("optimized_caption", "")),
            keyword_gaps=[str(k) for k in parsed.get("keyword_gaps", [])],
            performance_score=int(parsed.get("performance_score", 0)),
            ad_recommendation=str(parsed.get("ad_recommendation", "")),
        )

    except Exception:
        return ContentAnalyzerResponse(
            summary="Error al conectar con la IA.",
            strengths=[],
            weaknesses=[],
            recommendations=[],
            optimized_caption="",
            keyword_gaps=[],
            performance_score=0,
            ad_recommendation="No disponible."
        )


# ============================================================
#   NUEVO ENDPOINT: AUDITOR SEO 360° (/api/seo/audit)
# ============================================================

class VisibleSEO(BaseModel):
    title: Optional[str] = None
    h1: Optional[str] = None
    other_headings: List[str] = []
    content_focus: str = ""
    readability: str = ""
    keyword_usage: str = ""


class HiddenSEO(BaseModel):
    meta_title_ok: bool = False
    meta_description_ok: bool = False
    meta_robots: Optional[str] = None
    canonical_present: bool = False
    schema_org_present: bool = False
    open_graph_present: bool = False
    hreflang_present: bool = False
    notes: str = ""


class SeoAuditRequest(BaseModel):
    url: Optional[str] = None
    html: Optional[str] = None
    text: Optional[str] = None
    target_keywords: Optional[List[str]] = None
    language: Optional[str] = None


class SeoAuditResponse(BaseModel):
    url: Optional[str] = None
    summary: str
    visible_seo: VisibleSEO
    hidden_seo: HiddenSEO
    priority_fixes: List[str]
    quick_wins: List[str]
    recommended_keywords: List[str]
    checklist_90_days: List[str]


@app.post("/api/seo/audit", response_model=SeoAuditResponse)
async def seo_audit(request: SeoAuditRequest):
    """
    Auditoría SEO 360° para sitios web:
      - Analiza lo que ve el usuario (títulos, contenido, enfoque, legibilidad).
      - Analiza lo que ve el buscador (meta, canonical, OG, schema, etc.).
    Pensado para ser consumido por:
      - Ad.AI
      - Widgets / scripts instalables en cualquier sitio.
    """

    lang = request.language or DEFAULT_LANG or "es"

    # Necesitamos al menos algo de contenido (texto o HTML)
    if not (request.text or request.html):
        # Fallback rápido sin IA
        return SeoAuditResponse(
            url=request.url,
            summary=(
                "No se pudo realizar la auditoría SEO porque no se envió contenido "
                "('text' o 'html')."
            ),
            visible_seo=VisibleSEO(),
            hidden_seo=HiddenSEO(notes="Sin contenido para analizar."),
            priority_fixes=[
                "Vuelve a llamar el endpoint enviando al menos el HTML o texto plano de la página."
            ],
            quick_wins=[],
            recommended_keywords=[],
            checklist_90_days=[],
        )

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        # Fallback cuando no hay modelo avanzado
        return SeoAuditResponse(
            url=request.url,
            summary=(
                "AI.D está en modo básico (sin clave de OpenAI configurada). "
                "Se requiere OPENAI_API_KEY para activar la auditoría SEO avanzada."
            ),
            visible_seo=VisibleSEO(),
            hidden_seo=HiddenSEO(notes="Sin análisis técnico por falta de modelo."),
            priority_fixes=[
                "Configurar la variable de entorno OPENAI_API_KEY en el servidor.",
                "Reintentar la auditoría SEO una vez activado el modelo.",
            ],
            quick_wins=[],
            recommended_keywords=[],
            checklist_90_days=[],
        )

    base_content = request.text or request.html or ""
    base_content = base_content[:15000]  # recorte por seguridad

    target_keywords_str = (
        ", ".join(request.target_keywords) if request.target_keywords else "Ninguna keyword objetivo específica."
    )

    idioma = "español" if lang.startswith("es") else "inglés"

    user_prompt = f"""
Eres AI.D, una IA experta en SEO on-page y técnico.

Tu tarea es analizar una página web tanto en lo que el usuario ve (contenido, títulos, textos)
como en lo que el buscador ve (meta tags, schema, OG, robots, canonicals, etc.).

Responde SIEMPRE en {idioma} y SIEMPRE en formato JSON con esta estructura EXACTA:

{{
  "summary": "Resumen breve de 2–3 frases sobre el estado SEO de la página.",
  "visible_seo": {{
    "title": "Título principal que percibe el usuario (puede basarse en <title> o H1).",
    "h1": "Contenido del H1 principal si es identificable.",
    "other_headings": ["Lista de otros headings relevantes (H2, H3)."],
    "content_focus": "Evaluación del enfoque del contenido respecto a las keywords.",
    "readability": "Comentarios sobre claridad, extensión y estructura del contenido.",
    "keyword_usage": "Cómo se usan las palabras clave objetivo en el contenido."
  }},
  "hidden_seo": {{
    "meta_title_ok": true,
    "meta_description_ok": false,
    "meta_robots": "Valor de meta robots si se puede inferir, o 'desconocido'.",
    "canonical_present": true,
    "schema_org_present": false,
    "open_graph_present": true,
    "hreflang_present": false,
    "notes": "Notas resumidas sobre meta tags, OG, schema, indexabilidad, etc."
  }},
  "priority_fixes": [
    "Lista de 3–7 acciones SEO críticas y priorizadas (alto impacto)."
  ],
  "quick_wins": [
    "Lista de 3–7 mejoras rápidas que se puedan aplicar en 1–2 días."
  ],
  "recommended_keywords": [
    "Lista de 5–10 palabras clave sugeridas, relacionadas con el contenido y las keywords objetivo."
  ],
  "checklist_90_days": [
    "Lista de 5–10 acciones a completar en 90 días para mejorar SEO on-page y técnico."
  ]
}}

Ten en cuenta:
- Contenido proporcionado (HTML o texto plano recortado):
\"\"\"{base_content}\"\"\"

- Palabras clave objetivo del cliente:
{target_keywords_str}

Reglas:
- Si no puedes detectar algo con claridad (por ejemplo, schema.org o hreflang),
  responde de forma conservadora (por ejemplo, schema_org_present: false y anótalo en 'notes').
- En 'priority_fixes' y 'quick_wins', da acciones concretas, no genéricas.
- En 'recommended_keywords', mezcla variaciones de cola corta y cola larga.
- No agregues texto fuera del JSON. Solo responde el objeto JSON.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Eres AI.D, una IA experta en SEO on-page y técnico. "
                "Analizas tanto el contenido visible como las señales técnicas."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            return SeoAuditResponse(
                url=request.url,
                summary="No se pudo parsear la respuesta del modelo como JSON.",
                visible_seo=VisibleSEO(),
                hidden_seo=HiddenSEO(notes="Revisa la respuesta cruda del modelo y ajusta el prompt si es necesario."),
                priority_fixes=[],
                quick_wins=[],
                recommended_keywords=[],
                checklist_90_days=[],
            )

        visible_data = parsed.get("visible_seo", {}) or {}
        hidden_data = parsed.get("hidden_seo", {}) or {}

        visible = VisibleSEO(
            title=visible_data.get("title"),
            h1=visible_data.get("h1"),
            other_headings=visible_data.get("other_headings", []) or [],
            content_focus=visible_data.get("content_focus", "") or "",
            readability=visible_data.get("readability", "") or "",
            keyword_usage=visible_data.get("keyword_usage", "") or "",
        )

        hidden = HiddenSEO(
            meta_title_ok=bool(hidden_data.get("meta_title_ok", False)),
            meta_description_ok=bool(hidden_data.get("meta_description_ok", False)),
            meta_robots=hidden_data.get("meta_robots"),
            canonical_present=bool(hidden_data.get("canonical_present", False)),
            schema_org_present=bool(hidden_data.get("schema_org_present", False)),
            open_graph_present=bool(hidden_data.get("open_graph_present", False)),
            hreflang_present=bool(hidden_data.get("hreflang_present", False)),
            notes=hidden_data.get("notes", "") or "",
        )

        return SeoAuditResponse(
            url=request.url,
            summary=str(parsed.get("summary", "")),
            visible_seo=visible,
            hidden_seo=hidden,
            priority_fixes=[str(a) for a in parsed.get("priority_fixes", [])],
            quick_wins=[str(a) for a in parsed.get("quick_wins", [])],
            recommended_keywords=[str(k) for k in parsed.get("recommended_keywords", [])],
            checklist_90_days=[str(t) for t in parsed.get("checklist_90_days", [])],
        )

    except Exception as e:
        # Error al conectar con LLM
        return SeoAuditResponse(
            url=request.url,
            summary="Hubo un error al conectar con el modelo de IA para la auditoría SEO.",
            visible_seo=VisibleSEO(),
            hidden_seo=HiddenSEO(notes=f"Detalle técnico: {str(e)}"),
            priority_fixes=[
                "Reintentar la auditoría en unos minutos.",
                "Verificar la conectividad del servidor con el modelo configurado.",
            ],
            quick_wins=[],
            recommended_keywords=[],
            checklist_90_days=[],
        )


# ============================================================
#   ENDPOINT ESPECIALIZADO: GENERADOR DE CONTENIDO
#   (/api/content/generator)
# ============================================================

class ContentGeneratorRequest(BaseModel):
    platform: Literal["instagram", "linkedin", "tiktok", "facebook", "youtube", "generic"]
    topic: str                           # tema del contenido
    objective: Literal["awareness", "engagement", "leads", "sales", "authority", "mixed"]
    style: Literal["mkt360", "professional", "motivational", "direct", "friendly"] = "mkt360"
    keywords: Optional[List[str]] = None
    duration: Optional[str] = None       # ej: "10s", "30s", "3 slides", "1 minuto"
    language: Optional[str] = None       # si no viene → AI decide el idioma óptimo


class ContentGeneratorResponse(BaseModel):
    idea: str
    hooks: List[str]
    script: str
    variants: List[str]
    recommended_format: str
    difficulty: str
    ctas: List[str]
    scroll_stopper_score: int
    virality_probability: str
    differentiation_level: str


@app.post("/api/content/generator", response_model=ContentGeneratorResponse)
async def content_generator(request: ContentGeneratorRequest):
    """
    Genera contenido optimizado para redes sociales:
      - Hooks
      - Guion
      - Variantes A/B
      - CTA
      - Recomendación de formato
      - Métricas predictivas (scroll stopper, viralidad, diferenciación)
    """

    # Selección automática de idioma según plataforma
    if request.language:
        lang = request.language
    else:
        if request.platform in ["linkedin"]:
            lang = "en"
        elif request.platform in ["instagram", "facebook", "tiktok"]:
            lang = "es"
        else:
            lang = DEFAULT_LANG or "es"

    if request.style == "mkt360":
        tone = (
            "Usa tono MKT 360: inteligente, rápido, elegante, profesional, orientado al futuro, "
            "cero vendehumo, directo al punto, motivador y estratégico."
        )
    elif request.style == "professional":
        tone = "Tono profesional, claro, ejecutivo y directo."
    elif request.style == "motivational":
        tone = "Tono motivador, inspirador, enfocado en crecimiento y mentalidad."
    elif request.style == "direct":
        tone = "Tono directo, claro y sin rodeos."
    else:
        tone = "Tono cercano y amigable."

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        return ContentGeneratorResponse(
            idea="AI.D no tiene clave configurada.",
            hooks=[],
            script="",
            variants=[],
            recommended_format="",
            difficulty="",
            ctas=[],
            scroll_stopper_score=0,
            virality_probability="Desconocida",
            differentiation_level="Desconocido"
        )

    keywords_str = ", ".join(request.keywords) if request.keywords else "Ninguna"

    user_prompt = f"""
Genera contenido para la plataforma: {request.platform}
Objetivo del contenido: {request.objective}
Idioma: {lang}
Duración/estructura: {request.duration or "No especificada"}
Palabras clave obligatorias: {keywords_str}

Tema central: {request.topic}

Estilo requerido:
{tone}

Devuelve SIEMPRE un JSON válido con esta estructura exacta:

{{
  "idea": "Idea general del contenido.",
  "hooks": ["Hook 1", "Hook 2", "Hook 3"],
  "script": "Guion completo optimizado con palabras clave.",
  "variants": ["Variante A", "Variante B"],
  "recommended_format": "Reel 30s / Carrusel 3 slides / Post estático",
  "difficulty": "baja | media | alta",
  "ctas": ["CTA 1", "CTA 2"],
  "scroll_stopper_score": 0-100,
  "virality_probability": "baja | media | alta | muy alta",
  "differentiation_level": "bajo | medio | alto | único"
}}

No escribas nada fuera del JSON.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Eres AI.D, generador creativo avanzado para publicidad y contenido. "
                "Experto en comportamiento humano, scroll-stoppers, marketing digital, "
                "copywriting de alto rendimiento y visual strategy."
            )
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            return ContentGeneratorResponse(
                idea="No se pudo procesar el JSON.",
                hooks=[],
                script=raw,
                variants=[],
                recommended_format="",
                difficulty="",
                ctas=[],
                scroll_stopper_score=0,
                virality_probability="desconocida",
                differentiation_level="desconocido"
            )

        return ContentGeneratorResponse(
            idea=str(parsed.get("idea", "")),
            hooks=[str(h) for h in parsed.get("hooks", [])],
            script=str(parsed.get("script", "")),
            variants=[str(v) for v in parsed.get("variants", [])],
            recommended_format=str(parsed.get("recommended_format", "")),
            difficulty=str(parsed.get("difficulty", "")),
            ctas=[str(c) for c in parsed.get("ctas", [])],
            scroll_stopper_score=int(parsed.get("scroll_stopper_score", 0)),
            virality_probability=str(parsed.get("virality_probability", "")),
            differentiation_level=str(parsed.get("differentiation_level", "")),
        )

    except Exception as e:
        return ContentGeneratorResponse(
            idea="Error en servidor o conexión con el modelo.",
            hooks=[],
            script=str(e),
            variants=[],
            recommended_format="",
            difficulty="",
            ctas=[],
            scroll_stopper_score=0,
            virality_probability="Desconocida",
            differentiation_level="Desconocido"
        )


# ============================================================
#   NUEVO ENDPOINT: SCORE DE STOP SCROLL
#   (/api/content/scroll-stop)
# ============================================================

class ScrollStopRequest(BaseModel):
    platform: Literal["instagram", "linkedin", "tiktok", "threads", "facebook", "youtube", "generic"]
    topic: str
    language: Optional[str] = None  # por si más adelante quieres respuestas en inglés


class ScrollStopResponse(BaseModel):
    score: int
    level: str            # "ALTO", "MEDIO", "BAJO"
    details: List[str]    # explicación paso a paso
    advice: str           # recomendación estratégica


@app.post("/api/content/scroll-stop", response_model=ScrollStopResponse)
async def scroll_stop(request: ScrollStopRequest):
    """
    Calcula de forma rápida (sin modelo de IA) qué tanto 'stop scroll'
    tiene un tema/hook en función de:
      - longitud
      - números/datos
      - pregunta o no
      - palabras fuertes
      - CTA dentro del texto
      - plataforma elegida

    Sirve para LinkedIn, Instagram, Threads, TikTok, Facebook, YouTube, etc.
    """

    lang = request.language or DEFAULT_LANG or "es"
    platform = request.platform
    topic = request.topic or ""

    score, details = _evaluate_scroll_stop(topic, platform)
    level = _scroll_level(score)
    advice = _scroll_advice(score, platform)

    return ScrollStopResponse(
        score=int(score),
        level=level,
        details=details,
        advice=advice,
    )


# ============================================================
#   A) ENDPOINT: OPTIMIZADOR DE HOOKS
#   (/api/content/hook-optimizer)
# ============================================================

class HookOptimizerRequest(BaseModel):
    platform: Literal["instagram", "linkedin", "tiktok", "threads", "facebook", "youtube", "generic"]
    hook: str
    objective: Literal["awareness", "engagement", "leads", "sales", "authority", "mixed"] = "awareness"
    language: Optional[str] = None


class HookOptimizerResponse(BaseModel):
    original_hook: str
    original_score: int
    original_level: str
    improved_hook: str
    improved_score: int
    improved_level: str
    variants: List[str]
    explanation: str
    recommended_format: str
    suggested_keyword: str


@app.post("/api/content/hook-optimizer", response_model=HookOptimizerResponse)
async def hook_optimizer(request: HookOptimizerRequest):
    """
    Optimiza un hook/título corto para aumentar su capacidad de 'stop scroll'.
    Combina heurística propia + modelo de IA (cuando esté disponible).
    """

    lang = request.language or DEFAULT_LANG or "es"
    platform = request.platform
    original_hook = request.hook.strip()

    # Evaluamos hook original con la heurística
    orig_score, _ = _evaluate_scroll_stop(original_hook, platform)
    orig_level = _scroll_level(orig_score)

    # Si el proveedor es OpenAI y no hay API key, devolvemos versión básica sin IA
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        return HookOptimizerResponse(
            original_hook=original_hook,
            original_score=int(orig_score),
            original_level=orig_level,
            improved_hook=original_hook,
            improved_score=int(orig_score),
            improved_level=orig_level,
            variants=[],
            explanation="AI.D está en modo sin modelo avanzado. Usa la heurística básica de stop scroll.",
            recommended_format="Post estático",
            suggested_keyword="SYSTEM",
        )

    user_prompt = f"""
Plataforma: {platform}
Objetivo: {request.objective}
Idioma: {lang}

Hook original:
"{original_hook}"

Genera SIEMPRE un JSON válido con esta estructura:

{{
  "improved_hook": "Versión optimizada del hook, máximo 7 palabras, muy claro y potente.",
  "variants": [
    "Variante alternativa 1",
    "Variante alternativa 2",
    "Variante alternativa 3"
  ],
  "explanation": "Explica en 3–5 líneas qué se mejoró (longitud, fuerza, claridad, CTA, etc.).",
  "recommended_format": "Reel 30s / Carrusel 3 slides / Post estático",
  "suggested_keyword": "PALABRA_CLAVE_CORTA para el funnel (ej. SYSTEM, AUDIT, GROWTH)."
}}

No escribas nada fuera del JSON.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Eres AI.D, experto en hooks de alto rendimiento para redes sociales. "
                "Conoces a fondo el comportamiento de scroll en Instagram, LinkedIn, TikTok, Threads y YouTube. "
                "Tu misión es hacer los hooks más cortos, claros y potentes, alineados al funnel."
            )
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            return HookOptimizerResponse(
                original_hook=original_hook,
                original_score=int(orig_score),
                original_level=orig_level,
                improved_hook=original_hook,
                improved_score=int(orig_score),
                improved_level=orig_level,
                variants=[],
                explanation=f"No se pudo parsear el JSON. Respuesta cruda del modelo: {raw}",
                recommended_format="Post estático",
                suggested_keyword="SYSTEM",
            )

        improved_hook = str(parsed.get("improved_hook", original_hook)).strip()
        variants = [str(v) for v in parsed.get("variants", [])]

        # Evaluamos el hook mejorado con la misma heurística
        imp_score, _ = _evaluate_scroll_stop(improved_hook, platform)
        imp_level = _scroll_level(imp_score)

        return HookOptimizerResponse(
            original_hook=original_hook,
            original_score=int(orig_score),
            original_level=orig_level,
            improved_hook=improved_hook,
            improved_score=int(imp_score),
            improved_level=imp_level,
            variants=variants,
            explanation=str(parsed.get("explanation", "")),
            recommended_format=str(parsed.get("recommended_format", "")),
            suggested_keyword=str(parsed.get("suggested_keyword", "")),
        )

    except Exception as e:
        return HookOptimizerResponse(
            original_hook=original_hook,
            original_score=int(orig_score),
            original_level=orig_level,
            improved_hook=original_hook,
            improved_score=int(orig_score),
            improved_level=orig_level,
            variants=[],
            explanation=f"Error al conectar con el modelo de IA: {str(e)}",
            recommended_format="Post estático",
            suggested_keyword="SYSTEM",
        )


# ============================================================
#   B) ENDPOINT: GENERADOR DE SLIDES (CARRUSEL)
#   (/api/content/slide-generator)
# ============================================================

class SlideBlock(BaseModel):
    index: int
    title: str
    body: str
    cta: str


class SlideGeneratorRequest(BaseModel):
    platform: Literal["instagram", "linkedin"]
    topic: str
    phase: Literal["awareness", "consideration", "conversion", "retention"]
    slides: int = 5
    style: Literal["mkt360", "professional", "motivational", "direct", "friendly"] = "mkt360"
    language: Optional[str] = None


class SlideGeneratorResponse(BaseModel):
    topic: str
    phase: str
    slides: List[SlideBlock]
    global_caption: str
    keyword: str
    funnel_stage: str


@app.post("/api/content/slide-generator", response_model=SlideGeneratorResponse)
async def slide_generator(request: SlideGeneratorRequest):
    """
    Genera texto para carruseles siguiendo el Plan Clientes Nuevos.
    Devuelve slides con título, cuerpo y CTA por slide + caption global.
    """

    if request.language:
        lang = request.language
    else:
        # Por defecto: IG en español, LinkedIn en inglés
        lang = "es" if request.platform == "instagram" else "en"

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        # Fallback simple
        fake_slide = SlideBlock(
            index=1,
            title="AI.D sin modelo configurado",
            body="Configura la variable OPENAI_API_KEY o un modelo local para activar el generador de slides.",
            cta="Escribe SYSTEM para activar tu funnel."
        )
        return SlideGeneratorResponse(
            topic=request.topic,
            phase=request.phase,
            slides=[fake_slide],
            global_caption="AI.D aún no tiene modelo avanzado configurado.",
            keyword="SYSTEM",
            funnel_stage=request.phase,
        )

    if request.style == "mkt360":
        tone = (
            "Usa tono MKT 360: inteligente, rápido, profesional, cero vendehumo, con enfoque en sistemas y resultados."
        )
    elif request.style == "professional":
        tone = "Tono profesional, claro y ejecutivo."
    elif request.style == "motivational":
        tone = "Tono motivador, inspirador pero concreto."
    elif request.style == "direct":
        tone = "Tono directo, sin rodeos."
    else:
        tone = "Tono cercano y amigable."

    user_prompt = f"""
Plataforma: {request.platform}
Tema central del carrusel: {request.topic}
Fase del funnel (Plan Clientes Nuevos): {request.phase}
Número de slides: {request.slides}
Idioma: {lang}

Reglas:
- Slide 1: impacto / problema / pregunta.
- Slides intermedios: errores, explicación simple, comparativas o mini-caso según la fase.
- Última slide: CTA claro + palabra clave única para el funnel.
- Párrafos cortos (1–2 líneas).
- Mantén el tono: {tone}

Devuelve SIEMPRE un JSON válido con esta estructura:

{{
  "slides": [
    {{
      "index": 1,
      "title": "Título slide 1",
      "body": "Texto breve de la slide 1 (1–3 frases cortas).",
      "cta": "CTA específico de esta slide o vacío si no aplica."
    }}
  ],
  "global_caption": "Caption para acompañar el carrusel, con palabras clave y CTA final.",
  "keyword": "PALABRA_CLAVE_CORTA (ej. SYSTEM, AUDIT, GROWTH).",
  "funnel_stage": "awareness / consideration / conversion / retention"
}}

No agregues texto fuera del JSON.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Eres AI.D, experto en creación de carruseles estratégicos siguiendo el Plan Clientes Nuevos. "
                "Estructuras slides claras, accionables y con CTA en línea con el funnel."
            )
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            fake_slide = SlideBlock(
                index=1,
                title="Error al procesar JSON",
                body=raw,
                cta="Reintenta la petición."
            )
            return SlideGeneratorResponse(
                topic=request.topic,
                phase=request.phase,
                slides=[fake_slide],
                global_caption="No se pudo parsear la respuesta del modelo.",
                keyword="SYSTEM",
                funnel_stage=request.phase,
            )

        slides_list: List[SlideBlock] = []
        for s in parsed.get("slides", []):
            try:
                slides_list.append(
                    SlideBlock(
                        index=int(s.get("index", len(slides_list) + 1)),
                        title=str(s.get("title", "")),
                        body=str(s.get("body", "")),
                        cta=str(s.get("cta", "")),
                    )
                )
            except Exception:
                continue

        if not slides_list:
            slides_list.append(
                SlideBlock(
                    index=1,
                    title="Sin slides generadas",
                    body="No fue posible generar slides a partir de la respuesta del modelo.",
                    cta="Reintenta la petición."
                )
            )

        return SlideGeneratorResponse(
            topic=request.topic,
            phase=request.phase,
            slides=slides_list,
            global_caption=str(parsed.get("global_caption", "")),
            keyword=str(parsed.get("keyword", "")),
            funnel_stage=str(parsed.get("funnel_stage", request.phase)),
        )

    except Exception as e:
        fake_slide = SlideBlock(
            index=1,
            title="Error de conexión con IA",
            body=str(e),
            cta="Reintenta la petición."
        )
        return SlideGeneratorResponse(
            topic=request.topic,
            phase=request.phase,
            slides=[fake_slide],
            global_caption="Hubo un error al conectar con el modelo de IA.",
            keyword="SYSTEM",
            funnel_stage=request.phase,
        )


# ============================================================
#   C) ENDPOINT: MAPA DE FUNNEL / MINI SISTEMA
#   (/api/content/funnel-map)
# ============================================================

class FunnelStage(BaseModel):
    name: str
    goal: str
    content_role: str
    message_example: str


class FunnelMapRequest(BaseModel):
    topic: str
    platform: Literal["instagram", "linkedin", "tiktok", "facebook", "youtube", "generic"]
    objective: Literal["awareness", "engagement", "leads", "sales", "mixed"] = "mixed"
    language: Optional[str] = None


class FunnelMapResponse(BaseModel):
    topic: str
    platform: str
    objective: str
    keyword: str
    stages: List[FunnelStage]
    auto_messages: List[str]
    sequence_24h: List[str]
    metrics_focus: List[str]


@app.post("/api/content/funnel-map", response_model=FunnelMapResponse)
async def funnel_map(request: FunnelMapRequest):
    """
    Genera un mini-embudo automático para un tema de contenido:
    - palabra clave
    - etapas (Awareness / Consideration / Conversion / Retention)
    - mensajes automáticos
    - secuencia sugerida de 24h
    """

    lang = request.language or DEFAULT_LANG or "es"

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        return FunnelMapResponse(
            topic=request.topic,
            platform=request.platform,
            objective=request.objective,
            keyword="SYSTEM",
            stages=[
                FunnelStage(
                    name="Awareness",
                    goal="Llamar la atención sobre el problema principal.",
                    content_role="Educar rápido y mostrar el error clave.",
                    message_example="¿Quieres dejar de publicar sin ver resultados reales?"
                )
            ],
            auto_messages=[
                "Gracias por escribir SYSTEM. Cuéntame en una frase qué vendes y en qué país estás."
            ],
            sequence_24h=[
                "T0h: Respuesta automática con preguntas clave.",
                "T4h: Enviar recordatorio suave si no responde.",
                "T24h: Cerrar con propuesta de diagnóstico express."
            ],
            metrics_focus=[
                "Comentarios con palabra clave",
                "Respuestas al mensaje automático",
                "Diagnósticos completados"
            ],
        )

    user_prompt = f"""
Tema central del contenido: {request.topic}
Plataforma: {request.platform}
Objetivo principal: {request.objective}
Idioma: {lang}

Genera un mini-sistema tipo funnel basado en el Plan Clientes Nuevos.

Devuelve SIEMPRE un JSON válido con esta estructura:

{{
  "keyword": "PALABRA_CLAVE_CORTA (ej. SYSTEM, AUDIT, GROWTH).",
  "stages": [
    {{
      "name": "Awareness",
      "goal": "Objetivo de esta etapa.",
      "content_role": "Qué hace el contenido aquí.",
      "message_example": "Ejemplo de mensaje / ángulo."
    }},
    {{
      "name": "Consideration",
      "goal": "...",
      "content_role": "...",
      "message_example": "..."
    }},
    {{
      "name": "Conversion",
      "goal": "...",
      "content_role": "...",
      "message_example": "..."
    }},
    {{
      "name": "Retention",
      "goal": "...",
      "content_role": "...",
      "message_example": "..."
    }}
  ],
  "auto_messages": [
    "Mensaje automático 1 cuando alguien comenta la palabra clave.",
    "Mensaje automático 2 de seguimiento."
  ],
  "sequence_24h": [
    "Paso 1 dentro de las primeras horas.",
    "Paso 2...",
    "Paso 3..."
  ],
  "metrics_focus": [
    "Métrica 1",
    "Métrica 2"
  ]
}}

No incluyas texto fuera del JSON.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Eres AI.D, arquitecto de funnels para PYMEs y emprendedores. "
                "Conviertes un solo contenido en un mini sistema comercial: atraer, activar, diagnosticar y cerrar."
            )
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            return FunnelMapResponse(
                topic=request.topic,
                platform=request.platform,
                objective=request.objective,
                keyword="SYSTEM",
                stages=[
                    FunnelStage(
                        name="Error",
                        goal="No se pudo parsear el JSON.",
                        content_role="Revisa la respuesta cruda del modelo.",
                        message_example=raw,
                    )
                ],
                auto_messages=[],
                sequence_24h=[],
                metrics_focus=[],
            )

        stages_list: List[FunnelStage] = []
        for s in parsed.get("stages", []):
            try:
                stages_list.append(
                    FunnelStage(
                        name=str(s.get("name", "")),
                        goal=str(s.get("goal", "")),
                        content_role=str(s.get("content_role", "")),
                        message_example=str(s.get("message_example", "")),
                    )
                )
            except Exception:
                continue

        if not stages_list:
            stages_list.append(
                FunnelStage(
                    name="Awareness",
                    goal="Iniciar el interés.",
                    content_role="Educar rápido.",
                    message_example="Descubre por qué tu contenido no convierte."
                )
            )

        return FunnelMapResponse(
            topic=request.topic,
            platform=request.platform,
            objective=request.objective,
            keyword=str(parsed.get("keyword", "")),
            stages=stages_list,
            auto_messages=[str(m) for m in parsed.get("auto_messages", [])],
            sequence_24h=[str(s) for s in parsed.get("sequence_24h", [])],
            metrics_focus=[str(m) for m in parsed.get("metrics_focus", [])],
        )

    except Exception as e:
        return FunnelMapResponse(
            topic=request.topic,
            platform=request.platform,
            objective=request.objective,
            keyword="SYSTEM",
            stages=[
                FunnelStage(
                    name="Error",
                    goal="Fallo de conexión con IA.",
                    content_role="Mostrar detalle técnico.",
                    message_example=str(e),
                )
            ],
            auto_messages=[],
            sequence_24h=[],
            metrics_focus=[],
        )


# ============================================================
#   D) ENDPOINT: PREDICTOR DE ANUNCIOS
#   (/api/ads/predictor)
# ============================================================

class AdsPredictorRequest(BaseModel):
    platform: Literal["instagram", "facebook", "tiktok", "linkedin", "youtube", "generic"]
    objective: Literal["awareness", "traffic", "leads", "sales", "engagement"]
    headline: str
    primary_text: str
    description: Optional[str] = None
    cta: Optional[str] = None
    daily_budget: float
    currency: str = "USD"
    language: Optional[str] = None


class AdsPredictorResponse(BaseModel):
    platform: str
    objective: str
    estimated_ctr: float
    estimated_cpc: float
    estimated_cpm: float
    expected_result_summary: str
    risk_level: str
    recommendations: List[str]


@app.post("/api/ads/predictor", response_model=AdsPredictorResponse)
async def ads_predictor(request: AdsPredictorRequest):
    """
    Predice rendimiento estimado de un anuncio según texto + objetivo.
    No reemplaza pruebas reales, pero da una estimación orientativa.
    """

    lang = request.language or DEFAULT_LANG or "es"

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        # Valores ficticios conservadores
        return AdsPredictorResponse(
            platform=request.platform,
            objective=request.objective,
            estimated_ctr=1.0,
            estimated_cpc=1.5,
            estimated_cpm=8.0,
            expected_result_summary="AI.D está en modo sin modelo avanzado. Usa estos números solo como referencia.",
            risk_level="medio",
            recommendations=[
                "Configura OPENAI_API_KEY o un modelo local para activar la predicción avanzada.",
                "Prueba distintas creatividades y audiencias para reducir el riesgo.",
            ],
        )

    user_prompt = f"""
Plataforma: {request.platform}
Objetivo: {request.objective}
Idioma: {lang}
Presupuesto diario: {request.daily_budget} {request.currency}

Headline:
{request.headline}

Primary text:
{request.primary_text}

Descripción adicional:
{request.description or "Sin descripción adicional."}

CTA:
{request.cta or "Sin CTA específico."}

Analiza este anuncio y devuelve SIEMPRE un JSON válido con esta estructura:

{{
  "estimated_ctr": 0.0,
  "estimated_cpc": 0.0,
  "estimated_cpm": 0.0,
  "expected_result_summary": "Explica en 4–6 líneas qué se puede esperar del anuncio con este presupuesto.",
  "risk_level": "bajo | medio | alto",
  "recommendations": [
    "Recomendación 1 para mejorar antes de lanzar.",
    "Recomendación 2..."
  ]
}}

No escribas nada fuera del JSON.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Eres AI.D, media buyer senior. Estimas rangos de CTR, CPC y CPM basados en benchmarks, "
                "pero aclaras que son aproximaciones y das recomendaciones prácticas."
            )
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            return AdsPredictorResponse(
                platform=request.platform,
                objective=request.objective,
                estimated_ctr=0.8,
                estimated_cpc=1.2,
                estimated_cpm=7.0,
                expected_result_summary=f"No se pudo parsear el JSON. Respuesta cruda: {raw}",
                risk_level="medio",
                recommendations=[
                    "Revisa manualmente la respuesta cruda del modelo.",
                    "Ajusta el anuncio y haz un test A/B con bajo presupuesto.",
                ],
            )

        return AdsPredictorResponse(
            platform=request.platform,
            objective=request.objective,
            estimated_ctr=float(parsed.get("estimated_ctr", 0.0)),
            estimated_cpc=float(parsed.get("estimated_cpc", 0.0)),
            estimated_cpm=float(parsed.get("estimated_cpm", 0.0)),
            expected_result_summary=str(parsed.get("expected_result_summary", "")),
            risk_level=str(parsed.get("risk_level", "")),
            recommendations=[str(r) for r in parsed.get("recommendations", [])],
        )

    except Exception as e:
        return AdsPredictorResponse(
            platform=request.platform,
            objective=request.objective,
            estimated_ctr=0.0,
            estimated_cpc=0.0,
            estimated_cpm=0.0,
            expected_result_summary=f"Error al conectar con el modelo de IA: {str(e)}",
            risk_level="alto",
            recommendations=[
                "Lanza el anuncio con presupuesto mínimo y monitorea datos reales.",
                "Revisa segmentación, creatividad y oferta antes de escalar.",
            ],
        )


# ============================================================
#   E) ENDPOINT: GUIÓN DE VIDEO (REELS / TIKTOK / SHORTS)
#   (/api/content/video-script)
# ============================================================

class VideoBeat(BaseModel):
    second_from: int
    second_to: int
    action: str
    on_screen_text: str


class VideoScriptRequest(BaseModel):
    platform: Literal["instagram", "tiktok", "youtube", "facebook", "generic"]
    topic: str
    duration_seconds: int = 30
    objective: Literal["awareness", "engagement", "leads", "sales", "authority", "mixed"] = "awareness"
    style: Literal["mkt360", "professional", "motivational", "direct", "friendly"] = "mkt360"
    language: Optional[str] = None


class VideoScriptResponse(BaseModel):
    topic: str
    duration_seconds: int
    hook: str
    script: str
    beats: List[VideoBeat]
    closing_cta: str
    broll_ideas: List[str]


@app.post("/api/content/video-script", response_model=VideoScriptResponse)
async def video_script(request: VideoScriptRequest):
    """
    Genera un guion 9:16 para Reels/TikTok/Shorts en X segundos:
    - hook
    - guion completo
    - beats por segundos
    - CTA final
    - ideas de B-roll
    """

    if request.language:
        lang = request.language
    else:
        # Por defecto: TikTok/IG en español, YouTube en inglés
        if request.platform in ["youtube"]:
            lang = "en"
        else:
            lang = "es"

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        return VideoScriptResponse(
            topic=request.topic,
            duration_seconds=request.duration_seconds,
            hook="Configura OPENAI_API_KEY o un modelo local para activar el generador de guiones.",
            script="",
            beats=[],
            closing_cta="Escribe SYSTEM para recibir tu diagnóstico cuando AI.D esté activo.",
            broll_ideas=[
                "Capturas de pantalla de métricas.",
                "Texto grande en pantalla con tu hook."
            ],
        )

    if request.style == "mkt360":
        tone = (
            "Tono MKT 360: estratégico, profesional, rápido, con foco en sistemas y resultados reales."
        )
    elif request.style == "professional":
        tone = "Tono profesional, claro, directo."
    elif request.style == "motivational":
        tone = "Tono motivador, pero sin humo."
    elif request.style == "direct":
        tone = "Tono directo y contundente."
    else:
        tone = "Tono cercano y amigable."

    user_prompt = f"""
Plataforma: {request.platform}
Tema: {request.topic}
Duración: {request.duration_seconds} segundos
Objetivo: {request.objective}
Idioma: {lang}
Tono: {tone}

Estructura del video:
- 0–3s: Hook fuerte para detener scroll.
- Cuerpo: explicación simple, 1 idea central, ejemplos rápidos.
- Últimos 3–5s: CTA claro con palabra clave y próxima acción.

Devuelve SIEMPRE un JSON válido con esta estructura:

{{
  "hook": "Frase inicial de máximo 8 palabras para detener el scroll.",
  "script": "Guion completo del video, con indicaciones claras para la persona que habla.",
  "beats": [
    {{
      "second_from": 0,
      "second_to": 3,
      "action": "Qué debe ocurrir en pantalla.",
      "on_screen_text": "Texto breve en pantalla (si aplica)."
    }}
  ],
  "closing_cta": "Frase final con CTA y palabra clave.",
  "broll_ideas": [
    "Idea de B-roll 1",
    "Idea de B-roll 2"
  ]
}}

No agregues texto fuera del JSON.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Eres AI.D, guionista experto en videos cortos 9:16 para redes sociales. "
                "Creas guiones accionables, concretos y alineados a objetivos de marketing."
            )
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            return VideoScriptResponse(
                topic=request.topic,
                duration_seconds=request.duration_seconds,
                hook="Error al procesar JSON del modelo.",
                script=raw,
                beats=[],
                closing_cta="Revisa el contenido crudo y ajusta manualmente.",
                broll_ideas=[],
            )

        beats_list: List[VideoBeat] = []
        for b in parsed.get("beats", []):
            try:
                beats_list.append(
                    VideoBeat(
                        second_from=int(b.get("second_from", 0)),
                        second_to=int(b.get("second_to", 0)),
                        action=str(b.get("action", "")),
                        on_screen_text=str(b.get("on_screen_text", "")),
                    )
                )
            except Exception:
                continue

        return VideoScriptResponse(
            topic=request.topic,
            duration_seconds=request.duration_seconds,
            hook=str(parsed.get("hook", "")),
            script=str(parsed.get("script", "")),
            beats=beats_list,
            closing_cta=str(parsed.get("closing_cta", "")),
            broll_ideas=[str(i) for i in parsed.get("broll_ideas", [])],
        )

    except Exception as e:
        return VideoScriptResponse(
            topic=request.topic,
            duration_seconds=request.duration_seconds,
            hook="Error de conexión con IA.",
            script=str(e),
            beats=[],
            closing_cta="Intenta generar de nuevo el guion.",
            broll_ideas=[],
        )


# ============================================================
#   CEREBRO CENTRAL DE AI.D
#   (/api/chat/aid)
# ============================================================

class AIDChatHistoryMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AIDCoreChatRequest(BaseModel):
    message: str                               # lo que el usuario pregunta ahora
    goal: Optional[str] = None                 # objetivo del usuario (opcional)
    mode: Literal[
        "generic",         # conversación general
        "ads_planning",    # planificación de campañas
        "content",         # ideas y mejora de contenido
        "strategy",        # estrategia de negocio/marketing
        "support"          # soporte / acompañamiento
    ] = "generic"
    language: Optional[str] = None             # idioma preferido
    history: Optional[List[AIDChatHistoryMessage]] = None  # historial opcional


class AIDCoreChatResponse(BaseModel):
    reply: str
    mode: str
    used_language: str
    suggestions: List[str]        # ideas concretas / recomendaciones
    next_actions: List[str]       # próximos pasos accionables
    fallback: bool                # True si hubo que usar modo degradado


@app.post("/api/chat/aid", response_model=AIDCoreChatResponse)
async def aid_core_chat(request: AIDCoreChatRequest):
    """
    Cerebro central de AI.D.

    Este endpoint está pensado para:
      - Conversaciones generales con AI.D
      - Estrategia de campañas
      - Ideas y optimización de contenido
      - Decisiones de negocio y marketing
      - Soporte y acompañamiento para el usuario

    Siempre devuelve:
      - una respuesta principal (reply)
      - una lista de sugerencias
      - una lista de próximos pasos accionables
    """

    lang = request.language or DEFAULT_LANG or "es"

    base_identity = (
        "Eres AI.D, el cerebro central de la plataforma Ad.AI y de cualquier app "
        "donde se te instale. Tienes experiencia simulada de más de 40 años en "
        "marketing digital, publicidad, estrategia de negocio y creación de contenido. "
        "Tu prioridad es dar respuestas claras, accionables, cero vendehumo, "
        "orientadas a resultados reales, especialmente para PYMEs y emprendedores."
    )

    if request.mode == "ads_planning":
        mode_context = (
            "Modo: ads_planning. Enfócate en campañas pagadas (Meta, Google, LinkedIn, "
            "TikTok, etc.), distribución de presupuesto, objetivos, audiencias y creatividades."
        )
    elif request.mode == "content":
        mode_context = (
            "Modo: content. Enfócate en ideas de contenido, hooks, guiones, titulares, "
            "formatos para redes sociales y cómo diferenciar la marca."
        )
    elif request.mode == "strategy":
        mode_context = (
            "Modo: strategy. Enfócate en estrategia general de negocio, funnels, "
            "ofertas, posicionamiento y priorización de acciones."
        )
    elif request.mode == "support":
        mode_context = (
            "Modo: support. Enfócate en acompañar al usuario, ordenar sus ideas, "
            "bajar la ansiedad y convertir sus dudas en un plan simple de acción."
        )
    else:
        mode_context = "Modo: generic. Responde de forma equilibrada entre estrategia y acción."

    # Si proveedor es OpenAI y no hay clave → fallback
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        fallback_reply = (
            "Ahora mismo AI.D está en modo básico (sin conexión al modelo avanzado). "
            "Igual puedo ayudarte a ordenar tus ideas. Escríbeme 1) tu contexto, "
            "2) qué quieres lograr en los próximos 30 días y 3) qué tienes disponible hoy."
        )

        return AIDCoreChatResponse(
            reply=fallback_reply,
            mode=request.mode,
            used_language=lang,
            suggestions=[
                "Describe brevemente tu negocio o proyecto.",
                "Cuenta qué estás haciendo hoy en marketing o contenido.",
                "Di qué resultado te gustaría ver en 30 días."
            ],
            next_actions=[
                "Escribe esos tres puntos y vuelve a enviarlos a AI.D.",
                "Prioriza un solo objetivo principal para empezar.",
            ],
            fallback=True,
        )

    history_block = ""
    if request.history:
        joined = []
        for m in request.history:
            prefix = "Usuario" if m.role == "user" else "AI.D" if m.role == "assistant" else "Sistema"
            joined.append(f"{prefix}: {m.content}")
        history_block = "\n\nHistorial reciente:\n" + "\n".join(joined)

    user_prompt = f"""
Idioma de respuesta: {lang}.

Identidad base:
{base_identity}

Contexto de modo:
{mode_context}

Objetivo declarado (si existe):
{request.goal or "No especificado."}
{history_block}

Mensaje actual del usuario:
{request.message}

Devuelve SIEMPRE un JSON válido con esta estructura EXACTA:

{{
  "reply": "Respuesta principal para el usuario, en máximo 10–14 líneas, clara, directa y accionable.",
  "suggestions": [
    "Sugerencia concreta 1",
    "Sugerencia concreta 2"
  ],
  "next_actions": [
    "Próximo paso 1 muy específico",
    "Próximo paso 2 muy específico",
    "Próximo paso 3 muy específico"
  ]
}}

No incluyas explicación ni texto fuera del JSON.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "Eres AI.D, cerebro central de Ad.AI. Eres estratégico, práctico, "
                "respetuoso, motivador y cero vendehumo. Siempre bajas las ideas "
                "a pasos concretos."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = await call_llm_messages(messages, model="gpt-4.1-mini")

        if raw is None:
            raise RuntimeError("LLM sin respuesta")

        try:
            parsed = json.loads(raw)
        except Exception:
            return AIDCoreChatResponse(
                reply=raw,
                mode=request.mode,
                used_language=lang,
                suggestions=[],
                next_actions=[],
                fallback=True,
            )

        return AIDCoreChatResponse(
            reply=str(parsed.get("reply", "")),
            mode=request.mode,
            used_language=lang,
            suggestions=[str(s) for s in parsed.get("suggestions", [])],
            next_actions=[str(a) for a in parsed.get("next_actions", [])],
            fallback=False,
        )

    except Exception as e:
        return AIDCoreChatResponse(
            reply=(
                "Hubo un error al conectar con el modelo de IA. "
                "Puedes intentar de nuevo en unos minutos."
            ),
            mode=request.mode,
            used_language=lang,
            suggestions=[],
            next_actions=[f"Detalle técnico (para el equipo): {str(e)}"],
            fallback=True,
        )