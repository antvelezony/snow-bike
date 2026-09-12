import json
import re
from pathlib import Path
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel, Field
import streamlit as st
from ultralytics import YOLO

from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph
from langchain_ollama import ChatOllama

from datetime import datetime
import requests

# ==========================================
# RUTA DE GIFS Y CONFIGURACIÓN DE UMBRALES
# ==========================================
GIFS_DIR = Path("gifs")

CLASS_THRESHOLDS = {
    "assembly_1": 0.55,
    "assembly_2": 0.55,
    "assembly_3": 0.55,
    "assembly_4": 0.55,
    "assembly_5": 0.55,
    "assembly_6": 0.55,
    "assembly_7": 0.55,
    "engine": 0.60,
    "frame": 0.55,
    "handlebar": 0.55,
    "headlight": 0.60,
    "seat": 0.60,
    "ski": 0.50,
    "sproket": 0.60,
    "track": 0.50,
}


# ==========================================
# 0. ESQUEMA DE SALIDA ESTRUCTURADA (PYDANTIC)
# ==========================================
class AnalisisEnsambleSchema(BaseModel):
    ensambles_posibles: List[str] = Field(
        description=(
            "Lista de IDs de ensambles objetivos que se PUEDEN FORMAR o completar con"
            " los insumos detectados (ej: ['assembly_3']). Usar estrictamente el formato 'assembly_X'."
        )
    )
    etapa_actual: str = Field(
        description=(
            "Identificador exacto del Ensamble Objetivo (ejemplo estricto: 'assembly_3')."
            " NO agregar frases descriptivas, solo la clave 'assembly_X'."
        )
    )
    piezas_faltantes: List[str] = Field(
        description=(
            "Lista exacta de componentes faltantes para completar el ensamble"
            " objetivo según Neo4j."
        )
    )
    es_ensamble_valido: bool = Field(
        description=(
            "True si 'ListoParaEnsamblar' en el grafo es true o si la combinación"
            " de piezas es válida."
        )
    )
    resumen_tecnico: str = Field(
        description=(
            "Explicación detallada de cómo las piezas detectadas se unen para"
            " formar el ensamble objetivo y cuál es el siguiente paso."
        )
    )


# ==========================================
# 1. CONFIGURACIÓN DE PÁGINA Y RECURSOS
# ==========================================
st.set_page_config(page_title="Asistente de Ensamble Snow Bike", layout="centered")


@st.cache_resource
def load_yolo():
    return YOLO("best.pt")


@st.cache_resource
def load_neo4j():
    url = st.secrets.get("NEO4J_URI", "").strip()
    username = st.secrets.get("NEO4J_USER", "").strip()
    password = st.secrets.get("NEO4J_PASSWORD", "").strip()
    database = st.secrets.get("NEO4J_DATABASE", "").strip()

    if not url or not password:
        st.error("⚠️ Faltan las credenciales de Neo4j en los Secrets de Streamlit.")
        st.stop()

    return Neo4jGraph(
        url=url,
        username=username,
        password=password,
        database=database,
        refresh_schema=False,
    )


@st.cache_resource
def load_ollama():
    ollama_url = st.secrets.get("OLLAMA_BASE_URL", "").strip()
    model_name = st.secrets.get("OLLAMA_MODEL", "").strip()

    return ChatOllama(
        base_url=ollama_url,
        model=model_name,
        temperature=0.2,
        client_kwargs={
            "headers": {
                "ngrok-skip-browser-warning": "true",
                "User-Agent": "StreamlitCloudApp",
            }
        },
    )


# ==========================================
# HELPERS PARA MANEJO DE GIFS Y NOMBRES
# ==========================================
def extract_assembly_id(text: str) -> Optional[str]:
    """Extrae el identificador tipo 'assembly_X' de cualquier texto retornado."""
    if not text:
        return None
    match = re.search(r"(assembly_\d+)", text.lower().strip())
    if match:
        return match.group(1)
    return None

def format_assembly_name(text: str) -> str:
    """Convierte 'assembly_1' en 'Ensamble 1' para mostrar en la interfaz."""
    if not text:
        return ""
    # Reemplaza 'assembly_' por 'Ensamble ' y capitaliza
    name = text.lower().replace("assembly_", "Ensamble ")
    return name.capitalize()


def render_assembly_gif(raw_text: str, caption: str = ""):
    """Extrae la clave del ensamble (ej: assembly_3) y despliega el GIF centrado."""
    assembly_id = extract_assembly_id(raw_text)

    if assembly_id:
        gif_path = GIFS_DIR / f"{assembly_id}.gif"
        if gif_path.exists():
            st.image(str(gif_path), caption=caption, use_container_width=True)
        else:
            st.info(f"ℹ️ No se encontró la guía animada para: `{format_assembly_name(assembly_id)}`")
    else:
        st.info(f"ℹ️ No se pudo extraer una clave válida de ensamble.")


def resetear_proceso():
    """Limpia la sesión e incrementa el contador de clave para forzar el reseteo del widget de captura."""
    current_key_id = st.session_state.get("widget_key_id", 0)
    st.session_state.clear()
    st.session_state["widget_key_id"] = current_key_id + 1
    st.rerun()


# ==========================================
# 2. LÓGICA DE INFERENCIA DE VISIÓN
# ==========================================
def run_inference(model, PIL_image):
    results = model.predict(source=PIL_image, conf=0.25, verbose=False)
    r = results[0]

    detections = []
    names = r.names

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id] if cls_id in names else str(cls_id)
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox_xyxy": [round(v, 2) for v in xyxy],
                }
            )

    output = {
        "model": "best.pt",
        "num_detections": len(detections),
        "detections": detections,
    }
    return output, r.plot()


def get_assembly_sequence_from_neo4j(graph, detected_classes):
    query = """
    MATCH (target:Assembly)-[:REQUIRES]->(req)
    WHERE req.id IN $classes OR req.name IN $classes
    
    MATCH (target)-[:REQUIRES]->(all_req)
    OPTIONAL MATCH (target)-[:NEXT_STEP]->(next_step)
    
    WITH target, 
         collect(DISTINCT req.id) AS componentes_presentes,
         collect(DISTINCT all_req.id) AS componentes_totales,
         next_step.id AS siguiente_paso
         
    WITH target, componentes_presentes, componentes_totales, siguiente_paso,
         (size(componentes_presentes) = size(componentes_totales)) AS listo
         
    RETURN 
        target.id AS EnsambleObjetivo,
        target.name AS NombreEnsamble,
        componentes_presentes AS ComponentesPresentes,
        componentes_totales AS ComponentesTotalesRequeridos,
        [x IN componentes_totales WHERE NOT x IN $classes] AS ComponentesFaltantes,
        listo AS ListoParaEnsamblar,
        siguiente_paso AS SiguientePasoSiSeCompleta
    ORDER BY ListoParaEnsamblar DESC, target.id ASC
    """
    results = graph.query(query, params={"classes": detected_classes})

    return results


# ==========================================
# 3. LÓGICA DE BÚSQUEDA EN NEO4J & LLM ESTRUCTURADO
# ==========================================
def analyze_assembly_with_graph(detected_classes, graph, llm, user_instructions=""):
    if not detected_classes:
        return None

    raw_sequence = get_assembly_sequence_from_neo4j(graph, detected_classes)

    if not raw_sequence:
        return None

    graph_context_json = json.dumps(raw_sequence, indent=2)

    template_str = """System: Eres un ingeniero experto en secuencias de manufactura y ensambles mecánicos de motos de nieve (Snow Bike).
Tu objetivo es determinar qué nuevo ensamble se puede formar a partir de las piezas/ensambles detectados en la imagen.

Instrucciones adicionales del usuario:
{user_instructions}

Human:
Piezas/Ensambles detectados en la imagen:
{detected_classes}

Evaluación previa del Grafo de Conocimiento (Neo4j):
{graph_context}

Reglas estrictas para generar la respuesta:
1. Analiza la lista 'graph_context' y FILTRA únicamente los objetos donde 'ListoParaEnsamblar' sea TRUE. Ignora cualquier objeto donde sea FALSE.
2. 'ensambles_posibles': Debe contener únicamente las claves del 'EnsambleObjetivo' indicado en el JSON de Neo4j si 'ListoParaEnsamblar' es true (ejemplo estricto: ['assembly_3']).
3. 'etapa_actual': Debe ser ÚNICAMENTE el identificador del 'EnsambleObjetivo' (ejemplo estricto: 'assembly_3'). NO agregues texto como "construcción exitosa...".
4. 'piezas_faltantes': Usa exactamente la lista 'ComponentesFaltantes' de Neo4j para el ensamble seleccionado.
5. 'resumen_tecnico': Explica cómo la combinación de las piezas detectadas permite formar el ensamble objetivo y qué paso sigue.
"""

    prompt_template = PromptTemplate(
        input_variables=[
            "user_instructions",
            "detected_classes",
            "graph_context",
        ],
        template=template_str,
    )

    structured_llm = llm.with_structured_output(AnalisisEnsambleSchema)
    chain = prompt_template | structured_llm

    return chain.invoke(
        {
            "user_instructions": user_instructions
            or "Verifica el orden estricto de la secuencia.",
            "detected_classes": detected_classes,
            "graph_context": graph_context_json,
        }
    )


def render_survey_view():
    st.balloons()
    st.title("📋 Encuesta de Evaluación del Proceso")
    st.success("🎉 **¡Ensamble final completado con éxito!**")
    st.markdown(
        "Por favor, completa la siguiente encuesta para registrar los datos del experimento."
    )

    ollama_url = st.secrets.get("OLLAMA_BASE_URL", "").strip()

    with st.form("encuesta_satisfaccion"):
        st.subheader("Información del Participante")
        nombre_participante = st.text_input(
            "Nombre completo o Identificador del participante:",
            placeholder="Ej. Juan Pérez",
        )

        st.divider()

        st.subheader("Sección 1: Carga Mental Percibida (NASA-TLX Simplificado)")
        st.caption("Escala de respuesta: 1 (Muy bajo / Muy fácil) a 5 (Muy alto / Muy difícil)")

        q1 = st.slider("1. Exigencia Mental...", 1, 5, 3)
        q2 = st.slider("2. Frustración...", 1, 5, 3)
        q3 = st.slider("3. Esfuerzo Físico...", 1, 5, 3)

        st.divider()

        st.subheader("Sección 2: Usabilidad y Satisfacción")
        q4 = st.radio("4. Claridad...", [1, 2, 3, 4, 5], horizontal=True, index=2)
        q5 = st.radio("5. Confianza...", [1, 2, 3, 4, 5], horizontal=True, index=2)
        q6 = st.radio("6. Ritmo...", [1, 2, 3, 4, 5], horizontal=True, index=2)
        q7 = st.radio("7. Preferencia...", [1, 2, 3, 4, 5], horizontal=True, index=2)

        st.divider()

        st.subheader("Sección 3: Retroalimentación")
        q8 = st.text_area("Puntos de Fricción...")
        q9 = st.text_area("Fricción Operativa (Grupo A)...")
        q10 = st.text_area("Sugerencias...")

        submitted = st.form_submit_button("Guardar y Finalizar Experimento", type="primary")

        if submitted:
            payload = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "nombre_participante": nombre_participante.strip(),
                    "participante_id": st.session_state.get("participante_id", "P_DESCONOCIDO"),
                    "grupo_asignado": st.session_state.get("grupo", "SIN_GRUPO"),
                },
                "respuestas": { "nasa": [q1, q2, q3], "sus": [q4, q5, q6, q7], "feedback": [q8, q9, q10] }
            }

            endpoint = f"{ollama_url.rstrip('/')}/guardar_encuesta"
            headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}

            try:
                response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    st.session_state["encuesta_completada"] = True
                    st.success("✅ Datos guardados localmente.")
                else:
                    st.error(f"⚠️ Error {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.session_state.get("encuesta_completada"):
        if st.button("🔄 Iniciar Nuevo Experimento"):
            resetear_proceso()


# ==========================================
# 4. INTERFAZ PRINCIPAL DE STREAMLIT
# ==========================================
def main():
    if "widget_key_id" not in st.session_state:
        st.session_state["widget_key_id"] = 0
    if "mostrar_encuesta" not in st.session_state:
        st.session_state["mostrar_encuesta"] = False
    if "last_processed_file" not in st.session_state:
        st.session_state["last_processed_file"] = None
    if "analysis_data" not in st.session_state:
        st.session_state["analysis_data"] = None

    if st.session_state["mostrar_encuesta"]:
        render_survey_view()
        return

    st.title("🛠️ Asistente Inteligente de Ensamble")
    
    source_type = st.radio("Selecciona el método de entrada:", ["📸 Usar Cámara", "📁 Subir Imagen"], horizontal=True)

    uploaded_file = None
    camera_file = None
    key_suffix = st.session_state["widget_key_id"]

    if source_type == "📸 Usar Cámara":
        camera_file = st.camera_input("Toma la foto", key=f"cam_{key_suffix}")
    else:
        uploaded_file = st.file_uploader("Sube imagen", type=["jpg", "png", "webp"], key=f"file_{key_suffix}")

    active_image_source = camera_file if camera_file is not None else uploaded_file
    current_file_id = f"{active_image_source.name}_{key_suffix}" if active_image_source else None

    if active_image_source and current_file_id != st.session_state["last_processed_file"]:
        with st.spinner("Analizando..."):
            model, graph, llm = load_yolo(), load_neo4j(), load_ollama()
            image = Image.open(active_image_source)
            output_json, annotated_image = run_inference(model, image)
            
            valid_classes = []
            for d in output_json.get("detections", []):
                if d.get("confidence", 0) >= CLASS_THRESHOLDS.get(d.get("class_name"), 0.5):
                    valid_classes.append(d["class_name"])
            
            analysis = analyze_assembly_with_graph(list(set(valid_classes)), graph, llm) if valid_classes else None
            st.session_state["last_processed_file"] = current_file_id
            st.session_state["analysis_data"] = {"analysis": analysis}

    data = st.session_state.get("analysis_data")
    if data:
        st.divider()
        analysis: Optional[AnalisisEnsambleSchema] = data.get("analysis")

        if analysis:
            target_id = extract_assembly_id(analysis.etapa_actual)
            # Traducimos el ID a un nombre amigable (Ensamble X)
            target_display_name = format_assembly_name(target_id)
            
            # Condición de éxito: Listo para ensamblar y sin piezas faltantes
            is_ready = analysis.es_ensamble_valido and not analysis.piezas_faltantes
            # Verificamos si es específicamente el paso 7
            is_final_step = target_id == "assembly_7"

            if is_ready:
                st.success(f"✅ **Siguiente paso listo:** `{target_display_name}`")
                st.markdown(f"**Instrucción:**\n{analysis.resumen_tecnico}")
                st.subheader("🎬 Tutorial animado")
                render_assembly_gif(target_id, caption=f"Guía para {target_display_name}")

                st.divider()
                
                # REQUERIMIENTO 2: El botón de encuesta SOLO aparece si es el Ensamble 7 listo
                if is_final_step:
                    if st.button("Finalizar proceso e ir a Encuesta ➔", type="primary"):
                        st.session_state["mostrar_encuesta"] = True
                        st.rerun()
                else:
                    if st.button("Continuar al siguiente paso ➔", type="primary"):
                        resetear_proceso()

            else:
                st.error(f"⚠️ **Aún no puedes completar el `{target_display_name}`**")
                if analysis.piezas_faltantes:
                    st.warning("**Piezas faltantes:**")
                    for p in analysis.piezas_faltantes:
                        st.write(f"- 🔴 {p}")
                st.info(f"**Estado actual:**\n{analysis.resumen_tecnico}")
                st.divider()
                if st.button("🔄 Reintentar captura", type="secondary"):
                    resetear_proceso()
        else:
            st.warning("⚠️ No se detectaron piezas suficientes.")
            if st.button("🔄 Reintentar", type="secondary"):
                resetear_proceso()

if __name__ == "__main__":
    main()
