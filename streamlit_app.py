
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
                "User-Agent": "StreamlitCloudApp"
            }
        }
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


def render_assembly_gif(raw_text: str, caption: str = ""):
    """Extrae la clave del ensamble (ej: assembly_3) y despliega el GIF centrado."""
    assembly_id = extract_assembly_id(raw_text)

    if assembly_id:
        gif_path = GIFS_DIR / f"{assembly_id}.gif"
        if gif_path.exists():
            st.image(str(gif_path), caption=caption, use_container_width=True)
        else:
            st.info(f"ℹ️ No se encontró la guía animada para: `{assembly_id}`")
    else:
        st.info(f"ℹ️ No se pudo extraer una clave válida de ensamble.")


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

    print("\n" + "=" * 60)
    print("🔍 [LOG NEO4J] SECUENCIA Y RELACIONES EXTRAÍDAS DEL GRAFO")
    print("=" * 60)
    if results:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("❌ No se encontraron coincidencias en Neo4j para:", detected_classes)
    print("=" * 60 + "\n")

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

    response = chain.invoke(
        {
            "user_instructions": (
                user_instructions
                if user_instructions
                else "Verifica el orden estricto de la secuencia."
            ),
            "detected_classes": detected_classes,
            "graph_context": graph_context_json,
        }
    )

    return response


# ==========================================
# 4. INTERFAZ PRINCIPAL DE STREAMLIT (SIMPLIFICADA)
# ==========================================
def main():
    st.title("🛠️ Asistente Inteligente de Ensamble")
    st.caption("Captura o sube una imagen para obtener la recomendación de ensamble.")

    # 1. Selector de fuente de imagen directo en la pantalla principal
    source_type = st.radio(
        "Selecciona el método de entrada:",
        ["📸 Usar Cámara", "📁 Subir Imagen"],
        horizontal=True,
    )

    uploaded_file = None
    camera_file = None

    if source_type == "📸 Usar Cámara":
        camera_file = st.camera_input("Toma la foto de las piezas actuales")
    else:
        uploaded_file = st.file_uploader(
            "Selecciona un archivo de imagen",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

    active_image_source = camera_file if camera_file is not None else uploaded_file

    # Gestión de Estado
    if "last_processed_file" not in st.session_state:
        st.session_state["last_processed_file"] = None
        st.session_state["analysis_data"] = None

    current_file_id = None
    if camera_file is not None:
        current_file_id = f"camera_{camera_file.file_id}"
    elif uploaded_file is not None:
        current_file_id = f"file_{uploaded_file.name}"

    # 2. Procesamiento
    if (
        active_image_source is not None
        and current_file_id != st.session_state["last_processed_file"]
    ):
        with st.spinner("Analizando piezas y consultando el proceso de ensamble..."):
            model = load_yolo()
            graph = load_neo4j()
            llm = load_llm()

            image = Image.open(active_image_source)
            output_json, annotated_image = run_inference(model, image)

            raw_detections = output_json.get("detections", [])
            valid_detections = []

            for d in raw_detections:
                cls_name = d.get("class_name")
                conf = d.get("confidence", 0.0)
                required_thresh = CLASS_THRESHOLDS.get(cls_name, 0.50)

                if conf >= required_thresh:
                    valid_detections.append(d)

            valid_classes = list(set([d["class_name"] for d in valid_detections]))

            analysis = None
            if valid_classes:
                analysis = analyze_assembly_with_graph(valid_classes, graph, llm)

            st.session_state["last_processed_file"] = current_file_id
            st.session_state["analysis_data"] = {
                "valid_classes": valid_classes,
                "analysis": analysis,
            }

    # 3. Mostrar Resultados (Solo si hay datos procesados)
    data = st.session_state.get("analysis_data")
    if data:
        st.divider()
        analysis: Optional[AnalisisEnsambleSchema] = data.get("analysis")

        if analysis:
            # Caso A: El ensamble es posible y no faltan piezas requeridas
            if analysis.es_ensamble_valido and not analysis.piezas_faltantes:
                st.success(
                    f"✅ **Siguiente Ensamble Listo:** `{analysis.etapa_actual}`"
                )
                st.markdown(f"**Instrucción del Proceso:**\n{analysis.resumen_tecnico}")

                target_raw = (
                    analysis.ensambles_posibles[0]
                    if analysis.ensambles_posibles
                    else analysis.etapa_actual
                )

                st.subheader("🎬 Tutorial de Ensamble")
                render_assembly_gif(target_raw, caption=f"Paso a paso: {target_raw}")

            # Caso B: Faltan piezas para poder armar la siguiente etapa
            else:
                st.error("⚠️ **No es posible realizar un nuevo ensamble aún.**")

                if analysis.piezas_faltantes:
                    st.warning("**Piezas faltantes para continuar:**")
                    for p in analysis.piezas_faltantes:
                        st.write(f"- 🔴 {p}")

                st.info(f"**Detalle del estado:**\n{analysis.resumen_tecnico}")
        else:
            st.warning(
                "⚠️ No se identificaron piezas suficientes o válidas para sugerir un ensamble."
            )


if __name__ == "__main__":
    main()
