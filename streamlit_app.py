
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional
from PIL import Image
from pydantic import BaseModel, Field
import streamlit as st
from ultralytics import YOLO

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph

# Librerías de LangChain y Neo4j
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jGraph

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
st.set_page_config(page_title="Evaluador de Ensambles Snow Bike", layout="wide")

@st.cache_resource
def load_yolo():
    return YOLO("best.pt")


@st.cache_resource
def load_neo4j():
    url = st.secrets["NEO4J_URI"].strip()
    username = st.secrets["NEO4J_USER"].strip()
    password = st.secrets["NEO4J_PASSWORD"].strip()

    # Diagnóstico de Secrets (no imprime la contraseña por seguridad)
    st.write("🔍 **Diagnóstico de Conexión Neo4j:**")
    st.write(f"- **URI en Secrets:** `{st.secrets.get('NEO4J_URI')}`")
    st.write(f"- **Usuario en Secrets:** `{st.secrets.get('NEO4J_USER')}`")
    st.write(f"- **¿Contraseña presente?:** `{'Sí' if 'NEO4J_PASSWORD' in st.secrets else 'No'}`")

    return Neo4jGraph(
        url=url,
        username=username,
        password=password,
        refresh_schema=False,
    )


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
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
    """Extrae la clave del ensamble (ej: assembly_3) y despliega el GIF al 50% de tamaño."""
    assembly_id = extract_assembly_id(raw_text)

    if assembly_id:
        gif_path = GIFS_DIR / f"{assembly_id}.gif"
        if gif_path.exists():
            col_gif, _ = st.columns([1, 1])
            with col_gif:
                st.image(str(gif_path), caption=caption, use_container_width=True)
        else:
            st.info(f"ℹ️ No se encontró el archivo de animación: `{gif_path}`")
    else:
        st.info(
            f"ℹ️ No se pudo extraer una clave válida (ej. 'assembly_1') del texto: '{raw_text}'"
        )


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
    // 1. Encontrar los ensambles que requieren CUALQUIERA de los elementos detectados
    MATCH (target:Assembly)-[:REQUIRES]->(req)
    WHERE req.id IN $classes OR req.name IN $classes
    
    // 2. Obtener TODOS los componentes que exige ese ensamble objetivo
    MATCH (target)-[:REQUIRES]->(all_req)
    
    // 3. Buscar cuál es el siguiente paso si este ensamble se completa
    OPTIONAL MATCH (target)-[:NEXT_STEP]->(next_step)
    
    WITH target, 
         collect(DISTINCT req.id) AS componentes_presentes,
         collect(DISTINCT all_req.id) AS componentes_totales,
         next_step.id AS siguiente_paso
         
    RETURN 
        target.id AS EnsambleObjetivo,
        target.name AS NombreEnsamble,
        componentes_presentes AS ComponentesPresentes,
        componentes_totales AS ComponentesTotalesRequeridos,
        [x IN componentes_totales WHERE NOT x IN $classes] AS ComponentesFaltantes,
        (size(componentes_presentes) = size(componentes_totales)) AS ListoParaEnsamblar,
        siguiente_paso AS SiguientePasoSiSeCompleta
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
def analyze_assembly_with_graph(detected_classes, graph, llm, user_instructions):
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
1. 'ensambles_posibles': Debe contener únicamente las claves del 'EnsambleObjetivo' indicado en el JSON de Neo4j si 'ListoParaEnsamblar' es true (ejemplo estricto: ['assembly_3']).
2. 'etapa_actual': Debe ser ÚNICAMENTE el identificador del 'EnsambleObjetivo' (ejemplo estricto: 'assembly_3'). NO agregues texto como "construcción exitosa...".
3. 'piezas_faltantes': Usa exactamente la lista 'ComponentesFaltantes' de Neo4j.
4. 'resumen_tecnico': Explica cómo la combinación de las piezas detectadas (ej. 'assembly_2' + 'engine') permite formar exitosamente el 'EnsambleObjetivo' (ej. 'assembly_3'), y menciona cuál sería el 'SiguientePasoSiSeCompleta' ('assembly_4')."""

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
# 4. INTERFAZ PRINCIPAL DE STREAMLIT
# ==========================================
def main():
    st.title("🛠️ Sistema Inteligente de Análisis de Ensambles (Snow Bike)")
    st.write(
        "Carga la imagen de una pieza o ensamble para evaluar la secuencia con"
        " YOLO, Neo4j y Llama 3.1."
    )

    with st.sidebar:
        st.header("⚙️ Configuración del Sistema")
        st.subheader("1. System Prompting")
        user_prompt_extra = st.text_area(
            "Instrucciones técnicas adicionales para el LLM:",
            placeholder=(
                "Ejemplo: Sé extremadamente estricto con los requisitos de torque y"
                " seguridad. Si falta un perno clave, indícalo."
            ),
            help=(
                "Texto adicional que se le inyectará al prompt antes de razonar con"
                " el grafo."
            ),
        )

        uploaded_file = st.file_uploader(
            "Sube una imagen del ensamble",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_uploader",
        )

    if "last_processed_file" not in st.session_state:
        st.session_state["last_processed_file"] = None
        st.session_state["analysis_data"] = None

    # Procesamiento con Loader cuando hay una nueva imagen subida
    if (
        uploaded_file is not None
        and uploaded_file.name != st.session_state["last_processed_file"]
    ):
        with st.spinner(
            "Procesando imagen con YOLO, consultando Neo4j y evaluando con Llama 3.1..."
        ):
            model = load_yolo()
            graph = load_neo4j()
            llm = load_llm()

            image = Image.open(uploaded_file)
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
                analysis = analyze_assembly_with_graph(
                    valid_classes, graph, llm, user_prompt_extra
                )

            st.session_state["last_processed_file"] = uploaded_file.name
            st.session_state["analysis_data"] = {
                "image": image,
                "annotated_image": annotated_image,
                "output_json": output_json,
                "raw_detections": raw_detections,
                "valid_detections": valid_detections,
                "valid_classes": valid_classes,
                "analysis": analysis,
            }

    # Pestañas normales
    tab1, tab2 = st.tabs(["📷 1. Detección y Modelo", "🧠 2. Análisis LLM y Animación"])

    data = st.session_state.get("analysis_data")

    # ---------------------------------------------------------
    # TAB 1: SUBIDA DE IMAGEN, DETECCIÓN Y JSON DE COINCIDENCIAS
    # ---------------------------------------------------------
    with tab1:
        st.subheader("1. Carga de Imagen e Inferencia del Modelo YOLO")

        if data:
            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    data["image"], caption="Imagen cargada", use_container_width=True
                )

            with col2:
                st.image(
                    data["annotated_image"],
                    caption="Detecciones del Modelo",
                    channels="BGR",
                    use_container_width=True,
                )

            st.divider()

            st.subheader("📄 Coincidencias y Detecciones Válidas (JSON)")
            st.json(
                {
                    "total_detecciones_raw": len(data["raw_detections"]),
                    "clases_filtradas": data["valid_classes"],
                    "detalle": data["valid_detections"],
                }
            )
        else:
            st.info(
                "👆 Por favor sube una imagen desde el menú lateral para iniciar el procesamiento."
            )

    # ---------------------------------------------------------
    # TAB 2: RESPUESTA DEL LLM Y TUTORIAL ANIMADO (GIF)
    # ---------------------------------------------------------
    with tab2:
        st.subheader("2. Evaluación Estructurada (GraphRAG + LLM)")

        if data:
            analysis: Optional[AnalisisEnsambleSchema] = data.get("analysis")

            if analysis:
                st.success("Análisis Completado")

                if analysis.es_ensamble_valido:
                    st.badge(" Ensamble Válido / En Proceso", icon="✅")
                else:
                    st.badge(" Ensamble Inválido / Piezas Aisladas", icon="⚠️")

                st.markdown(f"**Etapa Actual:** {analysis.etapa_actual}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Ensambles Detectados/Posibles:**")
                    if analysis.ensambles_posibles:
                        for ens in analysis.ensambles_posibles:
                            st.write(f"- {ens}")
                    else:
                        st.write("Ninguno")

                with c2:
                    st.markdown("**Piezas Faltantes Requeridas:**")
                    if analysis.piezas_faltantes:
                        for p in analysis.piezas_faltantes:
                            st.write(f"- {p}")
                    else:
                        st.write("Ninguna")

                st.markdown(f"**Resumen Técnico:**\n{analysis.resumen_tecnico}")

                st.divider()

                # Clave del ensamble obtenida desde 'ensambles_posibles' o 'etapa_actual'
                target_raw = ""
                if analysis.ensambles_posibles and len(analysis.ensambles_posibles) > 0:
                    target_raw = analysis.ensambles_posibles[0]
                elif analysis.etapa_actual:
                    target_raw = analysis.etapa_actual

                # Desplegable del Tutorial GIF (Oculto si hay piezas faltantes)
                with st.expander(
                    "🎬 Ver/Ocultar Tutorial de Ensamble (GIF)", expanded=True
                ):
                    if analysis.piezas_faltantes and len(analysis.piezas_faltantes) > 0:
                        st.warning(
                            "⚠️ No se muestra el tutorial de animación porque faltan piezas para completar el ensamble."
                        )
                    elif target_raw:
                        target_id = extract_assembly_id(target_raw)
                        render_assembly_gif(
                            target_raw,
                            caption=f"Paso a paso de ensamble: {target_id or target_raw}",
                        )
                    else:
                        st.write(
                            "No hay ensamble objetivo asignado para mostrar el tutorial."
                        )

                # Desplegable del JSON Puro del LLM
                with st.expander("🔍 Ver JSON Estructurado del LLM"):
                    st.json(analysis.model_dump())
            else:
                st.warning(
                    "No se identificaron piezas o ensambles válidos para consultar en el grafo."
                )
        else:
            st.info(
                "👈 Sube una imagen para ver los resultados del LLM y los GIF tutoriales."
            )


if __name__ == "__main__":
    main()
