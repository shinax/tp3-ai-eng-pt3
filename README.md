# tp3-ai-eng-pt3

Proyecto de sistema multi-agente AI para clasificación y enrutamiento de consultas.

## Objetivo

Construir un orquestador que clasifique consultas en dominios (HR, IT Support, Finance, Legal) y enrute cada consulta a un agente RAG especializado usando LangChain. El flujo debe trazarse con Langfuse y se incluye un evaluador de calidad.

## Estructura

- `requirements.txt` - dependencias del proyecto.
- `src/config.py` - configuración de OpenAI y Langfuse.
- `src/data_loader.py` - carga documentos, crea embeddings y retrievers FAISS.
- `src/orchestrator.py` - clasificador de dominios y enrutador a agentes RAG.
- `src/evaluator.py` - agente evaluador para puntuar la calidad de la respuesta.
- `src/main.py` - flujo de ejemplo interactivo.

## Uso

1. Crear un entorno virtual e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

2. Copiar la plantilla de entorno y definir variables:

```bash
cp .env.example .env
```

Editar `.env` con tus credenciales:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGFUSE_API_KEY=your_langfuse_api_key_here
LANGFUSE_BASE_URL=https://api.langfuse.com
LANGFUSE_PROJECT_NAME=tp3-ai-eng-pt3
OPENAI_MODEL=gpt-3.5-turbo
RETRIEVER_K=4
```

3. Ejecutar el ejemplo interactivo:

```bash
source .venv/bin/activate
python -m src.main
```

4. Probar consultas definidas:

```bash
python3 -m json.tool test_queries.json
```

Esto muestra las preguntas de prueba que cubren los dominios HR, IT Support, Finance y Legal.

## Diseño técnico

- Se usa `langchain.chains.RetrievalQA` para los agentes RAG, con reintentos de documentación interna y políticas de respuesta basadas únicamente en evidencias.
- La clasificación de dominio se realiza con un `LLMChain` estructurado para generar JSON con la etiqueta del área.
- `LangfuseTracer` se inyecta mediante callbacks para instrumentar todo el flujo y poder depurar rutas, consultas y respuestas.
- El evaluador produce un puntaje de calidad 1-10 para cada respuesta, permitiendo detectar respuestas de baja calidad antes de ser entregadas.

## Notas

- Si falta un documento de dominio, el programa fallará con `FileNotFoundError`.
- El prototipo incluye datos de ejemplo en `data/`.
