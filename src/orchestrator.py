from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_classic.prompts import PromptTemplate

from .config import Settings

DOMAIN_PROMPTS = {
    "HR": "Eres un asistente especializado en Recursos Humanos. Responde con base en la documentación interna de RRHH.",
    "IT Support": "Eres un asistente especializado en soporte técnico. Responde con base en la documentación de IT Support.",
    "Finance": "Eres un asistente especializado en Finanzas. Responde con base en la documentación financiera.",
    "Legal": "Eres un asistente especializado en Legal. Responde con base en la documentación legal.",
}


class Orchestrator:
    def __init__(self, settings: Settings, retrievers: Dict[str, Any]):
        self.settings = settings
        self.callback_manager = settings.create_callback_manager()
        self.llm = ChatOpenAI(
            model_name=self.settings.model_name,
            temperature=0.0,
            verbose=False,
        )
        self.retrievers = retrievers
        self.classifier_parser = self._build_classifier_parser()
        self.classifier = self._build_classifier_chain()
        self.qa_chains = self._build_qa_chains()

    def _build_classifier_parser(self) -> StructuredOutputParser:
        schemas = [
            ResponseSchema(
                name="department",
                description="Etiqueta exacta del dominio: HR, IT Support, Finance, Legal.",
            )
        ]
        return StructuredOutputParser.from_response_schemas(schemas)

    def _build_classifier_chain(self) -> Any:
        # Add a few-shot examples to reduce misclassification (especially Finance vs Legal)
        examples = (
            "Ejemplos:\n"
            "- '¿Cuántos días de vacaciones me corresponden este año?' -> HR\n"
            "- 'Mi laptop no arranca y muestra error de disco, ¿qué debo hacer?' -> IT Support\n"
            "- '¿Puedo reclamar viáticos sin factura?' -> Finance\n"
            "- '¿Cómo gestionamos reclamaciones de empleados con el área legal?' -> Legal\n"
            "- '¿Cuál es el proceso para presentar un presupuesto anual?' -> Finance\n"
            "- '¿Podemos usar herramientas de terceros para backups?' -> IT Support\n\n"
        )

        prompt = PromptTemplate(
            template=(
                "Eres un enrutador de consultas. Clasifica la consulta del cliente en uno de estos dominios:"
                " HR, IT Support, Finance, Legal. Responde solo con la etiqueta exacta en formato JSON.\n"
                "{format_instructions}\n"
                "{examples}\n"
                "Consulta: {query}\n"
            ),
            input_variables=["query"],
            partial_variables={
                "format_instructions": self.classifier_parser.get_format_instructions(),
                "examples": examples,
            },
        )
        # Build a runnable prompt->llm sequence and return it. We'll call
        # `invoke()` on this runnable when classifying so we avoid using
        # the deprecated `LLMChain` class and `__call__` API.
        classifier_chain = prompt | self.llm
        return classifier_chain

    def _build_qa_chains(self) -> Dict[str, Any]:
        qa_chains: Dict[str, Any] = {}
        for domain, retriever in self.retrievers.items():
            system_prompt = DOMAIN_PROMPTS.get(domain, "")
            prompt = PromptTemplate(
                template=(
                    "{system_prompt}\n\nUsa solo la documentación recuperada para responder. "
                    "Si la respuesta no está respaldada por la evidencia, di que no hay suficiente información.\n\n"
                    "Contexto:\n{context}\n\nPregunta: {question}\nRespuesta:"
                ),
                input_variables=["context", "question"],
                partial_variables={"system_prompt": system_prompt},
            )
            # Build a simple prompt->llm runnable for combining documents.
            combine_documents_chain = prompt | self.llm

            # Simple wrapper implementing a minimal RetrievalQA-like interface
            class SimpleRetrievalQA:
                def __init__(self, retriever, combine_chain):
                    self.retriever = retriever
                    self.combine_documents_chain = combine_chain

                def invoke(self, inputs: Dict[str, Any]):
                    query = inputs.get("query")
                    # Support multiple retriever APIs for compatibility
                    docs = None
                    if hasattr(self.retriever, "get_relevant_documents"):
                        try:
                            docs = self.retriever.get_relevant_documents(query)
                        except Exception:
                            docs = None
                    if docs is None:
                        # Some retrievers implement `invoke` which accepts a
                        # plain string query and returns a list of Documents.
                        try:
                            docs = self.retriever.invoke(query)
                        except Exception:
                            docs = []
                    context = "\n\n".join([d.page_content for d in docs])
                    out = self.combine_documents_chain.invoke(
                        {"context": context, "question": query}
                    )
                    if isinstance(out, dict):
                        # try common keys
                        result = out.get("text") or out.get("result") or str(out)
                    else:
                        result = str(out)
                    return {"result": result, "source_documents": docs}

            qa_chains[domain] = SimpleRetrievalQA(
                retriever=retriever, combine_chain=combine_documents_chain
            )

        return qa_chains

    def classify_domain(self, query: str) -> str:
        # Fast rule: finance-related keywords -> Finance (helps avoid misclassification)
        ql = query.lower()
        finance_keywords = [
            "viático",
            "viaticos",
            "viáticos",
            "factura",
            "reembolso",
            "gastos",
            "viatico",
            "presupuesto",
            "presupuestos",
            "presupuesto anual",
        ]
        if any(kw in ql for kw in finance_keywords) and "Finance" in self.qa_chains:
            return "Finance"

        # Call the runnable classifier and parse output safely.
        try:
            raw = self.classifier.invoke({"query": query})
        except Exception:
            raw = None

        if isinstance(raw, dict):
            result = raw.get("text") or raw.get("result") or str(raw)
        else:
            result = str(raw) if raw is not None else ""

        # Try structured parse first; fall back to heuristics on parse failure.
        try:
            parsed = self.classifier_parser.parse(result)
            department = parsed.get("department", "HR")
            if department in self.qa_chains:
                return department
        except Exception:
            pass

        # Heuristic fallback based on keywords
        ql = ql
        it_keywords = [
            "laptop",
            "pantalla",
            "error",
            "wifi",
            "red",
            "arranca",
            "conexión",
            "backup",
            "backups",
            "respaldo",
            "copias",
            "copia de seguridad",
            "servidor",
        ]
        legal_keywords = ["legal", "contrato", "reclam", "derecho", "demanda"]
        hr_keywords = [
            "vacacion",
            "vacaciones",
            "liquidación",
            "finiquito",
            "sueldo",
            "nómina",
            "nomina",
        ]

        if any(kw in ql for kw in it_keywords) and "IT Support" in self.qa_chains:
            return "IT Support"
        if any(kw in ql for kw in legal_keywords) and "Legal" in self.qa_chains:
            return "Legal"
        if any(kw in ql for kw in hr_keywords) and "HR" in self.qa_chains:
            return "HR"

        # Default
        return "HR"

    def route_query(self, query: str) -> Dict[str, Any]:
        domain = self.classify_domain(query)
        if domain not in self.qa_chains:
            domain = "HR"

        qa_chain = self.qa_chains[domain]
        response = qa_chain.invoke({"query": query})
        answer = response.get("result")
        documents = response.get("source_documents", [])
        sources = [doc.metadata.get("source", "document") for doc in documents]

        return {
            "domain": domain,
            "answer": answer,
            "sources": sources,
        }
