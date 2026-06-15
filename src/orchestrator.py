from typing import Any, Dict

from langchain_classic.chains import LLMChain, RetrievalQA
from langchain_classic.chains.question_answering import load_qa_chain
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

    def _build_classifier_chain(self) -> LLMChain:
        prompt = PromptTemplate(
            template=(
                "Eres un enrutador de consultas. Clasifica la consulta del cliente en uno de estos dominios:"
                " HR, IT Support, Finance, Legal. Responde solo con la etiqueta exacta en formato JSON.\n"
                "{format_instructions}\n"
                "Consulta: {query}\n"
            ),
            input_variables=["query"],
            partial_variables={"format_instructions": self.classifier_parser.get_format_instructions()},
        )
        return LLMChain(
            llm=self.llm,
            prompt=prompt,
        )

    def _build_qa_chains(self) -> Dict[str, RetrievalQA]:
        qa_chains: Dict[str, RetrievalQA] = {}
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
            # Manually build the QA chain to have full control
            combine_documents_chain = load_qa_chain(
                self.llm, chain_type="stuff", prompt=prompt
            )
            qa_chains[domain] = RetrievalQA(
                retriever=retriever,
                combine_documents_chain=combine_documents_chain,
                return_source_documents=True,
                input_key="query",
                output_key="result",
            )
        return qa_chains
        return qa_chains

    def classify_domain(self, query: str) -> str:
        result = self.classifier.predict(query=query)
        parsed = self.classifier_parser.parse(result)
        department = parsed.get("department", "HR")
        return department if department in self.qa_chains else "HR"

    def route_query(self, query: str) -> Dict[str, Any]:
        domain = self.classify_domain(query)
        if domain not in self.qa_chains:
            domain = "HR"

        qa_chain = self.qa_chains[domain]
        response = qa_chain({"query": query})
        answer = response.get("result")
        documents = response.get("source_documents", [])
        sources = [doc.metadata.get("source", "document") for doc in documents]

        return {
            "domain": domain,
            "answer": answer,
            "sources": sources,
        }
