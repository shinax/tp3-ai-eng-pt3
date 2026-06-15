from langchain_classic.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_classic.prompts import PromptTemplate

from .config import Settings


class ResponseEvaluator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.callback_manager = settings.create_callback_manager()
        self.llm = ChatOpenAI(
            model_name=self.settings.model_name,
            temperature=0.0,
            callbacks=self.callback_manager,
            verbose=False,
        )

        schemas = [
            ResponseSchema(name="score", description="Puntaje de calidad de 1 a 10."),
            ResponseSchema(name="rationale", description="Explicación breve de la evaluación.")
        ]
        self.parser = StructuredOutputParser.from_response_schemas(schemas)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=(
                    "Eres un evaluador de calidad que puntúa respuestas con base en la pregunta original. "
                    "Devuelve un JSON válido con el puntaje (1 a 10) y una razón breve.\n"
                    "{format_instructions}\n\nPregunta: {question}\nRespuesta: {answer}\n"
                ),
                input_variables=["question", "answer"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            ),
            callbacks=self.callback_manager,
        )

    def evaluate(self, question: str, answer: str) -> dict:
        result = self.chain.predict(question=question, answer=answer)
        parsed = self.parser.parse(result)
        try:
            parsed["score"] = int(parsed["score"])
        except (TypeError, ValueError):
            parsed["score"] = 0
        return parsed
