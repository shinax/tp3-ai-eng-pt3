from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_classic.prompts import PromptTemplate

from .config import Settings


class ResponseEvaluator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.callback_manager = settings.create_callback_manager()
        # Do not pass callback handlers directly into the low-level SDK to
        # avoid unexpected kwarg forwarding. Use callbacks at invocation time.
        self.llm = ChatOpenAI(
            model_name=self.settings.model_name,
            temperature=0.0,
            verbose=False,
        )

        schemas = [
            ResponseSchema(name="score", description="Puntaje de calidad de 1 a 10."),
            ResponseSchema(
                name="rationale", description="Explicación breve de la evaluación."
            ),
        ]
        self.parser = StructuredOutputParser.from_response_schemas(schemas)
        # Build a runnable prompt -> llm sequence instead of LLMChain
        self.prompt = PromptTemplate(
            template=(
                "Eres un evaluador de calidad que puntúa respuestas con base en la pregunta original. "
                "Devuelve un JSON válido con el puntaje (1 a 10) y una razón breve.\n"
                "{format_instructions}\n\nPregunta: {question}\nRespuesta: {answer}\n"
            ),
            input_variables=["question", "answer"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.chain = self.prompt | self.llm

    def evaluate(self, question: str, answer: str) -> dict:
        out = self.chain.invoke({"question": question, "answer": answer})
        if isinstance(out, dict):
            result = out.get("text") or out.get("result") or str(out)
        else:
            result = str(out)
        try:
            parsed = self.parser.parse(result)
            try:
                parsed["score"] = int(parsed["score"])
            except (TypeError, ValueError):
                parsed["score"] = 0
            return parsed
        except Exception:
            # Fallback: return a minimal structure when parsing fails
            return {"score": 0, "rationale": result}
