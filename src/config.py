import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_classic.chains.llm import CallbackManager

load_dotenv()


@dataclass
class Settings:
    load_dotenv()
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")
    langfuse_public_key: str = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = os.environ.get("LANGFUSE_PRIVATE_KEY")
    langfuse_base_url: str = os.environ.get("LANGFUSE_BASE_URL")
    langfuse_project_name: str = os.environ.get("LANGFUSE_PROJECT_NAME", "tp3")
    model_name: str = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    retriever_k: int = int(os.environ.get("RETRIEVER_K", "4"))

    def validate(self):
        if not self.openai_api_key:
            raise ValueError("Set OPENAI_API_KEY before running the project.")
        if not self.langfuse_public_key:
            raise ValueError("Set LANGFUSE_PUBLIC_KEY before running the project.")
        if not self.langfuse_secret_key:
            raise ValueError("Set LANGFUSE_SECRET_KEY before running the project.")

    def create_callback_manager(self):

        callbacks = []

        Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_PRIVATE_KEY"],
            host=os.environ["LANGFUSE_BASE_URL"],
        )

        callbacks.append(CallbackHandler())

        return CallbackManager(callbacks)
