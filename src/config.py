import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()
from langfuse.langchain import CallbackHandler

@dataclass
class Settings:
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    langfuse_public_key: str = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.environ.get("LANGFUSE_SECRET_KEY", "")
    langfuse_base_url: str = os.environ.get("LANGFUSE_BASE_URL", os.environ.get("LANGFUSE_HOST", "https://api.langfuse.com"))
    langfuse_project_name: str = os.environ.get("LANGFUSE_PROJECT_NAME", "tp3-ai-eng-pt3")
    model_name: str = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    retriever_k: int = int(os.environ.get("RETRIEVER_K", "4"))

    def validate(self):
        if not self.openai_api_key:
            raise ValueError("Set OPENAI_API_KEY before running the project.")

    def create_callback_manager(self):
        # Create a CallbackManager compatible with the installed LangChain variant.
        # Prefer `langchain_classic` callback manager when available in this environment.
        try:
            # `langchain_classic` exposes a CallbackManager used by the LLM chains
            from langchain_classic.chains.llm import CallbackManager
        except Exception:
            # Fall back to the base callback manager if the above isn't present
            from langchain_classic.callbacks.base import BaseCallbackManager as CallbackManager

        callbacks = []

        # If Langfuse is installed and a public/secret key is provided, initialize
        # the Langfuse client so traces can be emitted. Langfuse does not provide
        # a direct `LangfuseTracer` inside LangChain; instead we prefer to use the
        # LangChain tracer if available and keep the Langfuse client for manual
        # instrumentation when needed.
        try:
            import langfuse
            langfuseHandler = CallbackHandler(
                public_key=self.langfuse_public_key,
                secret_key=self.langfuse_secret_key,
                host=self.langfuse_host,
            )

            # initialize client (will warn/disable if keys are missing)
            client = langfuse.get_client(public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"))

            # If langchain_classic provides a LangChainTracer, use it to capture
            # runs locally in LangChain format (this is a safe, available integration).
            try:
                from langchain_classic.callbacks.tracers import LangChainTracer

                callbacks.append(langfuseHandler.get_langchain_tracer())
            except Exception:
                # No LangChainTracer available; skip tracer setup but keep client
                # so users can call `client` directly for custom observations.
                pass
        except Exception:
            # Langfuse package not available or failed to initialize; continue
            # without adding a langfuse-specific tracer.
            pass

        # Construct and return the callback manager. `CallbackManager` expects the
        # list of handler instances as the first positional argument in this env.
        try:
            return CallbackManager(callbacks)
        except TypeError:
            # Some installs expect keyword `handlers`
            return CallbackManager(handlers=callbacks)
