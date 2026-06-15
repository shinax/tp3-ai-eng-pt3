from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .data_loader import build_domain_retrievers
from .orchestrator import Orchestrator
from .evaluator import ResponseEvaluator

load_dotenv()


def ensure_data_files():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    defaults = {
        "hr_docs.txt": "Política de vacaciones: los empleados tienen derecho a 20 días hábiles de licencia al año.\nPreguntas sobre nómina deben remitirse al área de Finanzas.\n",
        "it_docs.txt": "Procedimiento de soporte: reiniciar el equipo y verificar la conexión de red.\nEl equipo IT no gestiona solicitudes de nómina.\n",
        "finance_docs.txt": "La política de gastos permite reembolso de viáticos con factura válida.\nEl presupuesto anual se revisa en el primer trimestre.\n",
        "legal_docs.txt": "El área legal revisa contratos y cumplimiento normativo.\nLas consultas de reclamos laborales deben derivarse a RRHH cuando hay empleados involucrados.\n",
    }

    for file_name, content in defaults.items():
        file_path = data_dir / file_name
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")


def run_example():
    settings = Settings()
    settings.validate()

    ensure_data_files()
    retrievers = build_domain_retrievers(settings)
    orchestrator = Orchestrator(settings, retrievers)
    evaluator = ResponseEvaluator(settings)

    print("Sistema multiagent RAG con LangChain y Langfuse")
    print("Escribe una consulta o deja en blanco para salir.")

    while True:
        query = input("Consulta> ").strip()
        if not query:
            break

        result = orchestrator.route_query(query)
        evaluation = evaluator.evaluate(query, result["answer"])

        print("\nDominio:", result["domain"])
        print("Respuesta:\n", result["answer"])
        print("Fuentes:", result["sources"])
        print("Evaluación:", evaluation)
        print("\n---\n")


if __name__ == "__main__":
    run_example()
