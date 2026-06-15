#!/usr/bin/env python
"""Test runner para evaluar el sistema multiagente RAG."""

import json
from pathlib import Path
from src.config import Settings
from src.data_loader import build_domain_retrievers
from src.orchestrator import Orchestrator
from src.evaluator import ResponseEvaluator


def run_tests():
    """Ejecuta las queries de prueba y reporta resultados."""
    settings = Settings()
    settings.validate()

    # Cargar queries de prueba
    test_file = Path("test_queries.json")
    with open(test_file, "r", encoding="utf-8") as f:
        test_queries = json.load(f)

    # Inicializar componentes
    from src.main import ensure_data_files

    ensure_data_files()
    retrievers = build_domain_retrievers(settings)
    orchestrator = Orchestrator(settings, retrievers)
    evaluator = ResponseEvaluator(settings)

    # Ejecutar tests
    print("=" * 80)
    print("PRUEBAS DEL SISTEMA MULTIAGENTE RAG")
    print("=" * 80)
    print()

    passed = 0
    failed = 0
    results = []

    for idx, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_domain = test_case["expected_domain"]

        print(f"\n[Test {idx}/{len(test_queries)}]")
        print(f"Consulta: {query}")
        print(f"Dominio esperado: {expected_domain}")

        try:
            # Ejecutar orquestador
            result = orchestrator.route_query(query)
            domain = result["domain"]
            answer = (
                result["answer"][:100] + "..."
                if len(result["answer"]) > 100
                else result["answer"]
            )

            # Ejecutar evaluador
            evaluation = evaluator.evaluate(query, result["answer"])
            score = evaluation.get("score", 0)

            # Verificar clasificación
            is_correct = domain == expected_domain
            status = "✓ PASS" if is_correct else "✗ FAIL"

            print(f"Dominio clasificado: {domain} {status}")
            print(f"Puntuación: {score}/10")
            print(f"Respuesta: {answer}")

            results.append(
                {
                    "query": query,
                    "expected_domain": expected_domain,
                    "classified_domain": domain,
                    "score": score,
                    "correct": is_correct,
                }
            )

            if is_correct:
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append(
                {
                    "query": query,
                    "expected_domain": expected_domain,
                    "error": str(e),
                    "correct": False,
                }
            )
            failed += 1

    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN DE PRUEBAS")
    print("=" * 80)
    print(f"Total: {len(test_queries)}")
    print(f"Pasadas: {passed}")
    print(f"Fallidas: {failed}")
    accuracy = (passed / len(test_queries)) * 100
    print(f"Precisión: {accuracy:.1f}%")
    print("=" * 80)

    # Detalles por dominio
    domains = {}
    for result in results:
        if "correct" in result and "classified_domain" in result:
            domain = result["expected_domain"]
            if domain not in domains:
                domains[domain] = {"total": 0, "correct": 0}
            domains[domain]["total"] += 1
            if result["correct"]:
                domains[domain]["correct"] += 1

    print("\nPrecisión por dominio:")
    for domain in sorted(domains.keys()):
        stats = domains[domain]
        acc = (stats["correct"] / stats["total"]) * 100
        print(f"  {domain}: {stats['correct']}/{stats['total']} ({acc:.0f}%)")


if __name__ == "__main__":
    run_tests()
