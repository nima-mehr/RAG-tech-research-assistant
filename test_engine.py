from services.rag_engine import RAGEngine

engine = RAGEngine()

result = engine.ask("Which VPN protocol is faster?")

print("\nAnswer:\n")
print(result["answer"])

print("\nSources:\n")

for i, source in enumerate(result["sources"], 1):
    print(f"Source {i}:")
    print(source)
    print("-" * 60)