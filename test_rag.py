from services.rag import RAG


rag = RAG()

question = "Which VPN protocol is faster?"

answer = rag.ask(question)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)