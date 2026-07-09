"""chat.py — interactive terminal chat with Gemini, grounded in your knowledge base.

Run `python ingest.py` first, then:  python chat.py
"""

import sys

from google.genai import types

import rag  # importing rag loads config, which loads .env

try:
    store = rag.load_store()
except FileNotFoundError as err:
    sys.exit(str(err))

try:
    client = rag.make_client()
except RuntimeError as err:
    sys.exit(str(err))

chat = client.chats.create(
    model=rag.CHAT_MODEL,
    config=types.GenerateContentConfig(
        system_instruction=rag.SYSTEM_INSTRUCTION,
        temperature=0.2,
    ),
)


def ask(question: str) -> tuple[str, list[dict]]:
    hits = rag.retrieve(store, client, question)
    if not hits:
        return (
            "I don't have information about that in my knowledge base.",
            hits,
        )
    context = rag.build_context(hits)
    response = chat.send_message(f"CONTEXT:\n{context}\n\nQUESTION: {question}")
    return response.text, hits


def main() -> None:
    print(f"UFC/MMA RAG chat — {rag.CHAT_MODEL} over {len(store['entries'])} KB chunks.")
    print('Type a question, or "exit" to quit.\n')

    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        try:
            text, hits = ask(question)
            print(f"\nbot > {text}\n")
            if hits:
                sources = list(dict.fromkeys(f"{h['source']} ({h['score']:.2f})" for h in hits))
                print(f"      [retrieved: {', '.join(sources)}]\n")
        except Exception as err:  # keep the chat loop alive on API errors
            print(f"Error: {err}")


if __name__ == "__main__":
    main()
