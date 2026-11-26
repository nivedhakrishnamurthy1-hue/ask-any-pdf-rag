"""Command line interface:  python -m ragqa.cli build|ask|chat"""
import argparse

from ragqa import config


def show(result):
    print(f"\nAnswer: {result['answer']}")
    if result["pages"]:
        print("Pages:  " + ", ".join(str(p) for p in result["pages"]))
    if result.get("gated"):
        print("(no relevant passage found - the LLM was not called)")
    print("\nTop passages:")
    for h in result["sources"]:
        snippet = h["text"][:110].replace("\n", " ")
        print(f"  [p{h['page']} | score {h['score']:.2f}] {snippet}...")
    print()


def cmd_build(args):
    from ragqa.pipeline import build_from_pdf

    meta = build_from_pdf(args.pdf, args.chunk_words, args.chunk_overlap)
    print(f"\nIndexed {meta['pages']} pages into {meta['chunks']} chunks "
          f"({meta['chunk_words']} words, {meta['overlap']} overlap).")


def cmd_ask(args):
    from ragqa.answer import ask
    from ragqa.retriever import Retriever

    show(ask(args.question, Retriever.from_dir(), k=args.k))


def cmd_chat(args):
    from ragqa.answer import ask
    from ragqa.retriever import Retriever

    retriever = Retriever.from_dir()
    print("Ask questions about the document. Type 'exit' to quit.")
    while True:
        question = input("\nQuestion> ").strip()
        if question.lower() in {"exit", "quit", ""}:
            break
        show(ask(question, retriever, k=args.k))


def main():
    parser = argparse.ArgumentParser(prog="ragqa", description="PDF question answering with RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="read a PDF and build the search index")
    build.add_argument("--pdf", required=True)
    build.add_argument("--chunk-words", type=int, default=config.CHUNK_WORDS)
    build.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    build.set_defaults(func=cmd_build)

    ask_cmd = sub.add_parser("ask", help="ask one question")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("-k", type=int, default=config.TOP_K)
    ask_cmd.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="interactive question loop")
    chat.add_argument("-k", type=int, default=config.TOP_K)
    chat.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
