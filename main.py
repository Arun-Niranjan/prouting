import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")


def main():
    print("Hello from prouting!")

    embeddings = embeddings_model.embed_documents(
        [
            "Hi there!",
            "Oh, hello!",
            "What's your name?",
            "My friends call me World",
            "Hello World!",
        ]
    )
    print(len(embeddings), len(embeddings[0]))


if __name__ == "__main__":
    main()
