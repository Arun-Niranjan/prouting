import getpass
import os

from prouter import Prouter

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

utterances = {
    "greetings": [
        "hello",
        "hi",
        "bye",
        "goodbye",
        "how are you?",
    ],
    "advice": [
        "I need help",
        "help me to X",
        "what advice can you give me?",
        "I want to X",
    ],
    "calendar": [
        "Book it for X time and Y place",
        "Check when I'm free",
        "my working hours are X",
        "find a free half hour",
        "today is...",
        "what's the date?",
    ],
}

if __name__ == "__main__":
    prouter = Prouter(utterances, embeddings_model)
    print(prouter.predict_route("when should we have lunch?"))
    # {'greetings': np.float64(0.303), 'advice': np.float64(0.308), 'calendar': np.float64(0.389)}
