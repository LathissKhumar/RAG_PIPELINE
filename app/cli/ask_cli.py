import typer
import requests

app = typer.Typer()

API_URL = "http://127.0.0.1:8000/ask"

@app.command()
def ask(question: str):
    """
    Sends the question to the FastAPI /ask endpoint.
    """
    try:
        response = requests.get(API_URL, params={"question": question})
        data = response.json()
        print("\nQuestion:", data.get("question"))
        print("Answer:", data.get("answer"), "\n")

    except Exception as e:
        print("Error contacting API:", e)


if __name__ == "__main__":
    app()
