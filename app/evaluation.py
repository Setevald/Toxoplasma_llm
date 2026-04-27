import requests

questions = [
    "How is toxoplasmosis transmitted?",
    "What organs does toxoplasma affect?"
]

for q in questions:
    r = requests.post(
        "http://localhost:8000/chat",
        json={"message": q}
    )

    print("\nQuestion:", q)
    print("Answer:", r.json()["response"])
