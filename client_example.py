import json
import os
import sys

import requests
from cryptography.fernet import Fernet
from huggingface_hub import whoami


enc_key = os.getenv("enc_key")
access_token = os.getenv("access_token")
username = whoami(access_token)["name"]


def optimize_model(model_id, awq=False, scale_estimation=True, group_size=128, dataset="wikitext2"):
    url = "http://localhost:5000/optimize"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_id": model_id,
        "dtype": "4-bit",
        "awq": awq,
        "scale_estimation": scale_estimation,
        "group_size": group_size,
        "dataset": dataset,
        "username": username,
        "access_token": access_token,
        "private_repo": False,
        "overwritte": True,
    }

    fernet = Fernet(enc_key)
    encrypted = fernet.encrypt(json.dumps(payload, indent=2).encode())

    response = requests.post(url, headers=headers, data=encrypted)

    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)


if __name__ == "__main__":
    # Example usage
    optimize_model(model_id=sys.argv[1], awq=True, scale_estimation=True, group_size=64, dataset="wikitext2")
