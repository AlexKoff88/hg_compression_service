# Hugging Face Model Optimization Service with NNCF
This is a model optimization service that handles requests via REST API and can be used with Hugging Face Spaces to host the model optimization.

# Usage

## Start service
1. Create a secret key for personal data encryption when using the service.
```python
from cryptography.fernet import Fernet
import base64

key = Fernet.generate_key()
print(key)
```
2. Save the generated key into the "/etc/nncf-service/secret.key" file.
3. You can start `service.py` standalone. For this you should create a Python environment and install dependencies from `requirements.txt` file.
4. Alternatively, you can create a Linux service that will call `start_service.sh` script on start.

## Use client example
1. Login to HugginFace Hub using `huggingface_cli login` command and your HF token.
2. Create a similar Python environment as for service part 
3. Create environment variables
    - "access_token"
    - "enc_key"
4. Run `client_example.py` with model_id parameter, for example:
```sh
python client_example.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
```