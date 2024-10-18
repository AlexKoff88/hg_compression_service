import json
import os
import shutil
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from transformers import AutoTokenizer

from cryptography.fernet import Fernet
from diffusers import ConfigMixin
from flask import Flask, jsonify, request
from huggingface_hub import HfApi, ModelCard
from huggingface_hub.file_download import repo_folder_name
from openvino import serialize
from openvino_tokenizers import convert_tokenizer
from optimum.exporters import TasksManager
from optimum.intel import (
    OVLatentConsistencyModelPipeline,
    OVModelForAudioClassification,
    OVModelForCausalLM,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForTokenClassification,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLPipeline,
    OVWeightQuantizationConfig,
)
from optimum.intel.utils.modeling_utils import _find_files_matching_pattern


_HEAD_TO_AUTOMODELS = {
    "feature-extraction": "OVModelForFeatureExtraction",
    "fill-mask": "OVModelForMaskedLM",
    "text-generation": "OVModelForCausalLM",
    "text-classification": "OVModelForSequenceClassification",
    "token-classification": "OVModelForTokenClassification",
    "question-answering": "OVModelForQuestionAnswering",
    "image-classification": "OVModelForImageClassification",
    "audio-classification": "OVModelForAudioClassification",
    "stable-diffusion": "OVStableDiffusionPipeline",
    "stable-diffusion-xl": "OVStableDiffusionXLPipeline",
    "latent-consistency": "OVLatentConsistencyModelPipeline",
}


SECRET_FILE_PATH = "/etc/nncf-service/secret.key"
file = open(SECRET_FILE_PATH, "rb")
enc_key = file.read()
file.close()

app = Flask(__name__)


def quantize_model(
    model_id: str,
    dtype: str,
    username: str,
    oauth_token: str,
    calibration_dataset: str = None,
    ratio: str = 1.0,
    awq: bool = False,
    scale_estimation: bool = False,
    group_size: int = 128,
    private_repo: bool = False,
    overwritte: bool = True,
):
    if not model_id:
        return f"### Invalid input 🐞 Please specify a model name, got {model_id}"

    try:
        model_name = model_id.split("/")[-1]
        w_t = dtype.replace("-", "")
        suffix = f"{w_t}" if model_name.endswith("openvino") else f"openvino-{w_t}"
        new_repo_id = f"{username}/{model_name}-{suffix}"
        library_name = TasksManager.infer_library_from_model(model_id, token=oauth_token)

        if library_name == "diffusers":
            ConfigMixin.config_name = "model_index.json"
            class_name = ConfigMixin.load_config(model_id, token=oauth_token)["_class_name"].lower()
            if "xl" in class_name:
                task = "stable-diffusion-xl"
            elif "consistency" in class_name:
                task = "latent-consistency"
            else:
                task = "stable-diffusion"
        else:
            task = TasksManager.infer_task_from_model(model_id, token=oauth_token)

        if task == "text2text-generation":
            return "Export of Seq2Seq models is currently disabled."

        if task not in _HEAD_TO_AUTOMODELS:
            return f"The task '{task}' is not supported, only {_HEAD_TO_AUTOMODELS.keys()} tasks are supported"

        auto_model_class = _HEAD_TO_AUTOMODELS[task]
        ov_files = _find_files_matching_pattern(
            model_id,
            pattern=r"(.*)?openvino(.*)?\_model.xml",
            use_auth_token=oauth_token,
        )
        export = len(ov_files) == 0

        if calibration_dataset == "None":
            calibration_dataset = None

        is_int8 = dtype == "8-bit"
        # if library_name == "diffusers":
        # quant_method = "hybrid"
        if not is_int8 and calibration_dataset is not None:
            quant_method = "awq"
        else:
            if calibration_dataset is not None:
                print("Default quantization was selected, calibration dataset won't be used")
            quant_method = "default"

        quantization_config = OVWeightQuantizationConfig(
            bits=8 if is_int8 else 4,
            awq=awq,
            scale_estimation=scale_estimation,
            quant_method=quant_method,
            dataset=None if quant_method == "default" else calibration_dataset,
            ratio=1.0 if is_int8 else ratio,
            group_size=group_size,
            num_samples=None if quant_method == "default" else 20,
        )

        api = HfApi(token=oauth_token)
        if api.repo_exists(new_repo_id) and not overwritte:
            return (
                f"Model {new_repo_id} already exist, please tick the overwritte box to push on an existing repository"
            )

        with TemporaryDirectory() as d:
            folder = os.path.join(d, repo_folder_name(repo_id=model_id, repo_type="models"))
            os.makedirs(folder)

            try:
                api.snapshot_download(repo_id=model_id, local_dir=folder, allow_patterns=["*.json"])
                ov_model = eval(auto_model_class).from_pretrained(
                    model_id,
                    export=export,
                    cache_dir=folder,
                    token=oauth_token,
                    quantization_config=quantization_config,
                )
                ov_model.save_pretrained(folder)

                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    ov_tokenizer, ov_detokenizer = convert_tokenizer(
                        tokenizer, with_detokenizer=True, skip_special_tokens=True
                    )
                    serialize(ov_tokenizer, folder + "/openvino_tokenizer.xml")
                    serialize(ov_detokenizer, folder + "/openvino_detokenizer.xml")
                except Exception as e:
                    print("Cannot export tokenizer. Details:\n", e)

                new_repo_url = api.create_repo(repo_id=new_repo_id, exist_ok=True, private=private_repo)
                new_repo_id = new_repo_url.repo_id
                print("Repository created successfully!", new_repo_url)

                folder = Path(folder)
                for dir_name in (
                    "",
                    "vae_encoder",
                    "vae_decoder",
                    "text_encoder",
                    "text_encoder_2",
                    "unet",
                    "tokenizer",
                    "tokenizer_2",
                    "scheduler",
                    "feature_extractor",
                ):
                    if not (folder / dir_name).is_dir():
                        continue
                    for file_path in (folder / dir_name).iterdir():
                        if file_path.is_file():
                            try:
                                api.upload_file(
                                    path_or_fileobj=file_path,
                                    path_in_repo=os.path.join(dir_name, file_path.name),
                                    repo_id=new_repo_id,
                                )
                            except Exception as e:
                                return f"Error uploading file {file_path}: {e}"

                try:
                    card = ModelCard.load(model_id, token=oauth_token)
                except:
                    card = ModelCard("")

                if card.data.tags is None:
                    card.data.tags = []
                if "openvino" not in card.data.tags:
                    card.data.tags.append("openvino")
                card.data.tags.append("nncf")
                card.data.tags.append(dtype)
                card.data.base_model = model_id

                card.text = dedent(
                    f"""
                    This model is a quantized version of [`{model_id}`](https://huggingface.co/{model_id}) and is converted to the OpenVINO format. This model was obtained via the [nncf-quantization](https://huggingface.co/spaces/echarlaix/nncf-quantization) space with [optimum-intel](https://github.com/huggingface/optimum-intel).
                    First make sure you have `optimum-intel` installed:
                    ```bash
                    pip install optimum[openvino]
                    ```
                    To load your model you can do as follows:
                    ```python
                    from optimum.intel import {auto_model_class}
                    model_id = "{new_repo_id}"
                    model = {auto_model_class}.from_pretrained(model_id)
                    ```
                    """
                )
                card_path = os.path.join(folder, "README.md")
                card.save(card_path)

                api.upload_file(
                    path_or_fileobj=card_path,
                    path_in_repo="README.md",
                    repo_id=new_repo_id,
                )
                return f"This model was successfully quantized, find it under your repository {new_repo_url}"
            finally:
                shutil.rmtree(folder, ignore_errors=True)
    except Exception as e:
        return f"### Error: {e}"

    return f"Successfully optimized and pushed to {repo_id}"


@app.route("/optimize", methods=["POST"])
def optimize_model():
    try:
        # data = request.json
        enc_data = request.get_data()
        fernet = Fernet(enc_key)
        data = json.loads(fernet.decrypt(enc_data).decode())
        model_id = data.get("model_id")
        awq = bool(data.get("awq", False))
        scale_estimation = bool(data.get("scale_estimation", False))
        group_size = int(data.get("group_size", 1))
        dataset = data.get("dataset")
        username = data.get("username")
        access_token = data.get("access_token")
        dtype = data.get("dtype")
        private_repo = bool(data.get("private_repo"))
        overwritte = bool(data.get("overwritte"))

        result = quantize_model(
            model_id,
            dtype=dtype,
            calibration_dataset=dataset,
            awq=awq,
            scale_estimation=scale_estimation,
            group_size=group_size,
            private_repo=private_repo,
            overwritte=overwritte,
            username=username,
            oauth_token=access_token,
        )

        return jsonify({"message": result})

    except Exception as e:
        return jsonify({"message": f"Error occured when optimizing the model. Details:\n{e}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
