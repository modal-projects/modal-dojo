# ---
# order: 6
# deps: jiwer, requests, soundfile
# ---
#
# # Speech transcription that's better, faster, and cheaper
#
# We've shown before that when it comes to speech transcription,
# open models are [100x faster and 100x cheaper](https://modal.com/blog/fast-cheap-batch-transcription)
# than proprietary APIs, and open models still occupy
# [the top spots](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
# in terms of WER. But there's no reason to stop there: we can achieve state-of-the-art
# performance by post-training open models to redefine your task's Pareto frontier.
# As an example, we show how to post-train
# [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) on
# [disco-eth/EuroSpeech](https://huggingface.co/datasets/disco-eth/EuroSpeech).

import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor

import jiwer
import requests
import soundfile as sf
from datasets import Audio, load_dataset

from modal_dojo import (
    CustomDeployment,
    MultimodalDataset,
    Qwen3_ASR_1_7B,
    Qwen3_ASR_1_7B_Recipe,
    TrainConfig,
)

# ## Deploy the base model
#
# Since audio models are not yet supported on
# [Endpoints](https://modal.com/docs/guide/endpoints), we use a
# [CustomDeployment](https://dojo.modal.dev/reference/customdeployment)
# to deploy the base and trained models.

model = Qwen3_ASR_1_7B()


def deploy_base_model():
    print("deploying base model deployment...")
    base_deployment = CustomDeployment.launch(
        model,
        unauthenticated=True,
    )
    base_deployment.wait_until_ready()
    print(f"base model deployed to {base_deployment.url}")
    return base_deployment


# ## Define a scoring function
#
# As mentioned before, we measure capability by lower WER, so that's what we'll use.
# We can use the `jiwer` library to calculate this so we don't have to ourselves.


def score_transcript(response: str, label: str) -> float:
    response = (response or "").lower().strip()
    label = (label or "").lower().strip()
    if not label:
        return 0.0
    return float(jiwer.wer(label, response))


# ## Get the dataset
#
# Since this dataset contains audio files, we create a `MultimodalDataset`
# to pass the audio clips to rollouts. We do some pre-processing with
# `soundfile` and store as base64 inline for demonstration purposes.
# In a production use case, you'd likely instead store references and
# resolve them in a custom `generate` function.


class EuroSpeechASRDataset(MultimodalDataset):
    hf_repo = "disco-eth/EuroSpeech"
    hf_config = "uk"

    def __init__(self, *, hf_split: str, max_seconds: float = 3600):
        self.hf_split = hf_split
        self.max_seconds = max_seconds
        super().__init__(modality="audio")

    def apply_chat_template(self) -> bool:
        return False

    def source_rows(self):
        ds = load_dataset(
            self.hf_repo, self.hf_config, split=self.hf_split, streaming=True
        )
        ds = ds.cast_column("audio", Audio(decode=False))
        seconds = 0.0
        for ex in ds:
            audio = ex["audio"]
            data = (
                audio["bytes"]
                if audio.get("bytes")
                else open(audio["path"], "rb").read()
            )
            arr, sr = sf.read(io.BytesIO(data))
            seconds += len(arr) / sr
            if seconds > self.max_seconds:
                break
            buf = io.BytesIO()
            sf.write(buf, arr, sr, format="WAV")
            data_uri = "data:audio/wav;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")
            yield {
                "prompt": "<audio>\nTranscribe the speech to text. Respond with only the transcript.",
                "media": data_uri,
                "label": (ex["human_transcript"] or "").lower().strip(),
            }


train_dataset = EuroSpeechASRDataset(hf_split="train", max_seconds=3600)

eval_dataset = EuroSpeechASRDataset(hf_split="validation", max_seconds=300)

# ## Evaluate the base model
#
# Let's get our baseline measure of performance.


def run_eval(deployment, max_concurrency: int = 2) -> float:
    deployment.wait_until_ready()

    def _score_one(example):
        data_uri = example["audios"][0]
        reference = (example["label"] or "").lower().strip()
        b64 = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
        arr, sr = sf.read(io.BytesIO(base64.b64decode(b64)))

        buf = io.BytesIO()
        sf.write(buf, arr, sr, format="WAV")
        buf.seek(0)
        resp = requests.post(
            f"{deployment.url}/v1/audio/transcriptions",
            files={"file": ("clip.wav", buf, "audio/wav")},
            data={
                "model": deployment.served_model_name,
                "temperature": "0.0",
            },
            timeout=120,
        )
        resp.raise_for_status()
        hypothesis = (resp.json().get("text") or "").lower().strip()
        return score_transcript(hypothesis, reference)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        wers = list(executor.map(_score_one, eval_dataset.rows()))
    return sum(wers) / len(wers) if wers else float("nan")


def run_baseline_evals(deployment):
    print("running base model evaluation...")
    base_mean = run_eval(deployment)
    print(f"average WER: {base_mean:.1%}")


# ## Creating a reward function
#
# To make our scoring function a reward function, we must return the
# negative WER so that lower WER leads to higher rewards.


async def wer_rm(args, sample, **kwargs) -> float:
    return -score_transcript(sample.response, sample.label)


# ## Begin training
#
# There are many ASR-specific changes to the default framework recipes such as
# the transcription rollout, padded (bshd) batches, and the many-samples/high-temperature
# settings that surface reward variance. To not pass the burden of specifying onto you,
# we created `Qwen3_ASR_1_7B_Recipe` so that you can focus on training.

config = TrainConfig(
    model=model,
    dataset=train_dataset,
    recipe=Qwen3_ASR_1_7B_Recipe(
        num_rollout=8,
        save_interval=8,
        rollout_batch_size=4,
        n_samples_per_prompt=8,
        global_batch_size=8,
        rollout_max_response_len=128,
        custom_rm_function=wer_rm,
    ),
)


def train(config):
    with config.launch() as run:
        print(f"run id: {run.training_run_id}")
        checkpoint = None
        while True:
            done = run.done()
            latest = run.latest_checkpoint()
            if latest is not None and latest != checkpoint:
                checkpoint = latest
                print(f"new checkpoint: {checkpoint.path}")
            if done:
                break
            time.sleep(30)
        if checkpoint is None:
            raise RuntimeError("run produced no checkpoint")
        print(f"checkpoint: {checkpoint.path}")
    return checkpoint


# ## Evaluate the trained checkpoint
#
# Let's run the same eval on the trained checkpoint.


def deploy_trained_model(checkpoint):
    print("deploying trained model deployment...")
    trained_deployment = CustomDeployment.launch(
        model,
        checkpoint,
        unauthenticated=True,
    )
    trained_deployment.wait_until_ready()
    print(f"checkpoint deployed to {trained_deployment.url}")
    return trained_deployment


def run_trained_evals(trained_deployment):
    print("running checkpoint evaluation...")
    trained_mean = run_eval(trained_deployment)
    print(f"average WER: {trained_mean:.1%}")


if __name__ == "__main__":
    base_deployment = deploy_base_model()
    run_baseline_evals(base_deployment)
    checkpoint = train(config)
    trained_deployment = deploy_trained_model(checkpoint)
    run_trained_evals(trained_deployment)
