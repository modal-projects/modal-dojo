---
order: 0
---

# Choosing a base model

The first step to training is choosing a base model. The Training Gym supports the [top](https://gym.modal.dev/#supported-models) open-source LLMs and VLMs. 

## How do I choose?

A good first step is understanding your workload [type](https://modal.com/llm-almanac/workloads) and modality (e.g., vision or audio). This will help narrow down the model size you can afford to run based on your throughput and latency needs. 

Next, you'll want the strongest base model possible. In our experience, generally the Qwen series dominates the smaller (i.e., up to tens of billions of parameters) model landscape, while for larger models, the top choice changes frequently.

An even better solution is benchmarking the capabilities of each model for your use case: stay tuned for our solution to this!

## Once chosen

Then, it’s as easy as:

```python
from modal_dojo import Qwen3_6_27B

model = Qwen3_6_27B()
```

When you instantiate the object, weight downloading, response parsing, and architecture details are handled for you behind the scenes.

## To support a new model

We keep our [list of supported models](https://gym.modal.dev/#supported-models) comprehensive and up-to-date. However, if you find that we don’t support a certain model, you can easily extend [HFModelConfiguration](https://gym.modal.dev/reference/hfmodelconfiguration). See the [Qwen3.8 file](https://github.com/modal-projects/training-gym/blob/main/modal_dojo/common/models/qwen3_8_27b.py) for a good example. Our provided [agent skill](https://github.com/modal-projects/training-gym/blob/main/skills/model-support/SKILL.md) is very helpful for this.  
