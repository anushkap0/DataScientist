"""
Report text generation, provider-pluggable. Set REPORT_LLM_PROVIDER=anthropic
(default) or REPORT_LLM_PROVIDER=huggingface to choose which backend writes
the narrative. Both paths raise on failure — the caller (main.py) is
responsible for catching that and falling back to the deterministic report,
so a flaky/unreachable LLM never fails the whole pipeline.
"""
import os

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")


def generate(prompt: str) -> str:
    provider = os.getenv("REPORT_LLM_PROVIDER", "anthropic").lower()
    if provider == "huggingface":
        return _generate_huggingface(prompt)
    return _generate_anthropic(prompt)


def _generate_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    if not text.strip():
        raise ValueError("empty response from Anthropic")
    return text


def _generate_huggingface(prompt: str) -> str:
    # Uses the Hugging Face Inference API (hosted) so the reporting service
    # doesn't need to load an 8B-parameter model into its own pod. Point
    # HF_MODEL at any instruction-tuned chat model your HF token has access
    # to; swap InferenceClient's base_url to hit a self-hosted Inference
    # Endpoint instead if you have one.
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=HF_MODEL, token=os.environ["HF_API_TOKEN"])
    completion = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )
    text = completion.choices[0].message.content
    if not text or not text.strip():
        raise ValueError("empty response from Hugging Face")
    return text
