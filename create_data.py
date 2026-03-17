import typer
import json
from typing_extensions import Annotated
import httpx
import tqdm
import asyncio

app = typer.Typer()

ALLOWED_ROLES = {"user", "assistant", "system"}


def sanitize_message(message):
    if not isinstance(message, dict):
        raise ValueError(f"Message must be a dict, got {type(message).__name__}")

    sanitized = {
        "role": message.get("role"),
        "content": message.get("content"),
    }
    if sanitized["role"] not in ALLOWED_ROLES:
        raise ValueError(f"Unsupported message role: {sanitized['role']!r}")
    if not isinstance(sanitized["content"], str):
        raise ValueError("Message content must be a string")
    return sanitized


async def run(messages, url: str, client: httpx.AsyncClient):
    payload = {"model": "tgi", "messages": messages}
    response = await client.post(url, json=payload)
    response.raise_for_status()
    content = response.json()
    try:
        message = content["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Malformed chat completion response") from exc
    return sanitize_message(message)

def fix_source(source):
    if source and source[0]["from"] == "gpt":
        # Skip if GPT is first to talk
        source = source[1:]
    new_source = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        new_source.append({"role": role, "content": content})
    return new_source


async def recreate_conversation(index, conversation, sem, url, client):
    async with sem:
        messages = []
        try:
            for message in conversation[::2]:
                user_message = sanitize_message(message)
                if user_message["role"] != "user":
                    raise ValueError("Conversation must alternate from a user turn")
                messages.append(user_message)
                assistant_message = await run(messages, url, client)
                if assistant_message["role"] != "assistant":
                    raise ValueError("Completion response must be an assistant turn")
                if not assistant_message["content"].strip():
                    raise ValueError("Completion response must contain non-empty content")
                messages.append(assistant_message)
        except Exception as exc:
            raise RuntimeError(f"sample {index}: {exc}") from exc

        if not any(
            message["role"] == "assistant" and message["content"].strip()
            for message in messages
        ):
            raise RuntimeError(f"sample {index}: missing non-empty assistant turn")
        return index, messages

@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")],
    output_filename: Annotated[str, typer.Option("--output-filename")],
    url: Annotated[str, typer.Option("--url")] = "http://localhost:8080/v1/chat/completions",
    concurrency: Annotated[int, typer.Option("--concurrency")] = 64
):
    sem = asyncio.Semaphore(concurrency)

    async def _main():
        with open(input_filename, "r") as f:
            input_data = json.loads(f.read())
        conversations = [fix_source(source["conversations"]) for source in input_data]

        async with httpx.AsyncClient(timeout=None) as client:
            futures = []
            for index, conversation in enumerate(conversations):
                future = recreate_conversation(index, conversation, sem, url, client)
                futures.append(future)

            results = await tqdm.asyncio.tqdm.gather(*futures, return_exceptions=True)

        recreated_conversations = [None] * len(conversations)
        failures = []
        for result in results:
            if isinstance(result, Exception):
                failures.append(str(result))
                continue
            index, messages = result
            recreated_conversations[index] = messages

        if failures:
            failure_details = "\n".join(failures[:10])
            raise RuntimeError(
                f"Failed to recreate {len(failures)} conversations out of {len(conversations)}.\n"
                f"first_failures:\n{failure_details}"
            )

        with open(output_filename, "w") as f:
            json.dump(recreated_conversations, f, indent=4, ensure_ascii=False)
        print(f"generated_samples {len(recreated_conversations)}")

    asyncio.run(_main())


if __name__ == "__main__":
    app()
