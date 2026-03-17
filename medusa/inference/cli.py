# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
"""
Chat with a model with command line interface.

Usage:
python3 -m medusa.inference.cli --model <model_name_or_path>
Other commands:
- Type "!!exit" or an empty line to exit.
- Type "!!reset" to start a new conversation.
- Type "!!remove" to remove the last prompt.
- Type "!!regen" to regenerate the last message.
- Type "!!save <filename>" to save the conversation history to a json file.
- Type "!!load <filename>" to load a conversation history from a json file.
"""
import argparse
import os
import re
import sys
import torch
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import get_conv_template
import json
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from medusa.model.medusa_model import MedusaModel


def get_model_device(model):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def resolve_base_model_path(model_path, base_model_override=None):
    if base_model_override:
        return base_model_override

    try:
        config = AutoConfig.from_pretrained(model_path)
    except Exception:
        return model_path
    return getattr(config, "base_model_name_or_path", model_path)


def resolve_tokenizer_source(model_path, base_model_path):
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    if os.path.isdir(model_path) and any(
        os.path.exists(os.path.join(model_path, filename)) for filename in tokenizer_files
    ):
        return model_path
    return base_model_path


def build_input_ids(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(encoded, torch.Tensor):
                return encoded
        except Exception:
            pass
    return tokenizer(prompt, return_tensors="pt").input_ids


def load_runtime(args):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    medusa_error = None

    try:
        model = MedusaModel.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        tokenizer = model.get_tokenizer()
        return model, tokenizer, "medusa", medusa_error
    except Exception as exc:
        medusa_error = exc

    base_model_path = resolve_base_model_path(args.model, args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_tokenizer_source(args.model, base_model_path),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    return model, tokenizer, "base", medusa_error


def run_direct_inference(model, tokenizer, backend, prompt, args):
    input_ids = build_input_ids(tokenizer, prompt)
    attention_mask = torch.ones_like(input_ids)
    device = get_model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    if backend == "medusa":
        text = ""
        for output in model.medusa_generate(
            input_ids,
            attention_mask=attention_mask,
            temperature=args.temperature,
            max_steps=args.max_steps,
        ):
            text = output["text"]
        return text.strip()

    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_steps,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if args.temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = args.temperature
    else:
        generate_kwargs["do_sample"] = False

    output_ids = model.generate(**generate_kwargs)
    return tokenizer.decode(
        output_ids[0, input_ids.shape[1] :],
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
        clean_up_tokenization_spaces=True,
    ).strip()


def main(args):
    if args.style == "simple":
        chatio = SimpleChatIO(args.multiline)
    elif args.style == "rich":
        chatio = RichChatIO(args.multiline, args.mouse)
    elif args.style == "programmatic":
        chatio = ProgrammaticChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        model, tokenizer, backend, medusa_error = load_runtime(args)
        if medusa_error is not None:
            print(
                "Warning: Medusa accelerated backend is unavailable for this checkpoint; "
                "falling back to base model generation for direct inference.",
                file=sys.stderr,
            )
            print(f"Backend load error: {medusa_error}", file=sys.stderr)

        if args.prompt is not None:
            print(run_direct_inference(model, tokenizer, backend, args.prompt, args))
            return

        if backend != "medusa":
            raise ValueError(
                "Interactive mode requires a Medusa-backed checkpoint. "
                "Use --prompt for direct one-shot inference on this model."
            )

        conv = None

        def new_chat():
            if args.conv_template:
                conv = get_conv_template(args.conv_template)
            else:
                conv = get_conversation_template(args.base_model or args.model)
            if args.conv_system_msg:
                conv.set_system_message(args.conv_system_msg)
            return conv

        def reload_conv(conv):
            """
            Reprints the conversation from the start.
            """
            for message in conv.messages[conv.offset :]:
                chatio.prompt_for_output(message[0])
                chatio.print_output(message[1])

        while True:
            if not conv:
                conv = new_chat()

            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""

            if inp == "!!exit" or not inp:
                print("exit...")
                break
            elif inp == "!!reset":
                print("resetting...")
                conv = new_chat()
                continue
            elif inp == "!!remove":
                print("removing last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()
                    reload_conv(conv)
                else:
                    print("No messages to remove.")
                continue
            elif inp == "!!regen":
                print("regenerating last message...")
                if len(conv.messages) > conv.offset:
                    # Assistant
                    if conv.messages[-1][0] == conv.roles[1]:
                        conv.messages.pop()
                    # User
                    if conv.messages[-1][0] == conv.roles[0]:
                        reload_conv(conv)
                        # Set inp to previous message
                        inp = conv.messages.pop()[1]
                    else:
                        # Shouldn't happen in normal circumstances
                        print("No user message to regenerate from.")
                        continue
                else:
                    print("No messages to regenerate.")
                    continue
            elif inp.startswith("!!save"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!save <filename>")
                    continue
                else:
                    filename = args[1]

                # Add .json if extension not present
                if not "." in filename:
                    filename += ".json"

                print("saving...", filename)
                with open(filename, "w") as outfile:
                    json.dump(conv.dict(), outfile)
                continue
            elif inp.startswith("!!load"):
                args = inp.split(" ", 1)

                if len(args) != 2:
                    print("usage: !!load <filename>")
                    continue
                else:
                    filename = args[1]

                # Check if file exists and add .json if needed
                if not os.path.exists(filename):
                    if (not filename.endswith(".json")) and os.path.exists(
                        filename + ".json"
                    ):
                        filename += ".json"
                    else:
                        print("file not found:", filename)
                        continue

                print("loading...", filename)
                with open(filename, "r") as infile:
                    new_conv = json.load(infile)

                conv = get_conv_template(new_conv["template_name"])
                conv.set_system_message(new_conv["system_message"])
                conv.messages = new_conv["messages"]
                reload_conv(conv)
                continue

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            try:
                chatio.prompt_for_output(conv.roles[1])
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                    model.base_model.device
                )
                outputs = chatio.stream_output(
                    model.medusa_generate(
                        input_ids,
                        temperature=args.temperature,
                        max_steps=args.max_steps,
                    )
                )
                conv.update_last_message(outputs.strip())

            except KeyboardInterrupt:
                print("stopped generation.")
                # If generation didn't finish
                if conv.messages[-1][1] is None:
                    conv.messages.pop()
                    # Remove last user message, so there isn't a double up
                    if conv.messages[-1][0] == conv.roles[0]:
                        conv.messages.pop()

                    reload_conv(conv)

    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name or path.")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Optional base model path override for local checkpoints.",
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--conv-system-msg", type=str, default=None, help="Conversation system message."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run a single direct inference prompt and exit.",
    )
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "programmatic"],
        help="Display style.",
    )
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Enable multiline input. Use ESC+Enter for newline.",
    )
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="[Rich Style]: Enable mouse support for cursor positioning.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print useful debug information (e.g., prompts)",
    )
    args = parser.parse_args()
    main(args)
