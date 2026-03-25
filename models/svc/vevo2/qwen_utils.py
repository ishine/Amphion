import re
import torch


def format_chat_prompt(messages, add_assistant_token):
    """
    Convert the messages list into the Qwen chat template format.

    Args:
        messages: A list of messages containing role and content.
        add_assistant_token: Whether to add assistant token at the end.

    Returns:
        str: The formatted prompt string.
    """
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Add start and end tags for all messages except the last assistant message
        if msg != messages[-1] or role != "assistant":
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        else:
            # For the last assistant message, only add the start tag and content
            prompt += f"<|im_start|>{role}\n{content}"

    # If the last message is not from assistant and add_assistant_token is True
    if messages[-1]["role"] != "assistant" and add_assistant_token:
        prompt += f"<|im_start|>assistant\n"

    return prompt


def gen_chat_prompt(text, add_assistant_token, follow_prosody_instruction):
    """
    Args:
        text (str): The text to be spoken.
        add_assistant_token (bool): Whether to add assistant token at the end. For pre-training, False. For Inference, True.
        follow_prosody_instruction (bool): Whether to follow the prosody instruction. When prosody_ids is not None, True. Otherwise, False.
    """
    if follow_prosody_instruction:
        synthesis_instruction = "User will provide you with a text. Please first generate a good prosodic instruction, then vocalize the text based on it."
    else:
        synthesis_instruction = "User will provide you with a text. Please vocalize it with natural expression."

    template = [
        {
            "role": "system",
            "content": synthesis_instruction,
        },
        {
            "role": "user",
            "content": text,
        },
    ]
    return format_chat_prompt(template, add_assistant_token)


def gen_chat_response(prosody_ids, content_style_ids, is_full_response=True):
    """
    Args:
        prosody_ids (list): The prosody ids of the text.
        content_style_ids (list): The content style ids of the text.
    """
    if prosody_ids is not None:
        prosody_text = "".join(["<|prosody_{}|>".format(int(i)) for i in prosody_ids])
        prosody_text = "<|prosody_start|>" + prosody_text + "<|prosody_end|>"
    else:
        prosody_text = ""

    if content_style_ids is not None:
        content_style_text = "".join(
            ["<|content_style_{}|>".format(int(i)) for i in content_style_ids]
        )
    else:
        content_style_text = ""

    if is_full_response:
        return (
            prosody_text
            + "<|content_style_start|>"
            + content_style_text
            + "<|content_style_end|>"
            + "<|im_end|>"
        )
    else:
        return prosody_text + "<|content_style_start|>" + content_style_text


def extract_content_style_ids(text):
    """
    Extract the content_style IDs from the text

    Args:
        text (str): A string containing content_style tags

    Returns:
        torch.Tensor: [T]
    """
    # Use regex to match all <|content_style_数字|> patterns
    pattern = r"<\|content_style_(\d+)\|>"
    # Find all matches and extract the numeric part
    matches = re.findall(pattern, text)
    # Convert string numbers to integers
    return torch.tensor([int(match) for match in matches], dtype=torch.long)
