from huggingface_hub import InferenceClient

# Zephyr chat client
def mistral_chat(messages, kwargs):
    CLIENT: InferenceClient = InferenceClient(
        "huggingface/EleutherAI/gpt-j-6B"
    )

    # Initialize Zephyr response
    response: str = ""

    # Send request to client
    for chunk in CLIENT.text_generation(
        messages,
        **kwargs
    ):
        # Add chunks to response 
        response += chunk.token.text
        yield response
