from huggingface_hub import InferenceClient

# Zephyr chat client
def zephyr_chat(messages, kwargs):
    CLIENT: InferenceClient = InferenceClient(
        "HuggingFaceH4/zephyr-7b-beta"
    )

    # Initialize Zephyr response
    response: str = ""

    # Send request to client
    for chunk in CLIENT.chat_completion(
        messages,
        **kwargs
    ):
        # Add chunks to response 
        response += chunk.choices[0].delta.content
        yield response
