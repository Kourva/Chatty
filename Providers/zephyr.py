""""""
from huggingface_hub import InferenceClient

# Zephyr chat client
def zephyr_chat(messages, kwargs):
    CLIENT: InferenceClient = InferenceClient(
        "HuggingFaceH4/zephyr-7b-beta"
    )
    # Send request to client
    for chunk in CLIENT.chat_completion(
        messages,
        **kwargs
    ):
        # Add chunks to response 
        response += chunk.choices[0].delta.content
    return response
