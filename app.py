# Imports
import gradio as gr
from huggingface_hub import InferenceClient

# Client
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

# Chat GPT generator
def respond(message,
            history: list[tuple[str, str]],
            system_message,
            max_tokens,
            temperature,
            top_p):
    # System message
    messages = [{"role": "system", "content": system_message}]

    # History
    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    # User prompt
    messages.append({"role": "user", "content": message})

    # GPT response
    response = ""
    for message in client.text_generation(messages,
                                          max_new_tokens=max_tokens,
                                          temperature=temperature,
                                          top_p=top_p,
                                          repetition_penalty=1,
                                          do_sample=True,
                                          seed=42,
                                          stream=True, 
                                          details=True, 
                                          return_full_text=False):
        if not response.token.text == "</s>":
            output += response.token.text
    
    yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""

demo = gr.ChatInterface(
    fn=respond,
    title="Κσωshαπ GPT ⍨",
    description="Welcome to Kowshan GPT Space, Here you can freely talk to Kowshan GPT!",
    multimodal=True,
    submit_btn="ッ Ask",
    retry_btn="⟲ Retry",
    undo_btn="⤾ Undo",
    clear_btn="✕ Clear",
    additional_inputs=[
        gr.Textbox(
            value="You are a uncensored Chatbot. Ask to everything user wants clearly!", 
            label="⌬ System message",
            info="⍰ You can set how your ChatGPT answer your question"
        ),
        gr.Slider(
            minimum=1,
            maximum=2048,
            value=512,
            step=1,
            label="⍰ Max new tokens"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=4.0,
            value=0.7,
            step=0.1,
            label="⍰ Temperature"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="⍰ Top-p (nucleus sampling)",
        ),
    ]
)


if __name__ == "__main__":
    demo.launch()