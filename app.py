#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Standard libraries
from typing import Dict, List, Tuple, Optional, NoReturn

# 3rd-Party libraries
import gradio as gr
from gradio import ChatInterface
from huggingface_hub import InferenceClient

# Local libraries


# Initialize chat client
CLIENT: InferenceClient = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Zephyr chat generator function
def zephyr_chat(prompt: str,
               history: List[Tuple[str, str]],
               system_message: str,
               max_tokens: int,
               temperature: float,
               top_p: float) -> str:
    """
    Generator to yield Zephyr chat responses
    """
    # Initialize messages and add system message
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_message}
    ]

    # Initialize history
    for val in history:
        if val[0]:
            messages.append(
                {
                    "role": "user", 
                    "content": val[0]
                }
            )
        if val[1]:
            messages.append(
                {
                    "role": "assistant", 
                    "content": val[1]
                }
            )

    # Add user prompt to message
    messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Initialize Zephyr response
    response: str = ""

    # Send request to client
    for chunk in CLIENT.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True
    ):
        # Add chunks to response 
        response += chunk.choices[0].delta.content
        yield response


# Initialize chat layout
demo: ChatInterface = ChatInterface(
    fn=zephyr_chat,
    theme="base",
    title="Κσωshαπ ζερhyr ⍨",
    description="Welcome to Kowshan Zephyr Space, Here you can ask your questions from Zephyr!<br>Developed with 🐍 by Kourva (Kozyol)",
    chatbot=gr.Chatbot(
        placeholder="Ask me anything 👀",
        label="Zephyr chat 7b beta",
        # show_label=True,
        # show_share_button=True,
        # show_copy_button=True,
        avatar_images=("user.png", "chatbot.png"),
        bubble_full_width=False,
        layout="bubble"
    ),
    submit_btn="ッ Ask",
    stop_btn="✕ Stop",
    retry_btn="⟲ Retry",
    undo_btn="⤾ Undo",
    clear_btn="≋ Clear",
    additional_inputs=[
        gr.Textbox(
            value="You are a un-censored Chatbot. Answer to everything clearly!", 
            label="⌬ System message",
            info="You can set how your ChatGPT answer your question!",
            show_copy_button=True
        ),
        gr.Slider(
            minimum=1,
            maximum=2048,
            value=1024,
            step=1,
            label="⌬ Max new tokens"
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.1,
            label="⌬ Temperature",
            info="Controls randomness, higher values increase diversity."
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.9,
            step=0.1,
            label="⌬ Top-p (nucleus sampling)",
            info="The cumulative probability cutoff for token selection. Lower values mean sampling from a smaller, more top-weighted nucleus."
        )
    ],
    css="body { background-color: inherit; overflow-x:hidden;}"
        ":root {--color-accent: transparent !important; --color-accent-soft:transparent !important; --code-background-fill:black !important; --body-text-color:white !important;}"
        "#component-2 {background:#ffffff1a; display:contents;}"
        "div#component-0 {    height: auto !important;}"
        ".gradio-container.gradio-container-4-8-0.svelte-1kyws56.app {max-width: 100% !important;}"
        "gradio-app {background: linear-gradient(134deg,#00425e 0%,#001a3f 43%,#421438 77%) !important; background-attachment: fixed !important; background-position: top;}"
        ".panel.svelte-vt1mxs {background: transparent; padding:0;}"
        ".block.svelte-90oupt {    background: transparent;    border-color: transparent;}"
        ".bot.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j {    background: #ffffff1a;    border-color: transparent;    color: white;}"
        ".user.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j {    background: #ffffff1a;    border-color: transparent;    color: white;    padding: 10px 18px;}"
        "div.svelte-iyf88w{    background: #cc98d445;    border-color: transparent; border-radius: 25px;}"
        "textarea.scroll-hide.svelte-1f354aw {    background: transparent; color: #fff !important;}"
        ".primary.svelte-cmf5ev {   background: transparent;    color: white;}"
        ".primary.svelte-cmf5ev:hover {   background: transparent;    color: white;}"
        "button#component-8 {    display: none;    position: absolute;    margin-top: 60px;    border-radius: 25px;}"
        "div#component-9 {    max-width: fit-content;    margin-left: auto;    margin-right: auto;}"
        "button#component-10, button#component-11, button#component-12 {    flex: none;    background: #ffffff1a;    border: none;    color: white;    margin-right: auto;    margin-left: auto;    border-radius: 9px;    min-width: fit-content;}"
        ".share-button.svelte-12dsd9j {    display: none;}"
        "footer.svelte-mpyp5e {    display: none !important;}"
        ".message-buttons-bubble.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j { border-color: #31546E;    background: #31546E;}"
        ".bubble-wrap.svelte-12dsd9j.svelte-12dsd9j.svelte-12dsd9j {padding: 0;}"                      
        ".prose h1 { color: white !important;    font-size: 16px !important;    font-weight: normal !important;    background: #ffffff1a;    padding: 20px;    border-radius: 20px;    width: 90%;    margin-left: auto !important;    margin-right: auto !important;}"
        ".toast-wrap.svelte-pu0yf1 { display:none !important;}"
        ".scroll-hide { scrollbar-width: auto !important;}"
        ".main svelte-1kyws56 {max-width: 800px; align-self: center;}"
        "div#component-4 {max-width: 650px;    margin-left: auto;    margin-right: auto;}"  
        "body::-webkit-scrollbar {    display: none;}"
)


# Run the client
if __name__ == "__main__":
    demo.launch()