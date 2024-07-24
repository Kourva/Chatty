def transcribe(audio):
    lang = "en"
    model = engines[lang]
    text = model.stt_file(audio)[0]
    return text