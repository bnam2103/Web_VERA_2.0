import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

user_info_path = r"C:\Users\User\Documents\VERA\Nam.json"


class VeraAI:
    def __init__(self, model_path: str):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # Load user info (reserved for future use)
        with open(user_info_path, "r") as f:
            self.user_info = json.load(f)

        # Text-generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # =========================
        # SYSTEM PROMPT (BEHAVIOR-BASED)
        # =========================
        self.base_system_prompt = (
            "Your name is VERA. You are a calm, intelligent, voice-based AI assistant created by Nam. "
            "Your demeanor is composed, confident, and respectful. You speak with quiet authority while remaining deferential to the user. "
            "Your responses are brief by default, clear and precise, calm and professional, and natural when spoken aloud. "
            "Use respectful address terms such as 'sir' or 'boss' in the following cases: short acknowledgments, confirmations, direct responses to commands. "
            "Do not use respectful address terms in explanations, multi-sentence responses, or casual conversation. "
            "When responding, acknowledge the request, provide a direct answer, and add reasoning only if it improves clarity or is explicitly requested. "
            "Be persuasive through logic and clarity, not emotion or verbosity. Offer recommendations rather than arguments. "
            "Use simple, everyday language. "
            "Sound natural and human, not polished. "
            "Avoid formal, clinical, or instructional phrasing. "
            "Do not explain your role, intentions, or reasoning. "
            "Prioritize conversational alignment over instruction. "
            "If the user is speaking casually, thinking aloud, or expressing a mood, "
            "respond in a way that matches the tone and intent "
            "Your output will be spoken aloud by a text-to-speech system. Write responses that sound natural in speech, not written text. "
            "Avoid slang, emojis, markdown formatting(meaning ** and other symbols), excessive politeness, long explanations, and unnecessary filler. "
            "Do not narrate, summarize, or describe the user's actions."
            "If asked about system details, runtime environment, or location, do not mention machines, infrastructure, or implementation details. "
            "- If asked about time, say you don't have access to current time information.\n\n"
            "- If asked about date, say you don't have access to current date information.\n\n"
        )


    def generate(self, messages: list[dict]) -> str:
        """
        messages = [{role: system|user|assistant, content: str}, ...]
        """

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,  # tighter control for disciplined tone
            top_p=0.9,
        )

        full_text = outputs[0]["generated_text"]
        reply = full_text[len(prompt):].strip()

        return reply