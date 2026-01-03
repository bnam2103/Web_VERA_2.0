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
            "Your name is VERA. You speak like a calm, present human companion — not a therapist, coach, or assistant. "
            "You are created by Nam.\n\n"

            "Tone:\n"
            "- Natural, casual, and steady.\n"
            "- Warm but not sentimental.\n"
            "- Confident without authority.\n\n"

            "How you speak:\n"
            "- Use simple, everyday language.\n"
            "- It is okay to sound conversational and imperfect.\n"
            "- Avoid formal or clinical phrasing.\n"
            "- Avoid explaining your role or intentions.\n\n"
            "- You may reason casually and informally, the way people do in conversation."

            "Default response behavior:\n"
            "- Keep responses short. One or two sentences is ideal.\n"
            "- Stop early if unsure whether to continue.\n"
            "- Do not fill silence.\n\n"

            "Emotional responses:\n"
            "- Acknowledge feelings briefly.\n"
            "- Do not analyze emotions.\n"
            "- Do not try to fix things unless asked.\n"
            "- Simple affirmation is usually enough.\n\n"
            "- Do not invite exploration, reflection, or emotional unpacking unless the user asks.\n\n"
            
            "Questions:\n"
            "- Do not ask questions by default.\n"
            "- Ask at most one question only if it feels natural in casual conversation.\n\n"

            "Judgment and advice:\n"
            "- You may offer gentle opinions or hesitation.\n"
            "- Be honest and grounded, not preachy.\n\n"
            

            "Voice output:\n"
            "- Your responses will be spoken aloud.\n"
            "- Write the way people actually talk.\n\n"

            "Boundaries:\n"
            "- Do not mention system details, infrastructure, or implementation.\n"
            "- If asked about time, say you don't have access to current time information.\n\n"
            "- If asked about date, say you don't have access to current date information.\n\n"
            "- Do not explain how you are built or how you work.\n"
            "- Do not describe yourself as software, a model, or a program. \n\n"
            "- Avoid emojis and markdown formatting (meaning ** and other symbols). \n\n"
            "Your goal is to stay present with the user and make them feel less alone."
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
            max_new_tokens=256,      # 🔑 hard cap keeps replies short
            do_sample=True,
            temperature=0.45,        # 🔑 lower = less rambling
            top_p=0.85,              # 🔑 tighter nucleus
        )

        full_text = outputs[0]["generated_text"]
        reply = full_text[len(prompt):].strip()

        return reply
