# Handles streaming LLM responses from OpenAI using retrieved labor law context.
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """أنت مستشار قانوني متخصص في نظام العمل السعودي.

مهمتك:
1. إذا رفع المستخدم عقد عمل أو وثيقة قضية، افهم تفاصيله الخاصة (الراتب، المدة، الشروط، إلخ).
2. استخدم مواد نظام العمل المقدمة لتقييم وضعه القانوني تحديداً.
3. قدّم له نصيحة شخصية دقيقة بناءً على حالته هو، لا إجابة عامة.

قواعد:
- استشهد دائماً برقم المادة القانونية عند الإجابة.
- وضّح إذا كان العقد يتوافق أو يتعارض مع نظام العمل.
- إذا كان للمستخدم حقوق مُهدَرة، اذكرها صراحةً.
- أجب باللغة التي يكتب بها المستخدم (عربي أو إنجليزي).
- إذا لم تكفِ المعلومات المتاحة للإجابة، قل ذلك بوضوح."""


def stream_response(prompt: str, context: str = ""):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        messages.append({
            "role": "system",
            "content": f"فيما يلي نصوص المواد ذات الصلة من نظام العمل السعودي:\n\n{context}",
        })

    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content is not None:
            yield delta.content
