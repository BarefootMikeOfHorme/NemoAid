from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

print("Connecting to NVIDIA Cloud NIM...", flush=True)

completion = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain cloud GPU acceleration in one sentence."}
    ],
    temperature=0.5,
    max_tokens=256,
    stream=False
)

print("\n--- CLOUD RESPONSE ---")
print(completion.choices[0].message.content)
