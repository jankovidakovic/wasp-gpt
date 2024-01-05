from langchain_community.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-4-1106-preview")

while True:
    prompt = input("[GPT-4-1106-Preview] > ")
    for chunk in chat.stream(prompt):
        print(chunk.content, end="", flush=True)
    print()
