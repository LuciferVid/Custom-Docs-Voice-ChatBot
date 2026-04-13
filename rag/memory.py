from datetime import datetime

class ConversationMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, role: str, content: str, metadata: dict = None):
        """
        Appends a message to the conversation history with optional metadata.
        """
        self.messages.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })

    def get_history(self, last_n: int = 10) -> str:
        """
        Returns the last n messages as a formatted string.
        """
        history = ""
        for msg in self.messages[-last_n:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history += f"{role}: {msg['content']}\n"
        return history

    def clear(self):
        """
        Empties the conversation history.
        """
        self.messages = []

    def to_list(self) -> list[dict]:
        """
        Returns the message list.
        """
        return self.messages
