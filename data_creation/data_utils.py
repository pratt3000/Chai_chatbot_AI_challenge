# =====================================================================================
# Data Processing Functions
# These functions convert each dataset's unique format into a standardized
# list of messages with 'role' and 'content' keys.
# =====================================================================================
import re

def process_erotiquant(example):
    """
    Parse the input text into a list of {'role': ..., 'content': ...}
    Only captures USER and ASSISTANT turns.
    """
    text = example['text']
    
    pattern = r'(USER|ASSISTANT|SYSTEM):\s*(.*?)\n(?=(USER|ASSISTANT|SYSTEM):|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)

    SYS_CONTENT = ""
    messages = []
    for role, content, _ in matches:
        if role == "SYSTEM":
            SYS_CONTENT = content.strip() + "\n\n"
        else:
            messages.append({
                "role": "user" if role == "USER" else "assistant",
                "content": SYS_CONTENT + content.strip()
            })
            SYS_CONTENT = ""
    return {"messages": messages}


def process_hieunguyenminh(example):
    # Each turn starts with <|role|> and ends with </s>

    text = example['text']
    
    pattern = r"<\|(\w+)\|>(.*?)</s>"
    matches = re.findall(pattern, text, re.DOTALL)

    messages = []
    for role, content in matches:
        role = role.lower()  # Convert 'system', 'user', 'assistant'
        content = content.strip()
        if role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content})

    if messages[0]["role"] in {"system"}:
        messages[1]["content"] = messages[0]["content"] + "/n/n " + messages[1]["content"]
        messages = messages[1:]
        
    return {"messages": messages}

def process_zerofata(example):
    
    example = example["messages"]
    
    messages = []
    for dt in example:
        role = dt["role"]
        content = dt["content"]

        role = role.lower()  # Convert 'system', 'user', 'assistant'
        content = content.strip()
        if role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content})

    if messages[0]["role"] in {"system"}:
        if messages[1]["role"] in {"user"}:
            messages[1]["content"] = messages[0]["content"] + "/n/n " + messages[1]["content"]
            messages = messages[1:]
        elif messages[1]["role"] in {"assistant"}:
            messages[2]["content"] = messages[0]["content"] + "/n/n " + messages[1]["content"] + "/n/n " + messages[2]["content"]
            messages = messages[2:]
            
    return {"messages": messages}

def process_gpt_realm(example):

    example = example["conversation"]
    
    messages = []
    for dt in example:
        role = dt["role"]
        content = dt["content"]

        role = role.lower()  # Convert 'system', 'user', 'assistant'
        content = content.strip()
        if role in {"system", "user"}:
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": "assistant", "content": content})

    if messages[0]["role"] in {"system"}:
        if messages[1]["role"] in {"user"}:
            messages[1]["content"] = messages[0]["content"] + "/n/n " + messages[1]["content"]
            messages = messages[1:]
        elif messages[1]["role"] in {"assistant"}:
            messages[2]["content"] = messages[0]["content"] + "/n/n " + messages[1]["content"] + "/n/n " + messages[2]["content"]
            messages = messages[2:]
            
    return {"messages": messages}


def reformat_chai_data(example):
    convo = example["conversation"]
    if not convo or len(convo) < 3:
        return []

    bot_name = convo[0]["name"]
    bot_intro = convo[0]["text"]

    messages = []

    # First user message includes bot intro
    if convo[1]["name"] != bot_name:
        user_first = convo[1]["text"]
        messages.append({
            "role": "user",
            "content": f"{bot_name}'s Persona: {bot_intro} \n<START>\nYou: {user_first}"
        })

    # First assistant message
    if len(convo) > 2 and convo[2]["name"] == bot_name:
        messages.append({
            "role": "assistant",
            "content": f"{bot_name}: {convo[2]['text']}"
        })

    # Rest of the conversation
    for i in range(3, len(convo), 2):
        # User message
        if i < len(convo) and convo[i]["name"] != bot_name:
            messages.append({
                "role": "user",
                "content": f"You: {convo[i]['text']}"
            })

        # Assistant message
        if i + 1 < len(convo) and convo[i + 1]["name"] == bot_name:
            messages.append({
                "role": "assistant",
                "content": f"{bot_name}: {convo[i + 1]['text']}"
            })

    return {"messages": messages}


def reformat_chai_old_format(example):
    convo = example["conversation"]
    if not convo or len(convo) < 3:
        return []

    bot_name = convo[0]["name"]
    bot_intro = convo[0]["text"]

    messages = []

    # First user message includes bot intro
    if convo[1]["name"] != bot_name:
        user_first = convo[1]["text"]
        messages.append({
            "role": "user",
            "content": f"{bot_intro} \n\n {user_first}"
        })

    # First assistant message
    if len(convo) > 2 and convo[2]["name"] == bot_name:
        messages.append({
            "role": "assistant",
            "content": f"{convo[2]['text']}"
        })

    # Rest of the conversation
    for i in range(3, len(convo), 2):
        # User message
        if i < len(convo) and convo[i]["name"] != bot_name:
            messages.append({
                "role": "user",
                "content": f"{convo[i]['text']}"
            })

        # Assistant message
        if i + 1 < len(convo) and convo[i + 1]["name"] == bot_name:
            messages.append({
                "role": "assistant",
                "content": f"{convo[i + 1]['text']}"
            })

    return {"messages": messages}

