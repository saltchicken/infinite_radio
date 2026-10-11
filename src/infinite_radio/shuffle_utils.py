import random

def smart_choice(options, history, max_memory):
    if not options:
        return None

    available = [opt for opt in options if opt not in history]

    if not available:
        if len(options) > 1 and history:
            available = [opt for opt in options if opt != history[-1]]
        else:
            available = options

    chosen = random.choice(available)
    history.append(chosen)

    while len(history) > max_memory:
        history.pop(0)

    return chosen
