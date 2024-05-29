import json


def preprocess_intents_json(intents_file):
    with open(intents_file, "r") as f:
        data = json.load(f)
    
    preprocessed_data = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            preprocessed_data.append(f"User: {pattern}\n")
            for response in intent["responses"]:
                preprocessed_data.append(f"Assistant: {response}\n")
    
    return "".join(preprocessed_data)


def save_preprocessed_data(preprocessed_data, output_file):
    with open(output_file, "w") as f:
        f.write(preprocessed_data)


intents_file = "C:/multim/intents.json"
output_file = "C:/multim//mental_health_data.txt"


preprocessed_data = preprocess_intents_json(intents_file)
save_preprocessed_data(preprocessed_data, output_file)