import torch
import json
import os
from transformers import DistilBertTokenizer, DistilBertModel

MAX_LEN = 512

class DistilBERTClass(torch.nn.Module):

    def __init__(self):

        super(DistilBERTClass,self).__init__()

        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')

        self.pre_classifier = torch.nn.Linear(768,768)

        self.dropout = torch.nn.Dropout(0.3)

        self.classifier = torch.nn.Linear(768,4)

    def forward(self,input_ids, attention_mask):

        output_1 = self.l1(input_ids=input_ids,attention_mask=attention_mask)

        hidden_state = output_1[0]

        pooler = hidden_state[:,0]

        pooler = self.pre_classifier(pooler)

        pooler = torch.nn.ReLU()(pooler)

        pooler = self.dropout(pooler)

        output = self.classifier(pooler)

        return output


def model_fn(model_dir):

    print("Loading model from :", model_dir)

    model = DistilBERTClass() # was DistilBertClass
    model_state_dict = torch.load(os.path.join(model_dir, 'pytorch_distilbert_news.bin'),map_location = torch.device('cpu'))
    model.load_state_dict(model_state_dict)

    return model

def input_fn(request_body,request_content_type):

    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        sentence = input_data['inputs']
        return sentence
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    inputs = tokenizer(input_data, return_tensors="pt").to(device)

    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    model.eval()

    with torch.no_grad():
        outputs = model(ids,mask)

    probabilities = torch.softmax(outputs, dim = 1).cpu().numpy()

    class_names = ["Business", "Science", "Entertainment", "Health"]
    predicted_class = probabilities.argmax(axis=1)[0] # 3
    predicted_label = class_names[predicted_class] # Health


    return {'predicted_label': predicted_label, 'probabilities':probabilities.tolist()}



def output_fn(prediction, accept):

    if accept == 'application/json':
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
