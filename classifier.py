import os
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import DistilBertTokenizerFast, DistilBertModel
from flask import Flask

app = Flask(__name__)

class BERTClassifier(nn.Module):
    """Defines the classifier powering the webapp. Can be swapped for any PyTorch compatible multi-class multi-label
    classification model """

    def __init__(self, config=None, dr=0.5, freeze_bert=True):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
        # Freeze bert layers
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.drop = nn.Dropout(p=dr)
        self.fc = nn.Linear(768, 434)  # TODO Make this not a magic number, classify() should be agnostic to this number
        app.logger.info("Model Init")

    def forward(self, _input, attention_mask):

        output = self.bert(_input, attention_mask, output_hidden_states=False, output_attentions=False)
        output = output['last_hidden_state'][:, 0]
        text_fea = self.drop(output)
        text_fea = nn.ReLU()(text_fea)
        text_fea = self.fc(text_fea)
        text_out = torch.squeeze(text_fea, 1)

        return text_out


def clean_text(text):
    stripped_tokens = [token for token in word_tokenize(text) if token.isalpha() or token == '>']
    return ' '.join(stripped_tokens)


# TODO handle empty/garbage input
def classify(text):
    """Given Photoshop tutorial text as input, returns a nested dict of label probabilities for each sentence in
    input """
    # TODO Wonder if there's a better way of doing this so the model persists across fucn calls. Will see once I get
    #  Heroku running
    sentences = sent_tokenize(text)
    clean_sentences = list(map(clean_text, sentences))
    word_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # tokenize
    _input = word_tokenizer(clean_sentences, padding=True, return_tensors='pt', return_attention_mask=True)

    # pass tokenized text thru model
    model = BERTClassifier(dr=.3, freeze_bert=True)
    state_dict = torch.load(os.path.join('model', 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model_state_dict'])

    model.eval()
    with torch.no_grad():
        model_output = model(_input['input_ids'], _input['attention_mask'])

    #  We start with: A list of command names, a list of sentences, a matrix with
    #  each row corresponding to a sentence and each column corresponding to a label's probability of being
    #  represented in the sentence The list of command names is parallel to the columns of the matrix

    # We want to end with a nested dict with sentences as keys and dicts of label : probability pairs as values
    labels = model_output.topk(3)

    label_indices = labels[0].tolist()
    probabilities = labels[1].tolist()

    app.logger.debug(f'Label tensor shape: {labels[1].shape}')

    command_names = [f'placeholder{i}' for i in range(434)]  # TODO
    output = dict()
    for i, row in enumerate(probabilities):  # TODO vectorize this if possible
        sent = sentences[i]
        output[sent] = {command_names[idx]: label_indices[i][j] for j, idx in enumerate(row)}

    return output
