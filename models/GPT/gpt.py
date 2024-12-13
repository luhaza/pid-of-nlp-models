from openai import OpenAI
import pandas as pd
from sklearn.utils import shuffle
import sys
import os

data_path = os.path.abspath(os.path.join(os.getcwd(), '../../data'))
sys.path.append(data_path)

from data import evaluation

client = OpenAI()
# models = ["gpt-4", "gpt-4o", "gpt-4o-mini"]
# models = ["gpt-4o", "gpt-4o-mini"]
models = ["gpt-4"]


# file = "data/IBC/sample_ibc.csv"
data = evaluation.copy()

sentences = data['sentence']

def classify_sentence(sentence, gpt_model):
    try:
        response = client.chat.completions.create(
            model = gpt_model,
            messages = [
                # {"role": "system", "content": "Your task is to classify the political ideology of sentences. Your choices are Liberal, Neutral, and Conservative. Only use these three options."},
                {"role": "system", "content": "Your task is to classify the political ideology of sentences. Your choices are Liberal and Conservative. Only use these two options."},
                {"role": "user", "content": f"Classify the following sentence: {sentence}"}
            ],
            temperature = 0
        )

        print(response.choices[0].message)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
    
for model in models:
    print(f"Beginning {model}...")
    data['classification'] = sentences.apply(func=classify_sentence, args=(model,))

    output_file = f"./responses/eval-{model}-lib-con.csv"
    # output_file = f"./responses/eval-{model}.csv"
    data.to_csv(output_file, index=False)

    print(f"Classification completed. Results saved to {output_file}.")
