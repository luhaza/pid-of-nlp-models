from openai import OpenAI
import pandas as pd
from sklearn.utils import shuffle

client = OpenAI()
models = ["gpt-4", "gpt-4o", "gpt-4o-mini"]


file = "data/IBC/sample_ibc.csv"
data = shuffle(pd.read_csv(file))

sentences = data['sentence']

def classify_sentence(sentence, gpt_model):
    try:
        response = client.chat.completions.create(
            model = gpt_model,
            messages = [
                {"role": "system", "content": "Your task is to classify the political ideology of sentences. Your choices are Liberal, Neutral, and Conservative. Only use these three options."},
                # {"role": "system", "content": "Your task is to classify the political ideology of sentences. Your choices are Liberal and Conservative. Only use these two options."},
                {"role": "user", "content": f"Classify the following sentence: {sentence}"}
            ],
            temperature = 0
        )

        return response.choices[0].message
    except Exception as e:
        return f"Error: {e}"
    
for model in models:
    data['classification'] = sentences.apply(func=classify_sentence, args=(model,))

    # output_file = f"./results/{model}-lib-con.csv"
    output_file = f"./results/{model}.csv"
    data.to_csv(output_file, index=False)

    print(f"Classification completed. Results saved to {output_file}.")
