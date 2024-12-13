import tiktoken
import os
import sys

data_path = os.path.abspath(os.path.join(os.getcwd(), '../../data'))
sys.path.append(data_path)

from data import evaluation

data = evaluation.copy()
tokenizer = tiktoken.encoding_for_model("gpt-4")

price = 0.0

system = "Your task is to classify the political ideology of sentences. Your choices are Liberal and Conservative. Only use these two options."

for example in data["sentence"]:

    request = system + "Classify the following: " + example
    response = "The sentence does not contain any explicit political ideology. However, if forced to choose between the two options, it could be associated with Conservative due to its focus on crime and violence, which are often highlighted in conservative narratives about law and order"

    request_tokens = tokenizer.encode(request)
    response_tokens = tokenizer.encode(response)

    input_tokens = len(request_tokens)
    output_tokens = len(response_tokens)

    # costs per 1 million tokens
    cost_per_1M_input_tokens = 30 
    cost_per_1M_output_tokens = 60

    input_cost = (input_tokens / 10**6) * cost_per_1M_input_tokens
    output_cost = (output_tokens / 10**6) * cost_per_1M_output_tokens
    total_cost = input_cost + output_cost
    price += total_cost


print(f"Cost: ${price:.5f}")
