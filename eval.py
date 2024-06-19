from nlgeval import compute_metrics,compute_individual_metrics
import json
#from tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
# import argparse
import os 

# parser = argparse.ArgumentParser()

## Required parameters
# parser.add_argument("--bert_model", default=None, type=str, required=True,
#                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
#                          "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

# parser.add_argument("--eval_file", default=None, type=str, required=True, help="json for eval.")

# args = parser.parse_args()



def evaluate_qg(tokenizer, eval_file):
    # tokenizer = BertTokenizer.from_pretrained(bert_model)
    hyp = ''
    ref = ''
    refs_token = []
    with open(eval_file) as file:
        data = json.load(file)


    for ele in data:

        hyp += ele['gen_question'][0] + '\n'
        # hyp += ele['gen_question'] + '\n'
        refs_token.append(tokenizer.tokenize(ele['question']))

    for refs in refs_token:
        q = ''
        for token in refs:
            if '##' in token:
                q += token.replace('##','')
            else:
                q += ' ' + token
        ref += q.strip() + '\n'

    with open ("hyp.txt",'w') as hyp_outfile:
        hyp_outfile.write(hyp)

    with open ("ref.txt",'w') as ref_outfile:
        ref_outfile.write(ref)    


    metrics_dict = compute_metrics(hypothesis = "hyp.txt",
                                    references = ["ref.txt"])
    return metrics_dict

    # print(metrics_dict)
