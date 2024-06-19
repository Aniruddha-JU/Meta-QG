import os, torch

import numpy as np
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from copy import deepcopy

import os
import sys
import gzip
import json_lines
import logging
import collections
import json
import pickle
import multiprocessing
import argparse
#import tokenization
import math
from tqdm import tqdm

import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer

def get_logger(log_name):
    # Write log
    logging.basicConfig(filename=log_name,
                        level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', filemode='a')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))  # For print out the result on console
    logger.info('')
    # logger.info("#################################### New Start #####################################")
    return logger


logger = get_logger('qa.log')

def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

class SquadExample(object):
    
#   """A single training/test example for simple sequence classification.

#      For examples without an answer, the start and end position are -1.
#   """
    def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_context,
               orig_answer_text=None,
               start_position=None,
               end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_context = orig_context               #New
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        
        s = ""
        '''
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.orig_answer_text:
            s += ", orig_answer: [%s]" % (" ".join(self.orig_answer_text))
        '''
        return s

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               orig_context,
               orig_answer,
               orig_query):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.orig_context = orig_context
    self.orig_answer = orig_answer
    self.orig_query = orig_query
    self.representation = None

class QAProcessor(object):
    def read_squad_examples(self, language, mode, is_training=True):
        input_file = mode +"_"+ language +  ".json"
        with open(input_file, encoding='utf-8') as f:
            input_data = json.load(f)['data']
                  
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                  return True
            return False
        
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
#                     if is_training:
#                         if (len(qa["answers"]) != 1) and (not is_impossible):
# #                             raise ValueError("For training, each question should have exactly 1 answer.")
#                            print(qa["answers"])   
                    if not is_impossible:
                      answer = qa["answers"][0]
                      orig_answer_text = answer["text"]
                      answer_offset = answer["answer_start"]
                      answer_length = len(orig_answer_text)
                      start_position = char_to_word_offset[answer_offset]
                      end_position = char_to_word_offset[answer_offset + answer_length -
                                                     1]
                  # Only add answers where the text can be exactly recovered from the
                  # document. If this CAN'T happen it's likely due to weird Unicode
                  # stuff so we will just skip the example.
                  #
                  # Note that this means for training mode, every example is NOT
                  # guaranteed to be preserved.
                      actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                      cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
#                       if actual_text.find(cleaned_answer_text) == -1:
#                           tf.logging.warning("Could not find answer: '%s' vs. '%s'",actual_text, cleaned_answer_text)
#                     continue
                    else:
                      start_position = -1
                      end_position = -1
                      orig_answer_text = ""
                example = SquadExample(
                  qas_id=qas_id,
                  question_text=question_text,
                  doc_tokens=doc_tokens,
                  orig_context=paragraph_text,
                  orig_answer_text=orig_answer_text,
                  start_position=start_position,
                  end_position=end_position)
                examples.append(example)
                      
        return examples

def compute_represenation(sents, bert_model, logger, device="cuda", reprer=None):
    if reprer is None:
        model = BertModel.from_pretrained(bert_model).to(device)
    else:
        model = reprer.model
    model.eval()
    batch_size = 100
    for i in range(0, len(sents), batch_size):
        items = sents[i : min(len(sents), i + batch_size)]
        with torch.no_grad():
            input_ids = torch.tensor([item.input_ids for item in items], dtype=torch.long).to(device)
            segment_ids = torch.tensor([item.segment_ids for item in items], dtype=torch.long).to(device)
            input_mask = torch.tensor([item.input_mask for item in items], dtype=torch.long).to(device)
            all_encoder_layers, _ = model(input_ids, segment_ids, input_mask)  # batch_size x seq_len x target_size
        layer_output = all_encoder_layers[-1].detach().cpu().numpy() # batch_size x seq_len x target_size
        for j, item in enumerate(items):
            item.representation = layer_output[j][0]
        # item.representation = layer_output
        if i % (10 * batch_size) == 0:
            logger.info('  Compute sentence representation. To {}...'.format(i))
    logger.info('  Finish.')


class Reprer():
    def __init__(self, bert_model, device="cuda"):
        self.device = device
        self.model = BertModel.from_pretrained(bert_model).to(device)

class Corpus(object):
    def __init__(self, bert_model, max_seq_length, doc_stride, max_query_length, logger, language, mode, is_training=True, support_size=-1,
                 base_features=None, mask_rate=-1.0, compute_repr=False, shuffle=True, k_shot_prop=-1.0, reprer=None):
        self.processor = QAProcessor()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)

        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.language = language
        self.mode = mode
        self.logger = logger
        self.mask_rate = mask_rate

        # get original feature list (do not consider [MASK] scheme)
        self.original_features = self.build_original_features(language, mode, max_seq_length, doc_stride, max_query_length, is_training=is_training, skip_no_ans = True )

        print(len(self.examples))
        #print(len(self.original_features))
        #print(self.examples[0])
        #print(self.original_features[0].tokens)
        #print(self.original_features[0].start_position)
        if k_shot_prop > 0:
            n_tmp = len(self.original_features)
            #kept_idxs = np.random.permutation(n_tmp).tolist()[: int(n_tmp * k_shot_prop)] if k_shot_prop < 1.0 else np.random.permutation(n_tmp).tolist()[: int(k_shot_prop)]
            #logger.info('  The kept {}-shot-prop idxs are: {}'.format(k_shot_prop, ', '.join([str(i) for i in kept_idxs])))
            #self.original_features = [self.original_features[i] for i in kept_idxs]

            kept_idxs = [y for y in range(0,int(k_shot_prop))]
            logger.info('  The kept {}-shot-prop idxs are: {}'.format(k_shot_prop, ', '.join([str(i) for i in kept_idxs])))
            self.original_features = [self.original_features[i] for i in kept_idxs]
        # compute representations for original features (in-place operation)
        if compute_repr:
            compute_represenation(self.original_features, bert_model, logger, reprer=reprer)

        # build query set
        if mask_rate < 0: # NO [MASK] scheme
            self.query_features = self.original_features
        else:
            self.query_features = self.build_query_features_with_mask(mask_rate) # (masked)

        # build support set
        assert isinstance(support_size, int)
        if support_size > 0: # build support set (NOT masked)
            if base_features is None:
                self.support_features = self.build_support_features_(self.original_features, support_size=support_size)
            else:
                self.support_features = self.build_support_features_(base_features, support_size=support_size)

        self.n_total = len(self.query_features)
        self.batch_start_idx = 0
        self.batch_idxs = np.random.permutation(self.n_total) if shuffle else np.array([i for i in range(self.n_total)]) # for batch sampling in training


    def reset_batch_info(self, shuffle=False):
        self.batch_start_idx = 0
        self.batch_idxs = np.random.permutation(self.n_total) if shuffle else np.array([i for i in range(self.n_total)]) # for batch sampling in training
    
    def build_original_features(self, language, mode, max_seq_length, doc_stride, max_query_length, is_training=False,
                                 skip_no_ans=False, verbose=False):
        """
        max_seq_length: "The maximum total input sequence length after WordPiece tokenization. Sequences 
                        longer than this will be truncated, and sequences shorter than this will be padded."
        doc_stride: "When splitting up a long document into chunks, how much stride to take between chunks."
        max_query_length: "The maximum number of tokens for the question. Questions longer than this will be truncated to this length."
        """
        self.logger.info("Build original features for [{}-{}]...".format(language, mode))

        # examples: a list of sentences. each item is a tuple of a list of words and a list of tags > (['words'], ['tags'])
        self.examples = self.processor.read_squad_examples(language, mode, is_training = is_training)  # 'en', 'train'

        unique_id = 1000000000

        features = []
        for (example_index, example) in enumerate(self.examples):
            query_tokens = self.tokenizer.tokenize(example.question_text)
            
            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]
            ans_tokens = self.tokenizer.tokenize(example.orig_answer_text)
            #if example_index > 40: break
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None

            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, self.tokenizer,
                example.orig_answer_text)
            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(ans_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)
            #if len(doc_spans)>1: continue
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                #if doc_span_index > 0: continue
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                '''
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)
                '''
                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)
                for token in ans_tokens:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)
                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                # convert to numpy array
                input_ids = np.asarray(input_ids, dtype=np.int32)
                input_mask = np.asarray(input_mask, dtype=np.uint8)
                segment_ids = np.asarray(segment_ids, dtype=np.uint8)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_position = None
                end_position = None
                if is_training:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        if skip_no_ans:
                            continue
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

#                 if is_training:
#                     tokens = None
#                     token_to_orig_map = None
#                     token_is_max_context = None

                if example_index < 20 and verbose:
                    logger.info("*** Example ***")
                    logger.info("unique_id: %s" % (unique_id))
                    logger.info("example_index: %s" % (example_index))
                    logger.info("doc_span_index: %s" % (doc_span_index))
                    logger.info("tokens: %s" % " ".join(tokens))
                    logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("answer: %s" % (example.orig_answer_text))
                    logger.info("context: %s" % (example.orig_context))
                    logger.info("query: %s" % (example.question_text))

                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        orig_answer=example.orig_answer_text,
                        orig_context=example.orig_context,
                        orig_query=example.question_text))
                unique_id += 1

        return features

    def build_query_features_with_mask(self, mask_rate):
        self.logger.info("Build query features with MASK for [{}-{}]...".format(self.language, self.mode))

        assert mask_rate > 0
        features = deepcopy(self.original_features)
        mask_id = self.tokenizer.vocab['[MASK]']

        n_BIs = 0
        n_masked = 0
        for item in features:
            for i, label_id in enumerate(item.label_id):
                if label_id == 0: # [PAD] token
                    break
                label = self.label_list[label_id-1]
                if len(label) > 1 and label[1] == '-': # -: both B-XXX and I-XXX have a '-'
                    n_BIs += 1
                    if np.random.random() < mask_rate:
                        item.input_ids[i] = mask_id
                        n_masked += 1

        self.logger.info('  Masked {}/{} tokens in total.'.format(n_masked, n_BIs))

        return features

    def _prepare_data(self, examples, fn):

        def output_item(tokens, labels, res_list, fw):
            item = ' '.join(['[CLS]'] + tokens + ['[SEP]']) + '\t|\t' + ' '.join(['[CLS]'] + labels + ['[SEP]'])
            res_list.append(item)
            fw.write(item + '\n')

        fw = open(fn, 'w', encoding='utf-8')
        res = []

        for (ex_index, example) in enumerate(examples):
            textList = example.text_a
            labelList = example.label
            tokens = []
            labels = []
            for i, word in enumerate(textList):
                token = self.tokenizer.tokenize(word)
                label = [labelList[i]] + ['X'] * (len(token) - 1)
                if len(token) != len(label):
                    assert False
                tokens.extend(token)
                labels.extend(label)

            if len(tokens) >= self.max_seq_length - 1:
                tokens_ = tokens[0:(self.max_seq_length - 2)]
                labels_ = labels[0:(self.max_seq_length - 2)]
                output_item(tokens_, labels_, res, fw)

                curr_idx = self.max_seq_length - 2
                while len(tokens) >= curr_idx + self.max_seq_length // 2 - 2:
                    tokens_ = tokens[curr_idx - self.max_seq_length // 2: curr_idx + self.max_seq_length // 2 - 2]
                    labels_ = labels[curr_idx - self.max_seq_length // 2: curr_idx + self.max_seq_length // 2 - 2]
                    output_item(tokens_, labels_, res, fw)
                    curr_idx += self.max_seq_length // 2 - 2

                tokens_ = tokens[curr_idx - self.max_seq_length // 2:]
                labels_ = labels[curr_idx - self.max_seq_length // 2:]
                output_item(tokens_, labels_, res, fw)

            else:
                output_item(tokens, labels, res, fw)

        return res

    def _load_data(self, fn):
        res = []
        with open(fn, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                res.append(line)
        return res


    def get_support_ids(self, base_features, support_size=2):
        self.logger.info("Getting support feature ids for [{}-{}]...".format(self.language, self.mode))
        target_features = self.query_features

        target_reprs = np.stack([item.representation for item in target_features])
        base_reprs = np.stack([item.representation for item in base_features])  # sample_num x feature_dim

        # compute pairwise cosine distance
        dis = np.matmul(target_reprs, base_reprs.T)  # target_num x base_num

        base_norm = np.linalg.norm(base_reprs, axis=1)  # base_num
        base_norm = np.stack([base_norm] * len(target_features), axis=0)  # target_num x base_num

        dis = dis / base_norm  # target_num x base_num
        relevance = np.argsort(dis, axis=1)

        support_id_set = []
        for i, item in enumerate(target_features):
            chosen_ids = relevance[i][-1 * (support_size + 1): -1]
#             if i <= 9:
#                 self.logger.info('  Support set info: {}: {}'.format(i, ', '.join([str(id) for id in chosen_ids])))
            support_id_set.extend(chosen_ids)

        support_id_set = set(support_id_set)

#         self.logger.info('  size of support ids: {}'.format(len(support_id_set)))
        return list(support_id_set)

    def reset_query_features(self, feature_ids, shuffle=True):
        self.logger.info("Reset query features of [{}-{}]...".format(self.language, self.mode))
        self.query_features = [self.original_features[i] for i in feature_ids]

        self.n_total = len(self.query_features)
        self.reset_batch_info(shuffle=shuffle)

        self.logger.info('  size of current query features: {}'.format(self.n_total))

    def build_support_features_(self, base_features, support_size=2):
        self.logger.info("Build support features for [{}-{}]...".format(self.language, self.mode))
        target_features = self.query_features

        target_reprs = np.stack([item.representation for item in target_features])
        base_reprs = np.stack([item.representation for item in base_features]) # sample_num x feature_dim

        # compute pairwise cosine distance
        dis = np.matmul(target_reprs, base_reprs.T) # target_num x base_num

        base_norm = np.linalg.norm(base_reprs, axis=1) # base_num
        base_norm = np.stack([base_norm] * len(target_features), axis=0) # target_num x base_num

        dis = dis / base_norm # target_num x base_num
        relevance = np.argsort(dis, axis=1)

        support_set = []
        for i, item in enumerate(target_features):
            chosen_ids = relevance[i][-1 * (support_size + 1) : -1]
#             self.logger.info('  Support set info: {}: {}'.format(i, ', '.join([str(id) for id in chosen_ids])))
            support = [base_features[id] for id in chosen_ids]
            support_set.append(support)
            if i<20:
                print("***********************New example: Here are the tokens of test set ***********************\n")
                print(item.tokens)
                print("\n*****************Here are the support features tokens *******************************************\n")
                for spt in support:
                    print(spt.tokens)
                    print("\n\n")

        return support_set


    def get_batch_meta(self, batch_size, device="cuda", shuffle=True):
        if self.batch_start_idx + batch_size > self.n_total:
            self.reset_batch_info(shuffle=shuffle)
            if self.mask_rate >= 0:
                self.query_features = self.build_query_features_with_mask(mask_rate=self.mask_rate)
        
        features = self.query_features
        examples = self.examples

        query_batch = []
        support_batch = []
        start_id = self.batch_start_idx
        print(start_id)
        for i in range(start_id, start_id + batch_size):
            idx = self.batch_idxs[i]
            query_i = self.query_features[idx]
            query_item = {
            'orig_answer': ([query_i.orig_answer]),
            'orig_context': ([query_i.orig_context]),
            'orig_query': ([query_i.orig_query])
            # 'unique_id': torch.tensor([query_i.unique_id], dtype=torch.long).to(device),
            # 'example_index': torch.tensor([query_i.example_index], dtype=torch.long).to(device),
            # 'doc_span_index': torch.tensor([query_i.doc_span_index], dtype=torch.long).to(device),
            # 'input_ids': torch.tensor([query_i.input_ids], dtype=torch.long).to(device),
            # 'input_mask': torch.tensor([query_i.input_mask], dtype=torch.long).to(device),
            # 'segment_ids': torch.tensor([query_i.segment_ids], dtype=torch.long).to(device),
            # 'start_position': torch.tensor([query_i.start_position], dtype=torch.long).to(device),
            # 'end_position': torch.tensor([query_i.end_position], dtype=torch.long).to(device),
            #  'tokens': ([query_i.tokens])#,
            # 'flag_ids': torch.tensor([f.flag for f in batch_features], dtype=torch.long).to(device)
        }
            query_batch.append(query_item)

            support_i = self.support_features[idx]
            support_item = {
            'orig_answer': ([f.orig_answer for f in support_i]),
            'orig_context': ([f.orig_context for f in support_i]),
            'orig_query': ([f.orig_query for f in support_i])
            # 'unique_id': torch.tensor([f.unique_id for f in support_i], dtype=torch.long).to(device),
            # 'example_index': torch.tensor([f.example_index for f in support_i], dtype=torch.long).to(device),
            # 'doc_span_index': torch.tensor([f.doc_span_index for f in support_i], dtype=torch.long).to(device),
            # 'input_ids': torch.tensor([f.input_ids for f in support_i], dtype=torch.long).to(device),
            # 'input_mask': torch.tensor([f.input_mask for f in support_i], dtype=torch.long).to(device),
            # 'segment_ids': torch.tensor([f.segment_ids for f in support_i], dtype=torch.long).to(device),
            # 'start_position': torch.tensor([f.start_position for f in support_i], dtype=torch.long).to(device),
            # 'end_position': torch.tensor([f.end_position for f in support_i], dtype=torch.long).to(device),
            #  'tokens': ([f.tokens for f in support_i])#,#,
            # 'flag_ids': torch.tensor([f.flag for f in batch_features], dtype=torch.long).to(device)
        }
            support_batch.append(support_item)

        self.batch_start_idx += batch_size

        return query_batch, support_batch, examples, features

    def get_batch_NOmeta(self, batch_size, device="cuda", shuffle=True):
        if self.batch_start_idx + batch_size >= self.n_total:
            self.reset_batch_info(shuffle=shuffle)
            if self.mask_rate >= 0:
                self.query_features = self.build_query_features_with_mask(mask_rate=self.mask_rate)

        idxs = self.batch_idxs[self.batch_start_idx : self.batch_start_idx + batch_size]
        batch_features = [self.query_features[idx] for idx in idxs]
        # batch_features = self.query_features[self.batch_start_idx : self.batch_start_idx + batch_size]

        batch = {
            'orig_answer': ([f.orig_answer for f in batch_features]),
            'orig_context': ([f.orig_context for f in batch_features]),
            'orig_query': ([f.orig_query for f in batch_features])
            # 'unique_id': torch.tensor([f.unique_id for f in batch_features], dtype=torch.long).to(device),
            # 'example_index': torch.tensor([f.example_index for f in batch_features], dtype=torch.long).to(device),
            # 'doc_span_index': torch.tensor([f.doc_span_index for f in batch_features], dtype=torch.long).to(device),
            # 'input_ids': torch.tensor([f.input_ids for f in batch_features], dtype=torch.long).to(device),
            # 'input_mask': torch.tensor([f.input_mask for f in batch_features], dtype=torch.long).to(device),
            # 'segment_ids': torch.tensor([f.segment_ids for f in batch_features], dtype=torch.long).to(device),
            # 'start_position': torch.tensor([f.start_position for f in batch_features], dtype=torch.long).to(device),
            # 'end_position': torch.tensor([f.end_position for f in batch_features], dtype=torch.long).to(device)#,
            # 'flag_ids': torch.tensor([f.flag for f in batch_features], dtype=torch.long).to(device)
        }

        self.batch_start_idx += batch_size

        return batch

    def get_batches(self, is_training=True, batch_size=8, device="cuda", shuffle=False):
        batches = []
        bat_exam = []
        if shuffle:
            idxs = np.random.permutation(self.n_total)
            features = [self.query_features[i] for i in idxs]
            examples = [self.examples[i] for i in idxs]
        else:
            features = self.query_features
            examples = self.examples
        
        for i in range(0, self.n_total, batch_size):
            batch_features = features[i : min(self.n_total, i + batch_size)]
            batch = {
            'orig_answer': ([f.orig_answer for f in batch_features]),
            'orig_context': ([f.orig_context for f in batch_features]),
            'orig_query': ([f.orig_query for f in batch_features])
            # 'unique_id': torch.tensor([f.unique_id for f in batch_features], dtype=torch.long).to(device),
            # 
            # 'example_index': torch.tensor([f.example_index for f in batch_features], dtype=torch.long).to(device),
            # 'doc_span_index': torch.tensor([f.doc_span_index for f in batch_features], dtype=torch.long).to(device),
            # 'input_ids': torch.tensor([f.input_ids for f in batch_features], dtype=torch.long).to(device),
            # 'input_mask': torch.tensor([f.input_mask for f in batch_features], dtype=torch.long).to(device),
            # 'segment_ids': torch.tensor([f.segment_ids for f in batch_features], dtype=torch.long).to(device),
            # 'start_position': torch.tensor([f.start_position for f in batch_features], dtype=torch.long).to(device),
            # 'end_position': torch.tensor([f.end_position for f in batch_features], dtype=torch.long).to(device)#,
            # 'flag_ids': torch.tensor([f.flag for f in batch_features], dtype=torch.long).to(device)
            }

            batches.append(batch)

        return batches, examples, features


###########################################################################################
'''
def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      verbose_logging=False, version_2_with_negative=False, null_score_diff_threshold=0):
    """Write final predictions to the json file and log-odds of null if needed."""
    if verbose_logging:
        logger.info("Writing predictions to: %s" % (output_prediction_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
#         print("****************************************************")
#         print(example)
        features = example_index_to_features[example_index]
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0,
                             _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)
#         print(output)
#         print("******************************************")
        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json
    
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    return json.dumps(all_predictions)

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text

'''
