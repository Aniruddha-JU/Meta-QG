from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import CONFIG_NAME, WEIGHTS_NAME, BertConfig #, BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer


from torch.utils.data import RandomSampler
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder

from torch import nn
from copy import deepcopy
from eval_ani import evaluate_qg

import collections
import json

import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import seq2seq_loader
# from preprocessorQA import write_predictions
# from mrqa_official_eval import evaluate, read_answers


import  torch, os, numpy, json, time

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

class LearnerQA(nn.Module):
    def __init__(self, bert_model,freeze_layer, hidden_size, logger, lr_meta, lr_inner,
                 warmup_prop_meta, warmup_prop_inner, max_meta_steps, max_query_length=48, max_seq_length=512, model_dir='', cache_dir='', device = "cuda", gpu_no=0):
        super(LearnerQA, self).__init__()
        self.lr_meta = lr_meta
        self.lr_inner = lr_inner
        self.warmup_prop_meta = warmup_prop_meta
        self.warmup_prop_inner = warmup_prop_inner
        self.max_meta_steps = max_meta_steps

        self.bert_model = bert_model
        '''
        Parameters for question generation 
        '''
        self.max_seq_length = max_seq_length
        self.max_pred = max_query_length
        self.label_smoothing = 0
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.mask_prob = 0.7
        self.from_scratch = None
        self.new_segment_ids = True
        self.new_pos_ids = None
        self.max_len_a = 0
        self.max_len_b = 0
        self.trunc_seg = ''
        self.always_truncate_tail = None
        self.num_workers = 0
        self.mask_source_words = None
        self.skipgram_prb = 0.0
        self.skipgram_size = 1
        self.mask_whole_word = None
        self.has_sentence_oracle = None
        self.max_position_embeddings = None
        # self.relax_projection = None
        self.ffn_type = 0
        self.num_qkv = 0
        self.seg_emb = None
        self.s2s_special_token = None
        self.s2s_add_segment = None
        self.s2s_share_segment = None
        self.pos_shift = None
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=None)
        self.cls_num_labels = 2
        self.type_vocab_size = 6
        self.num_sentlvl_labels = 0
        self.relax_projection = 0

         ## load model
        if model_dir != '':
            logger.info('********** Loading saved model **********')
            output_config_file = os.path.join(model_dir, CONFIG_NAME)
            output_model_file = os.path.join(model_dir, 'en_{}'.format(WEIGHTS_NAME))
            print(output_config_file)
            print(output_model_file)
            config = BertConfig(output_config_file)
            model_recover = torch.load(output_model_file, map_location='cpu')
            self.model =BertForPreTrainingLossMask.from_pretrained(self.bert_model, state_dict=model_recover, num_labels=self.cls_num_labels, num_rel=0, type_vocab_size=self.type_vocab_size, config_path=output_config_file, task_idx=3, num_sentlvl_labels=self.num_sentlvl_labels, max_position_embeddings=self.max_position_embeddings, label_smoothing=self.label_smoothing, fp32_embedding=None, relax_projection=self.relax_projection, new_pos_ids=self.new_pos_ids, ffn_type=self.ffn_type, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, num_qkv=self.num_qkv, seg_emb=self.seg_emb)
            # self.model.load_state_dict(torch.load(output_model_file, map_location="cuda:{}".format(gpu_no)))
        else:
            logger.info('********** Loading pre-trained model **********')
            cache_dir = cache_dir if cache_dir else str(PYTORCH_PRETRAINED_BERT_CACHE)
            model_file= os.path.join(bert_model, 'pytorch_model.bin')
            model_recover = torch.load(model_file, map_location='cpu')
            self.model =BertForPreTrainingLossMask.from_pretrained(self.bert_model, state_dict=model_recover, num_labels=self.cls_num_labels, num_rel=0, type_vocab_size=self.type_vocab_size, config_path=None, task_idx=3, num_sentlvl_labels=self.num_sentlvl_labels, max_position_embeddings=self.max_position_embeddings, label_smoothing=self.label_smoothing, fp32_embedding=None, relax_projection=self.relax_projection, new_pos_ids=self.new_pos_ids, ffn_type=self.ffn_type, hidden_dropout_prob=self.hidden_dropout_prob, attention_probs_dropout_prob=self.attention_probs_dropout_prob, num_qkv=self.num_qkv, seg_emb=self.seg_emb)
        self.model.to(device)

        ## layer freezing
        freeze_layer = 0
        if freeze_layer == 0:
            no_grad_param_names = ['embeddings'] # layer.0
        else:
            no_grad_param_names = ['embeddings', 'pooler'] + ['layer.{}.'.format(i) for i in range(freeze_layer + 1)]
            
        logger.info("The frozen parameters are:")
        for name, param in self.model.named_parameters():
            if any(no_grad_pn in name for no_grad_pn in no_grad_param_names):
                param.requires_grad = False
                logger.info("  {}".format(name))

        self.opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=lr_meta,
                                       warmup=warmup_prop_meta, t_total=max_meta_steps)
        
    def get_optimizer_grouped_parameters(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters
    
    def get_names(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        return names
    
    def get_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return params
    
    def load_weights(self, names, params):
        model_params = self.model.state_dict()
        for n, p in zip(names, params):
            model_params[n].data.copy_(p.data)
            
    def load_gradients(self, names, grads):
        model_params = self.model.state_dict(keep_vars=True)
        for n, g in zip(names, grads):
            if model_params[n].grad is not None:
                model_params[n].grad.data.add_(g.data) # accumulate
#         print(model_params[names[3]].grad)
#             else:
#                 print("*******")
            
    def get_learning_rate(self, lr, progress, warmup, schedule='linear'):
        if schedule == 'linear':
            if progress < warmup:
                lr *= progress / warmup
            else:
                lr *= max((progress - 1.) / (warmup - 1.),  0.)
        return lr
    
    def inner_update(self, data_support, lr_curr, inner_steps, device='cuda'):
        batch_size = len(data_support['orig_query'])
        #print("Hello see the batch size here :", batch_size)
        inner_opt = BertAdam(self.get_optimizer_grouped_parameters(), lr=self.lr_inner, t_total = inner_steps)
        self.model.train()
        for i in range(inner_steps):
            inner_opt.param_groups[0]['lr'] = lr_curr
            inner_opt.param_groups[1]['lr'] = lr_curr
            inner_opt.zero_grad()
            bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(self.max_pred, self.mask_prob, list(self.tokenizer.vocab.keys()), self.tokenizer.convert_tokens_to_ids, self.max_seq_length, new_segment_ids=self.new_segment_ids, truncate_config={'max_len_a': self.max_len_a, 'max_len_b': self.max_len_b, 'trunc_seg': self.trunc_seg, 'always_truncate_tail': self.always_truncate_tail}, mask_source_words=self.mask_source_words, skipgram_prb=self.skipgram_prb, skipgram_size=self.skipgram_size, mask_whole_word=self.mask_whole_word, mode="s2s", has_oracle=self.has_sentence_oracle, num_qkv=self.num_qkv, s2s_special_token=self.s2s_special_token, s2s_add_segment=self.s2s_add_segment, s2s_share_segment=self.s2s_share_segment, pos_shift=self.pos_shift)]
            fn_src = [data_support['orig_context'][i] + '[SEP]' + data_support['orig_answer'][i] for i in range(batch_size)]
            fn_tgt = [data_support['orig_query'][i] for i in range(batch_size)]
            
            train_dataset = seq2seq_loader.Seq2SeqDataset(fn_src, fn_tgt, batch_size, self.tokenizer, self.max_seq_length, file_oracle=None, bi_uni_pipeline=bi_uni_pipeline)
            train_sampler = RandomSampler(train_dataset, replacement=False)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=self.num_workers, collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False)
            for batch in train_dataloader:
                batch = [t.to(device) if t is not None else None for t in batch]
                input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch
                oracle_pos, oracle_weights, oracle_labels = None, None, None
                loss_tuple = self.model(input_ids, segment_ids, input_mask, lm_label_ids, is_next, masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx, masked_pos_2=oracle_pos, masked_weights_2=oracle_weights, masked_labels_2=oracle_labels, mask_qkv=mask_qkv)
                masked_lm_loss, next_sentence_loss = loss_tuple
                loss = masked_lm_loss + next_sentence_loss
                # loss = self.model.forward(data_support['input_ids'], data_support['segment_ids'],
                        # data_support['input_mask'], data_support['start_position'], data_support['end_position'], dtype="qa")
                loss.backward()
    #             print(self.get_optimizer_grouped_parameters()[0]['params'][-1])
                inner_opt.step()
        return loss.item()
    
    def forward_meta(self, batch_query, batch_support, progress, inner_steps, lambda_max_loss, lambda_mask_loss): # for one task
        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)
        meta_grad = []
        device='cuda'
        #print(names)
        meta_loss = []
        task_num = len(batch_query)
        lr_inner = self.get_learning_rate(self.lr_inner, progress, self.warmup_prop_inner)
        # compute meta_grad of each task
        for task_id in range(task_num):
            print("****************************************************************")
#             print(batch_support[task_id])
            self.inner_update(batch_support[task_id], lr_inner, inner_steps)
            bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(self.max_pred, self.mask_prob, list(self.tokenizer.vocab.keys()), self.tokenizer.convert_tokens_to_ids, self.max_seq_length, new_segment_ids=self.new_segment_ids, truncate_config={'max_len_a': self.max_len_a, 'max_len_b': self.max_len_b, 'trunc_seg': self.trunc_seg, 'always_truncate_tail': self.always_truncate_tail}, mask_source_words=self.mask_source_words, skipgram_prb=self.skipgram_prb, skipgram_size=self.skipgram_size, mask_whole_word=self.mask_whole_word, mode="s2s", has_oracle=self.has_sentence_oracle, num_qkv=self.num_qkv, s2s_special_token=self.s2s_special_token, s2s_add_segment=self.s2s_add_segment, s2s_share_segment=self.s2s_share_segment, pos_shift=self.pos_shift)]
            fn_src = [batch_query[task_id]['orig_context'][0] + " [SEP] " + batch_query[task_id]['orig_answer'][0]]
            fn_tgt = [batch_query[task_id]['orig_query'][0]]
            print(fn_tgt[0])
            train_dataset = seq2seq_loader.Seq2SeqDataset(fn_src, fn_tgt, 1, self.tokenizer, self.max_seq_length, file_oracle=None, bi_uni_pipeline=bi_uni_pipeline)
            train_sampler = RandomSampler(train_dataset, replacement=False)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=self.num_workers, collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False)
            for batch in train_dataloader:
                batch = [t.to(device) if t is not None else None for t in batch]
                input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch
                oracle_pos, oracle_weights, oracle_labels = None, None, None
                loss_tuple = self.model(input_ids, segment_ids, input_mask, lm_label_ids, is_next, masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx, masked_pos_2=oracle_pos, masked_weights_2=oracle_weights, masked_labels_2=oracle_labels, mask_qkv=mask_qkv)
                masked_lm_loss, next_sentence_loss = loss_tuple
                loss = masked_lm_loss + next_sentence_loss

            print(loss)
            grad = torch.autograd.grad(loss, params, allow_unused=True)
            meta_grad.append(grad)
            meta_loss.append(loss.item())

            self.load_weights(names, weights)
        # accumulate grads of all tasks to param.grad
        self.opt.zero_grad()
        # similar to backward()
        for g in meta_grad:
            self.load_gradients(names, g)
        print("****************************************************************")
        self.opt.step()
        ave_loss = numpy.mean(numpy.array(meta_loss))
        return ave_loss
                                          
    def forward_NOmeta(self, batch_data): #, lambda_flag=-1.0):
        self.model.train()
        self.opt.zero_grad()
        loss = self.model.forward(batch_data['input_ids'], batch_data['segment_ids'],
                                       batch_data['input_mask'], batch_data['start_position'], 
                           batch_data['end_position'], dtype="qa") #, lambda_flag=lambda_flag)
        loss.backward()
        self.opt.step()
        return loss.item()
     ##---------------------------------------- Evaluation --------------------------------------##

    def write_result(self, words, y_true, y_pred, tmp_fn):
        assert len(y_pred) == len(y_true)
        with open(tmp_fn, 'w', encoding='utf-8') as fw:
            for i, sent in enumerate(y_true):
                for j, word in enumerate(sent):
                    fw.write('{} {} {}\n'.format(words[i][j], word, y_pred[i][j]))
            fw.write('\n')

    def evaluate_meta(self, corpus, result_dir, logger, lr, steps, lang='en', mode='dev', value=0):

        all_results = []
        prediction_file = os.path.join(result_dir, "eval_{}_{}_{}.json".format(lang, mode,value ))
        
        names = self.get_names()
        params = self.get_params()
        weights = deepcopy(params)

        t_tmp = time.time()
        device = 'cuda'
        for item_id in range(corpus.n_total):
            eval_query, eval_support, data_examples, data_features = corpus.get_batch_meta(batch_size=1, shuffle=False)

            # train on support examples
            self.inner_update(eval_support[0], lr_curr=lr, inner_steps=steps)
            print("***********************************************************888")
            # print(eval_query[0]['unique_id'])
            # print(data_features[item_id].unique_id)
            # eval on pseudo query examples (test example)
            mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])
            #torch.cuda.empty_cache()
            eval_model = BertForSeq2SeqDecoder.from_pretrained(self.bert_model, state_dict=self.model.state_dict(), num_labels=self.cls_num_labels, num_rel=0, type_vocab_size=self.type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=3, length_penalty=0, eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=None, forbid_ignore_set=None, not_predict_set=None, ngram_size=3, min_len=None, mode="s2s", max_position_embeddings=self.max_seq_length, ffn_type=self.ffn_type, num_qkv=self.num_qkv, seg_emb=self.seg_emb, pos_shift=self.pos_shift)
            torch.cuda.empty_cache()
            eval_model.to(device)
            eval_model.eval()
            bi_uni_pipeline = []
            bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(self.tokenizer.vocab.keys()), self.tokenizer.convert_tokens_to_ids, self.max_seq_length, max_tgt_length=self.max_pred, new_segment_ids=self.new_segment_ids, mode="s2s", num_qkv=self.num_qkv, s2s_special_token=self.s2s_special_token, s2s_add_segment=self.s2s_add_segment, s2s_share_segment=self.s2s_share_segment, pos_shift=self.pos_shift))

            next_i = 0
            max_src_length = self.max_seq_length - 2 - self.max_pred
            input_lines = [eval_query[0]['orig_context'][0] + " [SEP] " + eval_query[0]['orig_answer'][0]]
            input_lines = [self.tokenizer.tokenize(x)[:max_src_length] for x in input_lines]
            input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
            # output_lines = [""] * len(input_lines)
            batch_size = 1
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += batch_size
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = eval_model(input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                    #output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequence = ' '.join(detokenize(output_tokens))
                        # output_lines[buf_id[i]] = output_sequence
                        all_results.append({"context": eval_query[0]['orig_context'][0], "answers": eval_query[0]['orig_answer'][0], "question": eval_query[0]['orig_query'][0], "gen_question":[output_sequence]})
            
            self.load_weights(names, weights)
            if item_id % 50 == 0:
                logger.info('  To sentence {}/{}. Time: {}sec'.format(item_id, corpus.n_total, time.time() - t_tmp))

        json.dump(all_results, open(prediction_file, "w"), indent = 4)
           
        metrics_dict = evaluate_qg(corpus.tokenizer, prediction_file)
        #print(metrics_dict)
        logger.info('metrics dict: {}'.format(metrics_dict))
        return metrics_dict['Bleu_4'] 
            
    
    def evaluate_NOmeta(self, corpus, result_dir, logger, is_training=False,lang='en', mode='dev', value=0):
        batch_size = 4
        data_batches, data_examples, data_features = corpus.get_batches(is_training,batch_size=batch_size)
        all_results = []
        prediction_file = os.path.join(result_dir, "eval_{}_{}_{}.json".format(lang, mode,value ))
        device = 'cuda'
        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

        t_tmp = time.time()
        eval_model = BertForSeq2SeqDecoder.from_pretrained(self.bert_model, state_dict=self.model.state_dict(), num_labels=self.cls_num_labels, num_rel=0, type_vocab_size=self.type_vocab_size, task_idx=3, mask_word_id=mask_word_id, search_beam_size=3, length_penalty=0, eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=None, forbid_ignore_set=None, not_predict_set=None, ngram_size=3, min_len=None, mode="s2s", max_position_embeddings=self.max_seq_length, ffn_type=self.ffn_type, num_qkv=self.num_qkv, seg_emb=self.seg_emb, pos_shift=self.pos_shift)
        torch.cuda.empty_cache()
        eval_model.to(device)
        eval_model.eval()
        bi_uni_pipeline = []
        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(self.tokenizer.vocab.keys()), self.tokenizer.convert_tokens_to_ids, self.max_seq_length, max_tgt_length=self.max_pred, new_segment_ids=self.new_segment_ids, mode="s2s", num_qkv=self.num_qkv, s2s_special_token=self.s2s_special_token, s2s_add_segment=self.s2s_add_segment, s2s_share_segment=self.s2s_share_segment, pos_shift=self.pos_shift))
        
        for batch_id, eval_query in enumerate(data_batches):

            next_i = 0
            max_src_length = self.max_seq_length - 2 - self.max_pred
            size = len(eval_query['orig_context'])
            input_lines = [eval_query['orig_context'][i] + " [SEP] " + eval_query['orig_answer'][i] for i in range(size)]
            input_lines = [self.tokenizer.tokenize(x)[:max_src_length] for x in input_lines]
            input_lines = list(enumerate(input_lines))

            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += size
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = eval_model(input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                    #output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequence = ' '.join(detokenize(output_tokens))
                        all_results.append({"context": eval_query['orig_context'][i], "answers": eval_query['orig_answer'][i], "question": eval_query['orig_query'][i], "gen_question":[output_sequence]})
            
            #self.load_weights(names, weights)
            if batch_id % 50 == 0:
                logger.info('  To sentence {}/{}. Time: {}sec'.format(batch_id, corpus.n_total, time.time() - t_tmp))

        json.dump(all_results, open(prediction_file, "w"), indent = 4)   
        metrics_dict = evaluate_qg(corpus.tokenizer, prediction_file)
        logger.info('metrics dict: {}'.format(metrics_dict))
        return metrics_dict['Bleu_4']
    
    def evaluate(self, epoch):
            # result directory
        result_file = os.path.join(self.args.result_dir, "dev_eval_{}.txt".format(epoch))
        fw = open(result_file, "a")
        result_dict = dict()
        for dev_file in self.dev_files:
            file_name = dev_file.split(".")[0]
            prediction_file = os.path.join(self.args.result_dir, "epoch_{}_{}.json".format(epoch, file_name))
            file_path = os.path.join(self.args.dev_folder, dev_file)
            metrics = eval_qa(self.model, file_path, prediction_file, args=self.args, tokenizer=self.tokenizer, batch_size=self.args.batch_size)
            f1 = metrics["f1"]
            fw.write("{} : {}\n".format(file_name, f1))
            result_dict[dev_file] = f1
        fw.close()
        return result_dict
    
#     def save_model(self, epoch, loss):
#         loss = round(loss, 3)
#         model_type = "ba"
#         save_file = os.path.join(self.args.save_dir, "{}_{}_{:.3f}.pt".format(model_type, epoch, loss))
#         save_file_config = os.path.join(self.args.save_dir, "{}_config_{}_{:.3f}.json".format(model_type, epoch, loss))
#         model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-s
#         torch.save(model_to_save.state_dict(), save_file)
#         model_to_save.config.to_json_file(save_file_config)
   
    def save_model(self, result_dir, fn_prefix, max_seq_len):
        # Save a trained model and the associated configuration
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        output_model_file = os.path.join(result_dir, '{}_{}'.format(fn_prefix, WEIGHTS_NAME))
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(result_dir, CONFIG_NAME)
        with open(output_config_file, 'w', encoding='utf-8') as f:
            f.write(model_to_save.bert.config.to_json_string())
        model_config = {"bert_model": self.bert_model, "do_lower": False,
                        "max_seq_length": max_seq_len}
        json.dump(model_config, open(os.path.join(result_dir, "model_config.json"), "w", encoding='utf-8'))
