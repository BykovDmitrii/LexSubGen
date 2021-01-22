import logging
import os
import json
import random
import torch
import numpy as np
import pickle
import hashlib
from string import punctuation
from pathlib import Path
from typing import List, Tuple, Dict, NoReturn
from overrides import overrides
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM

from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator
from lexsubgen.utils.register import memory


class XLMRProbEstimator(EmbSimProbEstimator):
    def __init__(
        self,
        model_name: str = "xlm-roberta-large",
        cuda_device: int = 0,
        temperature: float = 1.0,
        sim_func: str = "dot-product",
        use_attention_mask: bool = False,
        verbose: bool = False,
        aggr_stratagy: str = "mul_probs_with_coefs",
        num_masks: int = 3,
        beam_search: bool = False,
        beam_size: int = 3,
        max_num_mask: int = 5,
        top_k: int = 150,
        coef_2_mask: float = 1,
        coef_3_mask: float = 1,
        coef_4_mask: float = 1,
        coef_5_mask: float = 1,
    ):
        cuda_device = 0
        super(XLMRProbEstimator, self).__init__(
            model_name=model_name,
            temperature=temperature,
            sim_func=sim_func,
            verbose=verbose,
        )
        self.coefs = [1.0, 1.0, coef_2_mask, coef_3_mask, coef_4_mask, coef_5_mask]
        self.aggr_stratagy = aggr_stratagy
        self.model_name = model_name
        self.num_masks = num_masks
        self.max_num_mask = max_num_mask
        self.beam_size = beam_size if beam_search else 1
        self.use_attention_mask = use_attention_mask
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        if cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")
        self.top_k = top_k
        self.descriptor = {
            "Prob_estimator": {
                "name": "xlmr",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "temperature": self.temperature,
                "use_attention_mask": self.use_attention_mask,
            }
        }
        self.register_model()
        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")
        
    @property
    def tokenizer(self):
        """
        Model tokenizer.

        Returns:
            `transformers.RobertaTokenizer` tokenzier related to the model
        """
        return self.loaded[self.model_name]["tokenizer"]

    @property
    def id2word(self) -> Dict[int, str]:
        """
        Attribute that acquires model vocabulary.

        Returns:
            vocabulary represented as a `dict`
        """
        return self.loaded[self.model_name]["id2word"]
    
    @property
    def model(self):
        """
        Attribute that acquires underlying vectorization model.

        Returns:
            Vectorization model.
        """
        return self.loaded[self.model_name]["model"]

    def _prepare_inputs(self):
        pass
    
    @staticmethod
    def load_word2id(tokenizer: XLMRobertaTokenizer) -> Dict[str, int]:
        """
        Loads model vocabulary in the form of mapping from words to their indexes.

        Args:
            tokenizer: `transformers.RobertaTokenizer` tokenizer

        Returns:
            model vocabulary
        """
        word2id = dict()
        for word_idx in range(tokenizer.vocab_size):
            word = tokenizer.convert_ids_to_tokens([word_idx])[0]
            word2id[word] = word_idx
        return word2id    


    @staticmethod
    def load_filter_word_ids(word2id: Dict[str, int], filter_chars: str) -> List[int]:
        """
        Gathers words that should be filtered from the end distribution, e.g.
        punctuation.

        Args:
            word2id: model vocabulary
            filter_chars: words with this chars should be filtered from end distribution.

        Returns:
            Indexes of words to be filtered from the end distribution.
        """
        filter_word_ids = []
        set_filter_chars = set(filter_chars)
        for word, idx in word2id.items():
            if len(set(word) & set_filter_chars):
                filter_word_ids.append(idx)
        return filter_word_ids    
    
    @staticmethod
    def tokenize_around_target(
        tokens: List[str],
        target_idx: int,
        tokenizer: XLMRobertaTokenizer = None,
    ):
        left_specsym_len = 1  # for BERT / ROBERTA there is 1 spec token before text
        input_text = ' '.join(tokens)
        tokenized_text = tokenizer.encode(' ' + input_text, add_special_tokens=True)

        left_ctx = ' '.join(tokens[:target_idx])
        target_start = left_specsym_len + len(tokenizer.encode(
            ' ' + left_ctx, add_special_tokens=False
        ))

        left_ctx_target = ' '.join(tokens[:target_idx + 1])
        target_subtokens_ids = tokenizer.encode(
            ' ' + left_ctx_target, add_special_tokens=False
        )[target_start - left_specsym_len:]

        return tokenized_text, target_start, target_subtokens_ids
    
    def prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        tokenizer: XLMRobertaTokenizer = None,
    ):
        """
        Prepares batch of contexts and target indexes into the form
        suitable for processing with BERT, e.g. tokenziation, addition of special tokens
        like [CLS] and [SEP], padding contexts to have the same size etc.

        Args:
            batch_of_tokens: list of contexts
            batch_of_target_ids: list of target word indexes
            tokenizer: tokenizer to use for word tokenization

        Returns:
            transformed contexts and target word indexes in these new contexts
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        roberta_batch_of_tokens, roberta_batch_of_target_ids = [], []
        max_seq_len = 0

        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            tokenized = self.tokenize_around_target(tokens, target_idx, tokenizer)
            context, target_start, target_subtokens_ids = tokenized

            context = context[:target_start] + \
                      [tokenizer.mask_token_id] * self.num_masks + \
                      context[target_start + len(target_subtokens_ids):]

            if len(context) > 512:
                first_subtok = context[target_start]
                # Cropping maximum context around the target word
                left_idx = max(0, target_start - 256)
                right_idx = min(target_start + 256, len(context))
                context = context[left_idx: right_idx]
                target_start = target_start if target_start < 256 else 255
                assert first_subtok == context[target_start]

            max_seq_len = max(max_seq_len, len(context))

            roberta_batch_of_tokens.append(context)
            roberta_batch_of_target_ids.append([ind for ind, val in enumerate(context) if val == tokenizer.mask_token_id])

        assert max_seq_len <= 512

        input_ids = np.vstack([
            tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
            for tokens in roberta_batch_of_tokens
        ])

        input_ids = torch.tensor(input_ids).to(self.device)

        return input_ids, torch.LongTensor(roberta_batch_of_target_ids)

    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        XLMRProbEstimator.loaded = dict()
        if self.model_name not in XLMRProbEstimator.loaded:
            xlm_roberta_model = XLMRobertaForMaskedLM.from_pretrained(self.model_name)
            xlm_roberta_model.to(self.device).eval()
            xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name)
            xlm_roberta_word2id = XLMRProbEstimator.load_word2id(xlm_roberta_tokenizer)
            xlm_roberta_id2word = {id: word for word, id in xlm_roberta_word2id.items()}
            filter_word_ids = XLMRProbEstimator.load_filter_word_ids(
                xlm_roberta_word2id, punctuation
            )
            word_embeddings = (
                xlm_roberta_model.lm_head.decoder.weight.data.cpu().numpy()
            )

            norms = np.linalg.norm(word_embeddings, axis=-1, keepdims=True)
            normed_word_embeddings = word_embeddings / norms

            XLMRProbEstimator.loaded[self.model_name] = {
                "model": xlm_roberta_model,
                "tokenizer": xlm_roberta_tokenizer,
                "embeddings": word_embeddings,
                "normed_embeddings": normed_word_embeddings,
                "id2word": xlm_roberta_id2word,
                "filter_word_ids": filter_word_ids,
            }
            XLMRProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            XLMRProbEstimator.loaded[self.model_name]["ref_count"] += 1

    def fill_first_mask(
        self, input_ids, target_ids, attention_mask,
    ) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]
            logits = np.vstack([
                logits[idx, target_idx, :].cpu().numpy()
                for idx, target_idx in enumerate(target_ids)
            ])
            return torch.Tensor(logits)
    
    def compute_batch_partialy(self, tensor: torch.Tensor, masked_indexes: torch.Tensor, batch_size: int = 2000):
        num_rows_per_butch = batch_size // tensor.shape[-1]
        result_features = []
        torch.cuda.empty_cache()
        tensor = tensor.long().to(self.device)
        for i in range(0, tensor.shape[0], num_rows_per_butch):
            features = self.model(tensor[i : i + num_rows_per_butch])[0]
            result_features.append(features[:, masked_indexes[:1], :].to('cpu'))
        tensor = tensor.long().to('cpu')
        logits = torch.cat(result_features, dim=0)
        return logits.reshape(logits.shape[0], logits.shape[-1])

    def fill_masks_continuation_with_beam_search(self, tokens: torch.Tensor, masked_indexes: torch.Tensor):
        result_tensors = tokens.reshape(tokens.shape[0], 1, -1).to('cpu')
        probs = torch.tensor([1.0] * tokens.shape[0]).to('cpu').reshape(tokens.shape[0], -1)
        for _ in range(masked_indexes.shape[0]):
            with torch.no_grad():
                logits = self.compute_batch_partialy(result_tensors.reshape(-1, tokens.shape[-1]), masked_indexes)
            result_tensors = result_tensors.reshape(tokens.shape[0], -1, tokens.shape[-1])
            logits = logits.reshape(tokens.shape[0], result_tensors.shape[1], -1)
            prob = logits.softmax(dim=-1)
            values, index = prob.topk(k=self.beam_size, dim=-1)
            probs = probs.reshape(result_tensors.shape[0], result_tensors.shape[1], 1)
            new_probs = (probs * values).reshape(probs.shape[0], -1)
            topk_index = new_probs.argsort(dim=1)[:, -self.beam_size:]
            result_tensors_lst = []
            probs_lst = []
            for i in range(topk_index.shape[0]):
                result_tensors_cur = result_tensors[i].repeat_interleave(self.beam_size, dim=0)[topk_index[i]]
                result_tensors_cur[:, masked_indexes[0]] = index[i].reshape(-1)[topk_index[i]]
                probs_lst.append(new_probs[i, topk_index[i]])
                result_tensors_lst.append(result_tensors_cur)
            masked_indexes = masked_indexes[1:]
            result_tensors = torch.stack(result_tensors_lst)
            probs = torch.stack(probs_lst)
        return probs[:, -1], result_tensors[:, -1]

    def fill_masks(self, tokens_lists: List[List[str]], target_ids: List[int]):
        input_ids, target_ids = self.prepare_batch(tokens_lists, target_ids)
        attention_mask = None
        if self.use_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id)
            attention_mask = attention_mask.float().to(input_ids)
        logprobs = self.fill_first_mask(input_ids, target_ids[:, 0], attention_mask)
        ids = logprobs.argsort()[:, -self.top_k:]
        probs = []
        result_subst = ids
        result_subst = []
        for tokens, ids_for_tokens, targets_ids_for_tokens, probs_for_tokens in zip(input_ids, ids, target_ids, logprobs.softmax(dim=-1)):
            t_l = []
            for idx in ids_for_tokens:
                tokens[targets_ids_for_tokens[0]] = int(idx)
                t_l.append(tokens.clone().detach())
            input_filled = torch.stack(t_l) 
            pred_probs, pred_tokens = self.fill_masks_continuation_with_beam_search(input_filled, targets_ids_for_tokens[1:])
            for idx, pred_prob, pred_token in zip(ids_for_tokens, pred_probs, pred_tokens):
                if len(self.get_subs_from_token_list([int(pred_token[i]) for i in targets_ids_for_tokens]).split()) == 1:
                    result_subst.append([int(pred_token[i]) for i in targets_ids_for_tokens])
                    probs.append(pred_prob * probs_for_tokens[idx])
        return probs, result_subst

    def get_subs_from_token_list(self, ids: List[int]):
        result_subs = ""
        for idx in ids:
            result_subs += self.id2word[idx]
        return result_subs.replace("\u2581", " ").strip()
    
    def apply_mul_to_align_probs(self, probs: List[float]):
        return [p * self.coefs[self.num_masks] for p in probs]

    def fill_masks_cashed(self, tokens_lists: List[List[str]], target_ids: List[int]):
        file_name = str(hashlib.md5((str(tokens_lists) + "_" + str(self.num_masks) + "_" + str(self.top_k) + "_" + str(self.beam_size)).encode()).hexdigest())
        file_path = "cash/" + file_name
        if not(os.path.exists("cash")):
            os.mkdir("cash")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                probs_fixed_num_mask, result_subst_fixed_num_masks =  pickle.load(f)
        else:
            probs_fixed_num_mask, result_subst_fixed_num_masks = self.fill_masks(tokens_lists, target_ids)
            with open(file_path, "wb") as f:
                pickle.dump((probs_fixed_num_mask, result_subst_fixed_num_masks), f)
        return probs_fixed_num_mask, result_subst_fixed_num_masks

    def predict(self, tokens_lists: List[List[str]], target_ids: List[int]):
        probs = []
        result_subst = []
        if self.aggr_stratagy == "mul_probs_with_coefs":
            for self.num_masks in range(1, self.max_num_mask+1):
                probs_fixed_num_mask, result_subst_fixed_num_masks = self.fill_masks_cashed(tokens_lists, target_ids)
                probs = probs + self.apply_mul_to_align_probs(probs_fixed_num_mask)
                result_subst = result_subst + result_subst_fixed_num_masks
        elif self.aggr_stratagy == "single":
            probs, result_subst = self.fill_masks_cashed(tokens_lists, target_ids)
        subs_to_id = {self.get_subs_from_token_list(subs): id for id, subs in enumerate(result_subst)}
        return torch.Tensor(probs).reshape(1, -1), subs_to_id

    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        if len(tokens_lists) > 1:
            raise Exception("batch size must be 1 for xlmr")
        return self.predict(tokens_lists, target_ids)
