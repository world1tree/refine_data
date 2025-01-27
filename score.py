import re
import json
import editdistance
import sacrebleu as sb
import unicodedata

class BleuScorer:
    def __init__(self, tokenizer_type="13a"):
        # print("tokenizer type: %s, please check it." % tokenizer_type)
        self.tokenizer = sb.BLEU(tokenize=tokenizer_type).tokenizer
        self.num_sents = 0
        self.pred = list()
        self.ref = list()
        self.pred_origin = list()
        self.ref_origin = list()

    def add_sent(self, ref=None, pred=None):
        assert ref is not None
        assert pred is not None
        self.pred.append(self.tokenizer(pred))
        self.ref.append(self.tokenizer(ref))

        self.pred_origin.append(pred)
        self.ref_origin.append(ref)
        self.num_sents += 1

    def result_string(self):
        return f"{self.score()} \nSents: {self.num_sents}"

    def score(self):
        return sb.corpus_bleu(self.pred, [self.ref], tokenize="none")

    def score_float(self):
        s = sb.corpus_bleu(self.pred, [self.ref], tokenize="none")
        score = float(str(s).split()[2])
        return score

    def prepare_for_comet(self):

        with open('pred.txt', 'w') as f:
            for line in self.pred_origin:
                f.write(line.strip() + '\n')

        with open('ref.txt', 'w') as f:
            for line in self.ref_origin:
                f.write(line.strip() + '\n')

class WerScorer:

    def __init__(self, tokenizer_type="13a"):
        self.distance = 0
        self.ref_length = 0
        self.num_sents = 0
        self.tokenizer = sb.BLEU(tokenize=tokenizer_type).tokenizer
        self.pred_origin = list()

    def remove_punctuation(self, sent: str):
        """Remove punctuation based on Unicode category."""
        SPACE = chr(32)
        return SPACE.join(
            t
            for t in sent.split(SPACE)
            if not all(unicodedata.category(c)[0] == "P" for c in t)
        )

    def add_sent(self, ref=None, pred=None):
        assert ref is not None
        assert pred is not None
        self.pred_origin.append(pred.strip())

        ref = self.tokenizer(ref)
        pred = self.tokenizer(pred)

        ref = self.remove_punctuation(ref)
        pred = self.remove_punctuation(pred)

        ref_items = ref.split()
        pred_items = pred.split()
        self.distance += editdistance.eval(ref_items, pred_items)
        self.ref_length += len(ref_items)
        self.num_sents += 1

    def result_string(self):
        return f"WER: {self.score():.2f}\nSents: {self.num_sents}"

    def score(self):
        return 100.0 * self.distance / self.ref_length if self.ref_length > 0 else 0

    def write_pred(self):

        with open('asr_pred.txt', 'w') as f:
            for line in self.pred_origin:
                f.write(line.strip() + '\n')

class Evaluator(object):

    def split_sent_st_only(self, pred_doc, ref_doc):
        # 只修复st, 不修复asr
        # 使用换行拆开，然后检测句子数是否能够对的上
        pred_list = pred_doc.split('\n')
        ref_list = ref_doc.split('\n')
        ret = {
            'split_ok': False,
            'pred_list': [],
            'ref_list': [],
        }
        # 移除可能的#num 开头
        pred_list = [re.sub(r'^(#\d+ )', '', sent.strip()) for sent in pred_list]
        ref_list = [re.sub(r'^(#\d+ )', '', sent.strip()) for sent in ref_list]
        # 长度对得上
        if len(pred_list) == len(ref_list):
            ret['split_ok'] = True
        else:
            pred_list = list(filter(lambda x: x!='', pred_list))
            if len(pred_list) == 4:
                ret['split_ok'] = True
            else:
                print(pred_list)
                if len(pred_list) == 2:
                    if 'Refined Transcription: ' in pred_list[0]:
                        pred_list[0] = pred_list[0].replace('Refined Transcription: ', 'Refined Transcription: \n')
                    if 'Refined Translation: ' in pred_list[1]:
                        pred_list[1] = pred_list[1].replace('Refined Translation: ', 'Refined Translation: \n')
                    pred_list = pred_list[0].split('\n') + pred_list[1].split('\n')
                if len(pred_list) == 4:
                    ret['split_ok'] = True
                else:
                    # 补空如果长度对不上
                    diff = len(ref_list) - len(pred_list)
                    if diff > 0:
                        pred_list = pred_list + [''] * diff
                    elif diff < 0:
                        pred_list = pred_list[:4]
                    assert len(pred_list) == len(ref_list)
                    ret['split_ok'] = False

        # st的结果
        assert len(ref_list) == 4
        ret['st_pred'] = [pred_list[-1]]
        ret['st_ref'] = [ref_list[-1]]
        ret['asr_pred'] = [pred_list[-3]]
        ret['asr_ref'] = [ref_list[-3]]
        
        return ret

    def split_sent(self, pred_doc, ref_doc):
        # 使用换行拆开，然后检测句子数是否能够对的上
        pred_list = pred_doc.split('\n')
        ref_list = ref_doc.split('\n')
        ret = {
            'split_ok': False,
            'pred_list': [],
            'ref_list': [],
        }
        # 移除可能的#num 开头
        pred_list = [re.sub(r'^(#\d+ )', '', sent.strip()) for sent in pred_list]
        ref_list = [re.sub(r'^(#\d+ )', '', sent.strip()) for sent in ref_list]
        # 长度对得上
        if len(pred_list) == len(ref_list):
            ret['split_ok'] = True
        else:
            # 如果长度对不上，那么用修复前的结果进行填充
            diff = len(ref_list) - len(pred_list)
            pred_list = pred_list + [''] * diff
            assert len(pred_list) == len(ref_list)
            ret['split_ok'] = False

        # asr的结果
        asr_begin = 1
        asr_end = len(ref_list) // 2
        ret['asr_pred'] = [pred_list[k] for k in range(asr_begin, asr_end)]
        ret['asr_ref'] = [ref_list[k] for k in range(asr_begin, asr_end)]

        # st的结果
        st_begin = len(ref_list) // 2 + 1
        st_end = len(ref_list)
        ret['st_pred'] = [pred_list[k] for k in range(st_begin, st_end)]
        ret['st_ref'] = [ref_list[k] for k in range(st_begin, st_end)]

        return ret

    def doc_to_map(self, doc):
        # 提取出asr和st的结果
        sent_list = doc.split('\n')
        asr = dict()
        st = dict()
        use_asr = True
        for sent in sent_list:
            res = re.search(r'^(#\d+) (.*)', sent)
            if res is not None:
                key = res.group(1)
                value = res.group(2)
                if key in asr:
                    use_asr = False
                if use_asr:
                    asr[key] = value
                else:
                    st[key] = value
        return asr, st

    def is_reasonable(self, sent):
        words = sent.split()
        if len(words) > 500:
            return False
        for word in words:
            if len(word) >= 50:
                return False
        return True

    def split_doc_cb(self, pred_doc, ref_doc, idx):
        # 使用换行拆开，然后检测句子数是否能够对的上
        asr_origin, st_origin = self.instruction_list[idx]['asr'], self.instruction_list[idx]['st']
        asr_pred, st_pred = self.doc_to_map(pred_doc)
        asr_gold, st_gold = self.doc_to_map(ref_doc)

        ret = {
            'split_ok': True,
            'reason': [],
            'index': idx,
            'pred_list': [],
            'ref_list': [],
        }

        # 先处理句子数对不上的情况
        if not (asr_origin.keys() == asr_pred.keys() == asr_gold.keys()):
            assert asr_origin.keys() == asr_gold.keys()
            assert len(asr_pred) < len(asr_gold)
            # 遍历asr_origin, 如果asr_pred中没有，那么设置
            for k in asr_origin.keys():
                if k not in asr_pred:
                    asr_pred[k] = asr_origin[k]
                    ret['split_ok'] = False
                    ret['reason'].append(f'ASR-{idx}-{k} 对不上')
            assert asr_origin.keys() == asr_pred.keys() == asr_gold.keys()

        # 再判断是否有超级长的句子，如果预测的句子超过原本句子5倍的长度，那么认为解码是错误的
        # 此时把解码结果设置为origin
        for k in asr_pred.keys():
            if not self.is_reasonable(asr_pred[k]):
                asr_pred[k] = asr_origin[k]
                ret['split_ok'] = False
                ret['reason'].append(f'ASR-{idx}-{k} 太长')

        if not (st_origin.keys() == st_pred.keys() == st_gold.keys()):
            assert st_origin.keys() == st_gold.keys()
            assert len(st_pred) < len(st_gold)
            for k in st_origin.keys():
                if k not in st_pred:
                    st_pred[k] = st_origin[k]
                    ret['split_ok'] = False
                    ret['reason'].append(f'ST-{idx}-{k} 对不上')
            assert st_origin.keys() == st_pred.keys() == st_gold.keys()
        for k in st_pred.keys():
            if not self.is_reasonable(st_pred[k]):
                st_pred[k] = st_origin[k]
                ret['split_ok'] = False
                ret['reason'].append(f'ST-{idx}-{k} 太长')

        ret['asr_pred'] = [asr_pred[k] for k in sorted(asr_pred.keys())]
        ret['asr_ref'] = [asr_gold[k] for k in sorted(asr_gold.keys())]
        ret['st_pred'] = [st_pred[k] for k in sorted(st_pred.keys())]
        ret['st_ref'] = [st_gold[k] for k in sorted(st_gold.keys())]

        return ret

    def split_doc_sw(self, pred_doc, ref_doc, idx):
        ret = self.split_doc_cb(pred_doc, ref_doc, idx)
        # 只保留最后一个asr结果
        ret['asr_pred'] = [ret['asr_pred'][-1]]
        ret['asr_ref'] = [ret['asr_ref'][-1]]
        # 只保留最后一个st结果
        ret['st_pred'] = [ret['st_pred'][-1]]
        ret['st_ref'] = [ret['st_ref'][-1]]
        return ret

    def __init__(self, json_file="generated_predictions.jsonl", input_json=None):
        self.json_file = json_file
        self.input_json = input_json

        self.origin_asr = list()
        self.origin_st = list()
        # 读取input_json中的数据
        with open(self.input_json) as f:
            data = json.load(f)
            for d in data:
                instruction = d['instruction'].strip()
                inst_list = [re.sub(r'^(#\d+ )', '', sent.strip()) for sent in instruction.split('\n')]
                inst_list = inst_list[1:]
                assert len(inst_list) == 4, inst_list
                self.origin_asr.append(inst_list[1])
                self.origin_st.append(inst_list[3])
                # asr_map, st_map = self.doc_to_map(instruction)
                # self.instruction_list.append(
                    # {
                        # 'asr': asr_map,
                        # 'st': st_map,
                    # }
                # )

    def calculate_sent_st_only(self):
        bleu_scorer = BleuScorer()
        wer_scorer = WerScorer()
        skip_doc = 0
        all_doc = 0
        result_list = list()
        with open(self.json_file) as f:
            for index, line in enumerate(f):
                data = json.loads(line)
                predict = data['predict']
                label = data['label']
                result = self.split_sent_st_only(predict, label)
                if not result["split_ok"]:
                    result['st_pred']= [self.origin_st[index]]
                    result['asr_pred']= [self.origin_asr[index]]
                    skip_doc += 1
                all_doc += 1
                json_dict = {}
                for p, r in zip(result['st_pred'], result['st_ref']):
                    bleu_scorer.add_sent(ref=r, pred=p)

                    _bleu = BleuScorer()
                    _bleu.add_sent(ref=r, pred=p)
                    json_dict['st_origin'] = self.origin_st[index]
                    json_dict['st_pred'] = p
                    json_dict['st_ref'] = r
                    json_dict['bleu'] = _bleu.score_float()
                for p, r in zip(result['asr_pred'], result['asr_ref']):
                    wer_scorer.add_sent(ref=r, pred=p)

                    _wer = WerScorer()
                    _wer.add_sent(ref=r, pred=p)
                    json_dict['asr_origin'] = self.origin_asr[index]
                    json_dict['asr_pred'] = p
                    json_dict['asr_ref'] = r
                    json_dict['wer'] = _wer.score()
                result_list.append(json_dict)

        # 输出BLEU相关结果
        print('skip ', skip_doc)
        print('all ', all_doc)
        print(bleu_scorer.result_string())
        print(wer_scorer.result_string())
        # 输出pred.txt和ref.txt用于comet计算
        # bleu_scorer.prepare_for_comet()
        print(len(result_list))
        # 排序
        self.write_sorted_wer(sorted(result_list, key = lambda key: -key['wer']))
        self.write_sorted_bleu(sorted(result_list, key = lambda key: -key['bleu']))
        # 写文件
        bleu_scorer.prepare_for_comet()
        wer_scorer.write_pred()

    def write_sorted_wer(self, lst):
        with open('wer.txt', 'w') as f:
            for idx, json_dict in enumerate(lst):
                f.write('-'*10 + str(idx) + '-'*10 + '\n')
                f.write('asr_origin: ' + json_dict['asr_origin'] + '\n')
                f.write('st_origin: ' + json_dict['st_origin'] + '\n')
                f.write('wer: ' + str(json_dict['wer']) + '\n')
                f.write('asr_pred: ' + json_dict['asr_pred'] + '\n')
                f.write('asr_ref: ' + json_dict['asr_ref'] + '\n')
                f.write('bleu: ' + str(json_dict['bleu']) + '\n')
                f.write('st_pred: ' + json_dict['st_pred'] + '\n')
                f.write('st_ref: ' + json_dict['st_ref'] + '\n')

    def write_sorted_bleu(self, lst):
        with open('bleu.txt', 'w') as f:
            for idx, json_dict in enumerate(lst):
                f.write('-'*10 + str(idx) + '-'*10 + '\n')
                f.write('asr_origin: ' + json_dict['asr_origin'] + '\n')
                f.write('st_origin: ' + json_dict['st_origin'] + '\n')
                f.write('bleu: ' + str(json_dict['bleu']) + '\n')
                f.write('st_pred: ' + json_dict['st_pred'] + '\n')
                f.write('st_ref: ' + json_dict['st_ref'] + '\n')
                f.write('wer: ' + str(json_dict['wer']) + '\n')
                f.write('asr_pred: ' + json_dict['asr_pred'] + '\n')
                f.write('asr_ref: ' + json_dict['asr_ref'] + '\n')

    def calculate_sent(self):
        wer_scorer = WerScorer()
        bleu_scorer = BleuScorer()
        skip_doc = 0
        all_doc = 0
        with open(self.json_file) as f:
            for line in f:
                data = json.loads(line)
                predict = data['predict']
                label = data['label']
                result = self.split_sent(predict, label)
                if not result["split_ok"]:
                    skip_doc += 1
                all_doc += 1
                for p, r in zip(result['asr_pred'], result['asr_ref']):
                    wer_scorer.add_sent(ref=r, pred=p)
                for p, r in zip(result['st_pred'], result['st_ref']):
                    bleu_scorer.add_sent(ref=r, pred=p)

        # 输出wer相关结果
        print("Doc skip/all: %d/%d" % (skip_doc, all_doc))
        print(wer_scorer.result_string())
        print('-'*40)
        # 输出BLEU相关结果
        print(bleu_scorer.result_string())
        # 输出pred.txt和ref.txt用于comet计算
        bleu_scorer.prepare_for_comet()

    def calculate_doc(self, use_cb=False, use_sw=False):
        assert use_cb ^ use_sw
        wer_scorer = WerScorer()
        bleu_scorer = BleuScorer()
        skip_doc = 0
        all_doc = 0
        with open(self.json_file) as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                predict = data['predict']
                label = data['label']

                if use_cb:
                    result = self.split_doc_cb(predict, label, idx)
                elif use_sw:
                    result = self.split_doc_sw(predict, label, idx)

                if not result["split_ok"]:
                    # reason
                    print('-'*25)
                    for r in result['reason']:
                        print(r)
                    print('-'*25)
                    # reason
                    skip_doc += 1
                all_doc += 1
                for p, r in zip(result['asr_pred'], result['asr_ref']):
                    wer_scorer.add_sent(ref=r, pred=p)
                for p, r in zip(result['st_pred'], result['st_ref']):
                    bleu_scorer.add_sent(ref=r, pred=p)

        # 输出wer相关结果
        print("Doc skip/all: %d/%d" % (skip_doc, all_doc))
        print(wer_scorer.result_string())
        print('-'*40)
        # 输出BLEU相关结果
        print(bleu_scorer.result_string())
        # 输出pred.txt和ref.txt用于comet计算
        bleu_scorer.prepare_for_comet()

if __name__ == '__main__':
    # doc2doc: cb
    # Evaluator(input_json='/data/hxdou/0-Inbox/Refine/data/refine/refine-doc-test-v4.json').calculate_doc(use_cb=True)
    # doc2doc: sw
    Evaluator(input_json='/public/home/zhxgong/hxdou/0-Inbox/Refine/data/refine/sent-test-v4.json').calculate_sent_st_only()
    # sent2sent: st only
    # Evaluator().calculate_sent_st_only()
    # sent2sent: asr + st
    # Evaluator().calculate_sent()
