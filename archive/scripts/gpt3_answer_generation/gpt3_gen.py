import os
import openai
import pickle
import sys
sys.path.append('src/')
import utils
from easydict import EasyDict
from typing import Optional
import json
from tqdm import tqdm
import copy



openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.Completion.create(
#     model="gpt-3.5-turbo",
#     prompt="Please answer the following question: who is the current president of the US?",
#     max_tokens=7,
#     temperature=0
# )
# PROMPT = " ".join([
#     "You are a question answering system. ",
#     "You will need to answer the questions based on the knowledge provided. You need to think and reason step by step. ",
#     "But at the end, You must output \"Answer:\" before giving out your final answer. Your answer should be short.",
#     "Here are some examples:",
#     # "Question: who is the current president of the US? Knowledge: The current president of the US is Hillary as of January 2028. Answer: Hillary.",
#     # "Question: who is the current home minister of the UK? Knowledge: Prof. Kate Dolly stepped on as the home minister of the UK in 2030. Answer:",
# ])

PROMPT = """
You are a question answering system. You will need to answer the questions based on the knowledge provided. You need to think and reason step by step. But at the end, You must output \"Answer:\" before giving out your final answer. Your answer should be short.
Here are some examples:

- Question: Name the type of plant this is?
- Caption:  a bathroom with a plant in the middle of it. 
- Object:  wood brown ceiling , white sink , white red blue toothpaste , white closed toilet , white toilet , tall green large plant , silver chrome metal faucet , large black silver mirror , white drawer , round on white light , glass clear shower , white small metal shelf , white shelf , gray white blue wall , white shelf , open white large doorway , green towel , white glass reflection , purple pink pillow , white chair , green hanging plant , white black wall , white round light , white plastic reflection , white small clean bathroom , white shelf , tiled gray tile floor , brown gray square tile , green white small bowl , white couch , white toilet , closed down white lid , open white door , gray brown square tile , wood brown small table , glass open metal door , large framed mirror , white electrical silver outlet , white gray black kitchen , metal glass silver shelf , white silver metal bowl , white cabinet , glass clear door , glass tiled gray wall , red flower , hanging white cup , white green small room , white down lid , white open large room , white toilet toilet paper , white electrical outlet , white plastic empty tissue box , brown tan beige floor , 
- Knowledge: 19 different types of fern plants about privacy contact search for: search menu home about privacy policy contact recent design software home design software interior design software kitchen design software room designer floor plan creator bathroom design software house plans interiors furniture appliances storage backyard home improvement home improvement services tools plumbing calculators top online home décor stores houses types of houses house styles celebrity houses castles mansions historic houses tiny houses all houses exteriors roofing shingles siding windows front doors garages gutters all topics menu search search for: search you are here: home the blog gardens and landscaping 19 different types of fern plants in gardens and landscaping 19 different types of fern plants whether you're into gardening or simply hunting for pretty home decors, you can make lovely houseplants of easy-to-grow ferns by learning about the different types of fern plants that can be found all over the world.  whether you're into gardening or simply hunting for pretty home decors, you can make lovely houseplants of easy-to-grow ferns by learning about the different types of fern plants that can be found all over the world.
- Answer: [UNKNOWN]

- Question: What grip does this woman have on this tennis racket?
- Caption:  a woman holding a tennis racket on a tennis court. 
- Object:  extended up raised arm , white blue string , white line , blue short , pink purple white shirt , blond long hair , yellow ball , gray metal wood bench , raised open up hand , black white red racket , yellow tennis green ball , raised up open hand , red clay orange court , white parked black car , blond brown short head , playing tennis playing tennis tennis , green grass , white pink large ear , white line , black red white tennis racket , playing blond playing tennis woman , green tree , black white green fence , black handle , black dark sunglasses , black white wall , red clay tennis tennis court , pink playing female girl , black white red string , black white sign , white black advertisement , blue white towel , large small white nose , playing playing tennis blond player , black dark brown headband , red clay tennis tennis court , red blond smiling face , red clay tennis court , De Gur Orie B WALTHER.SIKSMA.NL De Gur Orie B WALTHER.SIKSMA.NL 
- Knowledge: opponents simply by the fact that she has the biggest serve on the women's tour!the rules aren't going to change anytime soon, so you should do everything you can to use them to your advantage.  the continental grip is when you hold the racket as though you were holding an axe or a hammer.  streamlined and powerful service motion, allowing for the correct hand motion at contact that is a vital part of a good serve.  see if you can bounce a tennis ball with the bottom edge of your tennis racket.
- Answer: [NO] tight

- Question: How many teeth does this animal use to have?
- Caption:  a cat laying on top of a wooden table looking out a window. 
- Object:  white brown sitting cat , green large tree , pink nose , white long whisker , closed eye , pink pointy brown ear , white gray brown head , gray white back leg , white paw , closed open green eye , orange pink brown ear , white wood brown window , blue clear sky , white long whisker , white foot , wood brown table , bare brown leafless tree , white closed gray face , white pink small mouth , wood brown door , pink white nose , brown long furry tail , white paw , green leafy tall tree , white gray brown wall , white long whisker , white long orange leg , clear blue tree , green large blurry tree , large white green wall , pink mouth , white gray striped face , green tree , 
- Knowledge: for example, a meat-eating animal, such as a cat, has quite different teeth compared to a grass-eating animal, such as a horse.  most cats have 26 deciduous teeth and 30 permanent teeth.
- Answer: [YES] 26
"""

PREDICTION_FILENAME = "/home/wl356/cvnlp_rds/wl356/projects/KBVQA/experiments/OKVQA_VisualColBERT_with_pretrained_ViT(WIT)_ColBERT_mapping_trainable_ViT_frozen_10ROI_with_text_based_vision/test/generate_index/generate_index_test_OKVQADatasetForDPR.test_predictions.json"
DATASET_QUESTION_FILENAME = "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/OpenEnded_mscoco_val2014_questions.json"
DATASET_ANNOTATION_FILENAME = "/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/mscoco_val2014_annotations.json"
PICKLE_FILENAME = "cache/process:LoadOKVQAData-de094bdebf8727ceefec330d85101276.pkl"

CAPTION_MODULE_CONFIG = \
    EasyDict({"type": "TextBasedVisionInput",  "option": "caption",
    "separation_tokens": {'start': '', 'end': ''}})

OBJECT_MODULE_CONFIG = \
    EasyDict({"type": "TextBasedVisionInput",  "option": "object", 
    "object_max": 40, "attribute_max": 3, "attribute_thres":0.05, "ocr": 1,
    "separation_tokens": {'start': '', 'sep': ',', 'end': ''}})

in_context_examples = None

class RAVQAHelper():
    def __init__(self, pickle_filename, caption_module_config, object_module_config, prediction_filename):
        self.caption_module_config = caption_module_config
        self.object_module_config = object_module_config

        self.prediction_filename = prediction_filename
        with open(self.prediction_filename, 'r') as f:
            self.prediction_data = json.load(f)
            self.prediction_lookup = {
                str(item['question_id']): {
                    'top_ranking_passages': item['top_ranking_passages'],
                    'answers': item['answers'],
                    'gold_answer': item['gold_answer']
                } for item in self.prediction_data['output'] 
            }
        print(f"Load predictions from {self.prediction_filename}")

        self.pickle_filename = pickle_filename
        with open(self.pickle_filename, 'rb') as f:
            loaded_dict = pickle.load(f) # keys: 'cache'
        print(f"Load preprocessed data from {self.pickle_filename}")

        self.data = loaded_dict['cache']['okvqa_data']
        self.train_data, self.test_data, self.lookup_data = self.data['train'], self.data['test'], self.data['lookup']
    

    def get_text_based_vision(self, sample: EasyDict, module: EasyDict) -> Optional[EasyDict]:
        """
        Default TextBasedVisionInput module parser
        object: text-based objects, with attributes and OCR'ed texts
        caption: iamge captions
        """
        return_dict = EasyDict(
            text_sequence="",
        )

        # Input from Vision
        vision_sentences = []
        if module.option == 'object':
            vision_sentences += [module.separation_tokens.start]
            for obj in sample.objects:
                attribute_max = module.get('attribute_max', 0)
                if attribute_max > 0:
                    # find suitable attributes
                    suitable_attributes = []
                    for attribute, att_score in zip(obj['attributes'], obj['attribute_scores']):
                        if att_score > module.attribute_thres and len(suitable_attributes) < attribute_max:
                            suitable_attributes.append(attribute)
                    # append to the sentence
                    vision_sentences += suitable_attributes
                vision_sentences.append(obj['class'])
                vision_sentences.append(module.separation_tokens.sep)
            
            ocr = module.get('ocr', 0)
            if ocr > 0:
                text_annotations = sample.img_ocr
                filtered_descriptions = []
                for text_annoation in text_annotations:
                    description = text_annoation['description'].strip()
                    description = description.replace('\n', " ") # remove line switching
                    # vision_sentences += [description]
                    # print('OCR feature:', description)
                    if description not in filtered_descriptions:
                        filtered_descriptions.append(description)
                # print('OCR feature:', filtered_descriptions)
                vision_sentences += filtered_descriptions

            vision_sentences += [module.separation_tokens.end]
            return_dict.text_sequence = ' '.join(vision_sentences)
        
        elif module.option == 'caption':
            return_dict.text_sequence = ' '.join([module.separation_tokens.start] + [sample.img_caption['caption']] + [module.separation_tokens.end])
            
        return return_dict


    
    
    def get_question_and_annotation_by_id(self, question_id, with_knowledge=True, top_k_knowledge=5, with_answer=False):
        def _remove_special_chars(content):
            content = content.replace('<BOK>', '')
            content = content.replace('<EOK>', '')
            content.strip()
            return content
        
        def get_gt_document(documents, answer):
            doc_to_return = []
            for doc in documents:
                if answer.lower() in doc['content'].lower():
                    doc_to_return.append(doc)
            return doc_to_return

        id_ = None
        if type(question_id) == int:
            id_ = str(question_id)
        else:
            id_ = question_id
        data = self.lookup_data[id_]
        dict_to_return = data
        caption_vision = self.get_text_based_vision(data, self.caption_module_config)
        object_vision = self.get_text_based_vision(data, self.object_module_config)
        related_documents = self.prediction_lookup[id_]['top_ranking_passages']
        gt_document = get_gt_document(related_documents, data['gold_answer'])


        disp_str = "\n"
        disp_str += f"- Question: {data['question']}\n"
        disp_str += f"- Caption: {caption_vision['text_sequence']}\n"
        disp_str += f"- Object: {object_vision['text_sequence']}\n"
        all_input_strings = []
        pass
        # input_string = f"Question: {data['question']}\nCaption: {caption_vision['text_sequence']}.\nObjects: {object_vision['text_sequence']}."
        if with_knowledge:
            for k in range(top_k_knowledge):
                disp_str_copy = disp_str + f"- Knowledge: {_remove_special_chars(related_documents[k]['content'])}\n"
                disp_str_copy += f"- Answer: "
                # disp_str += pprint_knowledge_docs(data_dict["top_ranking_passages"], gold_answer=data_dict['gold_answer'], top_k=5) + '\n'
                # disp_str += f"- Answer: {data_dict['gold_answer']}\n"
                if with_answer:
                    disp_str_copy += f"{data['gold_answer']}"
                all_input_strings.append(disp_str_copy)

        dict_to_return.update({
            # 'input_string': input_string,
            'all_input_strings': all_input_strings, 
            'top_ranking_passages': related_documents,
            'caption_string': caption_vision['text_sequence'],
            'object_string': object_vision['text_sequence'],
            'gt_documents': gt_document
        })

        return dict_to_return


IN_CONTEXT_QUESTION_IDS_TO_TEST = [3397615, 3575865] 
QUESTION_IDS_TO_TEST = [2971475, 3397615, 3575865, 949225, 2076115]
TRAIN_QUESTION_IDS = [516065, 817215, 4802085, 5706185, 4789035, 5426515, 250245, 20565, 1795265, 3869275, 4078095, 2141375, 1398785, 4600955, 4106225, 400855, 1665335, 97455, 4731005, 521535, 756435, 977475, 4479775, 1832405, 3161705, 1282645, 4567925, 3829975, 2295255, 2706885, 536155, 751905, 584355, 2961375, 821315, 2694905, 5734855, 2974885, 4383065, 2148345, 593675, 2360155, 4155695, 4532875, 4480465, 367555, 2173065, 1232015, 2838755, 543865, 4682765, 214515, 598435, 2710795, 795525, 2953705, 4082725, 3704935, 1372815, 2957765, 37135, 1235355, 1036765, 3046255, 3701215, 5388215, 710055, 4262745, 4104255, 1824685, 4968915, 546915, 4378575, 817045, 2826695, 564425, 648275, 1653475, 2108065, 839795, 314375, 575155, 250605, 2366265, 4491975, 40425, 3863905, 2897665, 2514755, 2834455, 2476045, 5506605, 831745, 192515]
TEST_QUESTION_IDS = [2971475, 3397615, 3575865, 949225, 2076115, 5723996, 5723995, 5759705, 3045575, 2183655, 2863135, 299845, 115115, 3234605, 5169165, 217115, 3133865, 4026395, 4575845, 4883775, 3017535, 1833195, 3970635, 4976605, 2357845, 3190735, 5144685, 4371265, 5182875, 3804595, 5002005, 3156685, 3736625, 4293695, 4273415, 3938365, 179845, 3268375, 2780325, 1110325, 5096415, 3413095, 1121605, 2645405, 1089825, 2929715, 1902045, 3774865, 1282805, 1318415, 1305165, 2845485, 5069455, 4247445, 5681485, 857725, 1396845, 4732085, 3030245, 5342525, 1649245, 1197855, 511915, 1943135, 3325125, 3763585, 3959785, 2162545, 2913705, 3304055, 505145, 5666345, 1569995, 4535205, 3282845, 2335535, 3589765, 1050405, 5379826, 2352215, 4854245, 2864225, 5192715, 1449855, 5605665, 3430685, 1527025, 1603935, 2597615, 3883985, 4071465, 5525735, 1000065, 5685555, 2873285, 5649115, 4471175, 3294555, 1365725, 5775245, 2707535, 2231575, 3087645, 2940305, 2741565, 5522215, 4684875, 5156125, 2085245, 1970975, 2365425, 2365426, 1041195, 2264595, 2749575, 1341195, 1708525, 1667045, 4100045, 639735, 979885, 4959965, 2980515, 2921885, 5735275, 5461715, 3989925, 3159865, 418675, 1708495, 1803295, 5341945, 195795, 2769645, 5407405, 243965, 5132835, 2993555, 3848225, 5683585, 357265, 2293835, 2542775, 3404725, 1008115, 3591265, 3064155, 584726, 584725, 4258485, 586905, 1547215, 1690895, 3884225]
IN_CONTEXT_QUESTION_IDS_CHEATING_TEST = [3397615]

def ensemble_gpt3_input(prompt, in_context_example_list, question_data_dict, return_with_top_k_doc=5):
    query_string = question_data_dict['input_string']
    in_context_prompt = ''
    for in_context_example_dict in in_context_example_list:
        in_context_prompt += in_context_example_dict['input_string']
        in_context_prompt += ' '

    related_documents = question_data_dict['top_ranking_passages']
    def _remove_special_chars(content):
        content = content.replace('<BOK>', '')
        content = content.replace('<EOK>', '')
        content.strip()
        return content

    strings_to_return = []
    for k in range(return_with_top_k_doc):
        gpt3_input = f"{prompt} {in_context_prompt} Knowledge: {_remove_special_chars(related_documents[k]['content'])}. {query_string}"
        strings_to_return.append(gpt3_input)

    return strings_to_return


def pprint_example(ravqa_helper, question_id, f=None):
    def pprint_knowledge_docs(documents, start_ch='\t', top_k=5, gold_answer=None, display_has_gold_answer=True, write_to_file=None):
        str_to_disp = ""
        for i, doc in enumerate(documents):
            if i >= top_k:
                break
            str_to_disp += f"{start_ch}"
            content = doc['content']
            if gold_answer is not None and display_has_gold_answer:
                if gold_answer.lower() in content.lower():
                    str_to_disp += ' [YES] '
                else:
                    str_to_disp += ' [NO] '
            str_to_disp += f'{i}. {content}\n'
            # print(str_to_disp)
            # if write_to_file is not None:
            #     write_to_file.write(str_to_disp)
        return str_to_disp

    data_dict = ravqa_helper.get_question_and_annotation_by_id(question_id)
    disp_str = "\n"
    disp_str += "="*150 + '\n'
    disp_str += f"Question id: {data_dict['question_id']}".center(150)
    disp_str += '\n'+ "="*150 + '\n'
    disp_str += f"- Question: {data_dict['question']}\n"
    disp_str += f"- Caption: {data_dict['caption_string']}\n"
    disp_str += f"- Object: {data_dict['object_string']}"
    disp_str += f"- Knowledge:\n"
    disp_str += pprint_knowledge_docs(data_dict["top_ranking_passages"], gold_answer=data_dict['gold_answer'], top_k=5) + '\n'
    disp_str += f"- Answer: {data_dict['gold_answer']}\n"
    disp_str += "="*150 + '\n'

    print(disp_str)

    if f is not None:
        f.write(disp_str)


if __name__ == '__main__':
    ravqa_helper = RAVQAHelper(
        pickle_filename=PICKLE_FILENAME, 
        caption_module_config=CAPTION_MODULE_CONFIG,
        object_module_config=OBJECT_MODULE_CONFIG, 
        prediction_filename=PREDICTION_FILENAME
    )

    f = open('scripts/gpt3_answer_generation/okvqa_in_context_test_split_display.txt', 'w')

    # for question_id in QUESTION_IDS_TO_TEST:
    for question_id in tqdm([item['question_id'] for item in ravqa_helper.test_data['data_items']]):
        pprint_example(ravqa_helper, question_id, f=f) 
    print("DONE!")

    f.close()
        # question_data_dict = ravqa_helper.get_question_and_annotation_by_id(question_id)
        # in_context_examples_list = [ravqa_helper.get_question_and_annotation_by_id(q_id, with_answer=True) for q_id in IN_CONTEXT_QUESTION_IDS_TO_TEST]
        # print(question_data_dict)
        # print(in_context_examples_list)
    # res_dict = {}
    # for question_id in QUESTION_IDS_TO_TEST:
    #     question_data = ravqa_helper.get_question_and_annotation_by_id(question_id)
    #     for query in question_data['all_input_strings']:
    #         gpt3_input = PROMPT + query
    #         print(gpt3_input)
    #         input()
    #         res = openai.ChatCompletion.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                 {"role": "system", "content": ""},
    #                 {"role": "user", "content": gpt3_input},
    #                 # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    #                 # {"role": "user", "content": "Where was it played?"}
    #             ]
    #         )
    #         print(res)
    #         pass
        #     pass
    # with open(PICKLE_FILENAME, 'rb') as f:
    #     processed_dict = pickle.load(f) # keys: 'cache'
    # for _, data in processed_dict.items():
    #     print(data.keys()) # 'okvqa_data', 'images': keyed by image path (e.g. '/rds/project/rds-hirYTW1FQIw/shared_space/vqa_data/KBVQA_data/ok-vqa/train2014/COCO_train2014_000000300629.jpg' )
    #     print(data['okvqa_data'].keys()) # ['train', 'test', 'lookup', 'vqa_helpers', 'answer_candidate_list']
    #     print(data['okvqa_data']['test']) # key: 'data_item': [<item0>, <item1>]
    #     print(type(data['okvqa_data']['train']['data_items'])) 

    #     # item example data['okvqa_data']['train']['data_items'][0]
    #     # {'answers': ['race', 'race', 'race', 'race', 'race', 'race', 'motocross', 'motocross', 'ride', ...], 
    #     # 'gold_answer': 'race', 'question': 'What sport can you u... this for?', 
    #     # 'question_id': 2971475, 'img_path': '/rds/project/rds-hir...297147.jpg', 
    #     # 'img_key_full': '000000297147', 'img_key': 297147, 'img_file_name': 'COCO_val2014_000000297147.jpg', 
    #     # 'img': None, 'img_caption': {'caption': 'a black and silver m... building.', 'conf': 0.9351266622543335}, 
    #     # 'objects': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...], 'img_ocr': []}
    #     # object item = {'class': 'seat', 'rect': [375.7681579589844, 142.425048828125, 556.8671875, 230.422119140625], 'attributes': ['black', 'asian', 'airborne', 'american', 'above', '__no_attribute__', 'adult', 'aluminum', 'assorted', ...], 'attribute_scores': [0.8886103630065918, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...], 'ocr': []}
        

    #     print(data['okvqa_data']['lookup'].keys()) # keyed by question id. e.g. '516065'
    #     # lookup item data['okvqa_data']['lookup']['516065']
    #     # {'answers': ['pony tail', 'pony tail', 'pony tail', 'pony tail', 'pony tail', 'pony tail', 'braid', 'braid', 'ponytail', ...], 
    #     # 'gold_answer': 'pony tail', 'question': 'What is the hairstyl...nd called?', 'question_id': 516065, 
    #     # 'img_path': '/rds/project/rds-hir...051606.jpg', 'img_key_full': '000000051606', 'img_key': 51606, 
    #     # 'img_file_name': 'COCO_train2014_00000...051606.jpg', 'img': None, 
    #     # 'img_caption': {'caption': 'two women standing n...s rackets.', 'conf': 0.9791389107704163}, 
    #     # 'objects': [{...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, {...}, ...], 'img_ocr': []}
        
    #     print(data['okvqa_data']['vqa_helpers'].keys())
    #     print(data['okvqa_data']['answer_candidate_list'].keys())
    