import warnings
warnings.filterwarnings("ignore")

import torch, gc
from abc import ABC
from tqdm import tqdm
from typing import List, Dict
from transformers import BlipProcessor, BlipForQuestionAnswering

class VQAModel(ABC):
    def __init__(self, device:str):
        self.device = device
        self.model = None
        self.processor = None

    def __call__(self,
                 questions:List[str],
                 expected_answers:List[List[str]],
                 images:List,
                 question_weights:List[float]=None,
                 threshold:float=None,
                 idx_to_name:dict=None,
                 use_confidence:bool=True) -> Dict:
        """_Process images in batches and generate answers for each question._

        Args:
            questions (List[str]): List of questions to ask an image
            expected_answers (List[List[str]]): List of expected answers for each question.
            images (List): List of images to process
            question_weights (List[float], optional): List of weights for each question. Defaults to None.
            threshold (float, optional): Threshold for accepting images. Defaults to None.

        Returns:
            dict: Dictionary containing the indices of accepted images, their answers' correctness and their corresponding scores.
        """
        assert len(questions) == len(expected_answers)
        if question_weights:
            assert len(questions) == len(question_weights)
        num_questions = len(questions)
        num_images = len(images)
        
        # Two cases: 
        # (1) threshold a.k.a. questions are weighted
        # (2) no threshold a.k.a. all requirements must be satisfied
        if threshold:
            question_weights = torch.tensor(question_weights, device=self.device)
        else:
            question_weights = torch.ones(num_questions, device=self.device)
            threshold = num_questions
        
        # Looping through questions
        raw_scores = torch.zeros(num_images, num_questions, requires_grad=False, device=self.device)
        scores = torch.zeros(num_images, num_questions, requires_grad=False, device=self.device)
        with torch.no_grad():
            for question_idx, question in tqdm(enumerate(questions), total=num_questions):
                
                # Process batch of images for one particular question
                processed_images = self.processor(images, question, padding=True, return_tensors='pt').to(self.device)
                generated_ids = self.model.generate(**processed_images, output_scores=True, return_dict_in_generate=True)
                answer_scores = generated_ids.scores

                # Calculate confidence
                if use_confidence:
                    topks = [s.softmax(-1).topk(1) for s in answer_scores] 
                    for i, tk in enumerate(topks):
                        if i == 0:
                            probs = tk.values.view(-1).unsqueeze(0)
                        else:
                            probs = torch.concat([probs, tk.values.view(-1).unsqueeze(0)], dim=0)
                    answer_confidences = probs.prod(dim=0).to(self.device)

                generated_answers = self.processor.batch_decode(generated_ids['sequences'], skip_special_tokens=True)
                
                # Check generated-expected answers mismatch 
                assert len(generated_answers) == num_images
                answer_match = torch.tensor([1 if answer in expected_answers[question_idx] else -0.1 for answer in generated_answers], device=self.device)
                raw_scores[:, question_idx] = answer_match
                scores[:, question_idx] = answer_match * answer_confidences if use_confidence else answer_match

                # Clean-up
                del processed_images
                del generated_ids
                del generated_answers
                del answer_match
                if use_confidence:
                    del answer_confidences

                gc.collect()
                torch.cuda.empty_cache()
        
            weighted_scores = torch.sum(scores * question_weights.unsqueeze(0), dim=1)
            accepted_indices = torch.nonzero(weighted_scores >= threshold).squeeze(dim=1).tolist()

        # Format output
        output = {}
        output['accepted_images'] = []
        for idx in range(num_images):
            key = idx_to_name[idx] if idx_to_name else idx
            results = list(map(int, raw_scores[idx].tolist()))
            score = round(weighted_scores[idx].item(), 2)
            output[key] = {
                'results': results,
                'score': score
            }
            if idx in accepted_indices:
                output[key]['accepted'] = True
                output['accepted_images'].append({'image_id': key, 'score': score, 'results': results})
            else:
                output[key]['accepted'] = False
        return output

class BLIPCapliftLarge(VQAModel):
    def __init__(self, device:str):
        super().__init__(device=device)
        self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(self.device).eval()
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")

class VQAEngine:
    def __init__(self, model_name:str, device:str) -> None:
        if model_name == 'blip-vqa-capfilt-large':
            self.model = BLIPCapliftLarge(device=device)
        else:
            raise ValueError(f"Model {model_name} not supported.")
    
    def __call__(self, **kwargs):
        return self.model(**kwargs)