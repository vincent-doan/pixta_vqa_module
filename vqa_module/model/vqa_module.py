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
                 batch_size:int,
                 question_weights:List[float]=None,
                 threshold:float=None,
                 idx_to_name:dict=None) -> Dict:
        """_Process images in batches and generate answers for each question._

        Args:
            questions (List[str]): List of questions to ask an image
            expected_answers (List[List[str]]): List of expected answers for each question.
            images (List): List of images to process
            batch_size (int): Number of images to process at once
            question_weights (List[float], optional): List of weights for each question. Defaults to None.
            threshold (float, optional): Threshold for accepting images. Defaults to None.

        Returns:
            dict: Dictionary containing the indices of accepted images, their answers' correctness and their corresponding scores.
        """
        assert len(questions) == len(expected_answers)
        assert len(questions) == len(question_weights)
        assert sum(question_weights) == 1.0
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

        # For each question, loop over a batch of images
        scores = torch.zeros(num_images, num_questions, requires_grad=False, device=self.device)
        with torch.no_grad():
            for image_idx in tqdm(range(0, num_images, batch_size)):
                for question_idx, question in enumerate(questions):
                    # Determine batch of images to process
                    start_idx = image_idx
                    end_idx = image_idx + batch_size if num_images - start_idx >= batch_size else num_images
                    
                    # Process batch of images for one particular question
                    processed_images = self.processor(images[start_idx: end_idx], question, padding=True, return_tensors='pt').to(self.device)
                    generated_ids = self.model.generate(**processed_images)
                    generated_answers = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    
                    # Check generated-expected answers mismatch
                    assert len(generated_answers) == end_idx - start_idx
                    answer_match = torch.tensor([1 if answer in expected_answers[question_idx] else 0 for answer in generated_answers])
                    scores[start_idx: end_idx, question_idx] = answer_match

                    # Clean-up
                    del processed_images
                    del generated_ids
                    del generated_answers
                    del answer_match

                    gc.collect()
                    torch.cuda.empty_cache()
        
            weighted_scores = torch.sum(scores * question_weights.unsqueeze(0), dim=1)
            accepted_indices = torch.nonzero(weighted_scores >= threshold).squeeze(dim=1).tolist()

        # Format output
        output = {}
        for idx in range(num_images):
            key = idx_to_name[idx] if idx_to_name else idx
            output[key] = {
                'results': list(map(int, scores[idx].tolist())),
                'score': round(weighted_scores[idx].item(), 2)
            }
            if idx in accepted_indices:
                output[key]['accepted'] = True
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