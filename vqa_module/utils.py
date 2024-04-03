import io
from PIL import Image
from typing import List

def preprocess_image(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    return image

def process_concatenated_questions(questions:List[str]) -> List[str]:
    questions_str = questions[0]
    questions_lis = questions_str.split('?,')

    questions = [question.strip() + "?" for question in questions_lis if question.strip()]
    questions[-1] = questions[-1][:-1]

    return questions

def process_concatenated_expected_answers(expected_answers:List[str]) -> List[List[str]]:
    expected_answers_str = expected_answers[0]
    expected_answers_lis = expected_answers_str.split(',')

    expected_answers = [answer.strip().split() for answer in expected_answers_lis if answer.strip()]

    return expected_answers