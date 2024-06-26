import time
from typing import List, Dict, Union
from fastapi import FastAPI, UploadFile, Depends, Form
from fastapi.responses import RedirectResponse
from fastapi.exceptions import HTTPException, RequestValidationError
from pydantic import BaseModel
from loguru import logger

from .model.vqa_module import VQAEngine
from .utils import preprocess_image, process_concatenated_questions, process_concatenated_expected_answers

logger.add("app.log", rotation="500 MB", level="DEBUG")
open("app.log", "w").close()
app = FastAPI()

class Request(BaseModel):
    model_name: str = Form(..., title="Model Name", description="Name of the model to use for processing.")
    images: List[UploadFile] = Form(..., title="Images", description="List of images to process.")
    questions: List[str] = Form(..., title="Questions", description="List of questions to ask.")
    expected_answers: List[str] = Form(..., title="Expected Answers", description="List of expected answers for each question. Possible answers should be separated by a space.")
    use_confidence: bool = Form(..., title="Use Confidence", description="Whether to use confidence scores for answers.")
    question_weights: Union[str, None] = Form(None, title="Question Weights", description="List of weights for each question. Weights should be separated by a space.")
    threshold: Union[float, None] = Form(None, title="Threshold", description="Threshold for accepting images.")

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse("/docs")

@app.post("/process")
async def process_images(request: Request = Depends()) -> Dict:
    try:
        start = time.time()
        
        # Process request data
        questions = process_concatenated_questions(request.questions) if request.questions[0].count('?') > 1 else request.questions
        expected_answers = process_concatenated_expected_answers(request.expected_answers) if ',' in request.expected_answers[0] else [x.split() for x in request.expected_answers]
        question_weights = [float(x) for x in request.question_weights.split()] if request.question_weights else None
        threshold = request.threshold if request.threshold else None
        use_confidence = request.use_confidence

        # Preprocess images
        logger.info("Processing images...")
        processed_images = []
        idx_to_name = {}
        for idx, image in enumerate(request.images):
            idx_to_name[idx] = image.filename
            image = preprocess_image(await image.read())
            processed_images.append(image)
        logger.info("Images processed successfully.")

        # Send data to VQA engine for processing
        logger.info("Sending data to VQA engine for processing...")
        engine = VQAEngine(model_name=request.model_name, device='cuda')
        output = engine(
            questions=questions,
            images=processed_images,
            expected_answers=expected_answers,
            question_weights=question_weights,
            threshold=threshold,
            idx_to_name=idx_to_name,
            use_confidence=use_confidence
        )
        logger.info(f"Data processed by VQA engine. Out of {len(processed_images)} images, {len(output['accepted_images'])} images were accepted.")
        time_taken = time.time() - start
        output['time_taken'] = round(time_taken, 2)
        logger.info(f"Time taken: {round(time_taken, 2)} seconds.")
        return output

    except RequestValidationError as e:
        logger.error(f"Validation error: {e}.")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("An unexpected error occurred.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred") from e