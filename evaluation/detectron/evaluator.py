from .object_detector import Detectron2ObjectDetector
from evaluation.base_evaluator import BaseEvaluator
import os
from utils.skills import Skills

class DetectronEvaluator(BaseEvaluator):
    def __init__(self):
        self.object_detector = Detectron2ObjectDetector()
        self.cache_dir = "cache"

    def evaluate(self, images_path, synthetic_prompts, images_metadata):
        self.object_detector.run_on_images([x for x in os.listdir(images_path)], out_dir=self.cache_dir, save=False)
        return super().evaluate(images_path, synthetic_prompts, images_metadata)

    def evaluate_image(self, image_path, synthetic_prompt, image_metadata):
        skill_set = synthetic_prompt['skill']
        result = []
        if Skills.COUNTING in skill_set:
            result.append(self.evaluate_counting(image_path, image_metadata))
        return result

    def evaluate_counting(self, image_path, image_metadata):
        pass
