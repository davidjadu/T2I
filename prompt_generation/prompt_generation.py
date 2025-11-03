import numpy as np
from utils.skills import Skills,RobustnessTests

class MetaPromptGeneration:
    def __init__(self, metadata):
        self.levels = ["easy", "medium", "hard"]
        self.data_type = "coco"
        self.metadata = metadata
        self.scene = None

    def _setup_scene(self, skills, level):

        k=0
        match level:
            case "easy": k=2
            case "medium": k=3
            case "hard": k=4
        scene = {"objects":[{"object":x} for x in self.metadata.get_rnd_objects(k=k)]}

        if Skills.COUNTING in skills:
            scene = self.apply_counting(level, scene)
        if Skills.COLOR in skills:
            scene = self.apply_color(level, scene)
        if Skills.SPATIAL in skills:
            scene = self.apply_spatial(level, scene)
        if Skills.SIZE in skills:
            scene = self.apply_size(level, scene)
        if Skills.EMOTION in skills:
            scene = self.apply_emotion(level, scene)

        self.scene = scene

        return scene

    def generate_meta_prompt(self, skills, level):
        scene = self._setup_scene(skills, level)
        template = "You are generating a prompt for an image generation model. The scene contains the following objects"

        if Skills.COLOR in skills:
            template += " of the given colors"
        if Skills.COUNTING in skills:
            template += " with the given quantities"

        template += ": "
        for obj in scene["objects"]:
            template += f"{obj}, "
        template = template[:-2] + ".\n"

        if Skills.EMOTION in skills:
            for emotion in scene['emotion']:
                template += f" The scene should have a {emotion} person.\n"

        if Skills.SPATIAL in skills:
            for o1, relationship, o2 in scene['spatial_relations']:
                template += f" The {o1}(s) must be {relationship} the {o2}(s).\n"

        if Skills.SIZE in skills:
            for o1, relationship, o2 in scene['size_relations']:
                template += f" The {o1}(s) must be {relationship} the {o2}(s).\n"

        print("Generated Scene:", scene)
        return template

    def apply_counting(self, level, scene):
        low, high = 0, 0
        k = len(scene['objects'])
        match level:
            case "easy": low = 1; high = 2
            case "medium": low = 2; high = 3
            case "hard": low = 4; high = 5

        # Note that randint high is exclusive
        for i in range(k):
            scene['objects'][i]['count'] = np.random.randint(low=low, high=high+1)
        return scene

    def apply_color(self, _, scene):
        k = len(scene['objects'])
        colors = self.metadata.get_rnd_colors(k=k)
        for i in range(k):
            scene['objects'][i]['color'] = colors[i]
        return scene

    def apply_spatial(self, _, scene):
        raise NotImplementedError

    def apply_size(self, _, scene):
        raise NotImplementedError

    def apply_emotion(self, _, scene):
        raise NotImplementedError

class PromptGeneration:
    def __init__(self,llm_interface):
        self.llm_interface = llm_interface

    def generate_prompt(self, meta_prompt, robustness_test=[], k=1):
        """
        Generate prompts using the LLM interface with optional robustness tests.
        Args:
            meta_prompt (str): The meta prompt to be used for generation.
            robustness_test (list): List of robustness tests to apply.
            k (int): Number of prompts to generate.
        Returns:
            list: Generated prompts.
            In the case of robustness, the first prompt is the original, and the rest are modified versions.
        """
        if not robustness_test:
            return [self.llm_interface.generate(meta_prompt) for _ in range(k)]

        elif k==1:
            raise ValueError("k must be greater than 1 when robustness tests are applied.")

        prompt = self.llm_interface.generate(meta_prompt)
        prompts = [prompt]

        template = ""

        if RobustnessTests.TYPOS in robustness_test and RobustnessTests.CONSISTENCY in robustness_test:
            template = "Paraphrase the following prompt and introduce some typos while keeping its meaning intact:\n"

        elif RobustnessTests.TYPOS in robustness_test:
            template = "Introduce some typos in the following prompt while keeping its meaning intact:\n"

        elif RobustnessTests.CONSISTENCY in robustness_test:
            template = "Paraphrase the following prompt while keeping its meaning intact:\n"

        prompts.extend([self.llm_interface.generate(template + prompt) for _ in range(k-1)])

        return prompts

if __name__ == "__main__":
    # Mock metadata class for testing
    class MockMetadata:
        def get_rnd_objects(self, k):
            objects = ["cat", "dog", "car", "tree", "house", "ball", "book", "chair"]
            return np.random.choice(objects, k, replace=False).tolist()

        def get_rnd_colors(self, k):
            colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]
            return np.random.choice(colors, k, replace=False).tolist()

    # Create instance with mock metadata
    metadata = MockMetadata()
    generator = MetaPromptGeneration(metadata)

    # Test different skill combinations and levels
    test_cases = [
        ([Skills.COUNTING], "easy"),
        ([Skills.COLOR, Skills.COUNTING], "medium"),
        ([Skills.COLOR, Skills.COUNTING, Skills.SPATIAL], "hard"),
    ]

    for skills, level in test_cases:
        print(f"\n--- Testing {skills} at {level} level ---")
        prompt = generator.generate_prompt(skills, level)
        print(prompt)
