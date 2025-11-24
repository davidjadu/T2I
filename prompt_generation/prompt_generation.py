import numpy as np
from utils.skills import Skills,RobustnessTests

#Normalisation
def normalize_skills(skills):
    normalized = []
    for s in skills:
        try:
            normalized.append(Skills.from_string(s))
        except:
            normalized.append(s)
    return normalized

class MetaPromptGeneration:
    def __init__(self, metadata):
        self.levels = ["easy", "medium", "hard"]
        self.data_type = "coco"
        self.metadata = metadata
        self.scene = None

    def _setup_scene(self, skills, level):

        skills = normalize_skills(skills)
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
        skills = normalize_skills(skills)        
        scene = self._setup_scene(skills, level)
        template = "You are generating a prompt for an image generation model. The scene contains the following objects"

        if Skills.COLOR in skills:
            template += " of the given colors"
        if Skills.COUNTING in skills:
            template += " with the given quantities"

        template += ": "

        obj_descriptions = []
        for obj in scene["objects"]:
            desc = obj["object"]

            if "count" in obj:
                desc += f" ({obj['count']})"
            if "color" in obj:
                desc += f" in {obj['color']}"
            if "spatial_relations" in obj:
                desc += f" {obj['spatial_relations']}" 
            if "size_relations" in obj:
                desc += f" {obj['size_relations']}"                
            obj_descriptions.append(desc)
            template += desc + ", "
        template += ", ".join(obj_descriptions) + ".\n"

        # Spatial between objects
        if Skills.SPATIAL in skills and "spatial_relations" in scene:
            for (o1, relationship, o2) in scene["spatial_relations"]:
                template += f" The {o1}(s) must be {relationship} the {o2}(s).\n"

        if Skills.EMOTION in skills:
            for emotion in scene['emotion']:
                template += f" The scene should have a {emotion} person.\n"
        #Size between objects
        if Skills.SIZE in skills and "size_relations" in scene:
            for (o1, relationship, o2) in scene['size_relations']:
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
        objects = scene["objects"]
        k = len(objects)
        if k < 2:
            scene["spatial_relations"] = []
            return scene
        num_relations = min(k - 1, 2)
        relations = []
        used_pairs = set()
        for _ in range(num_relations):
            while True:
                i, j = np.random.choice(k, 2, replace=False)
                if(i,j) not in used_pairs and (j,i) not in used_pairs:
                    used_pairs.add((i,j))
                    break
            obj1 = objects[i]["object"]
            obj2 = objects[j]["object"]
            relation = self.metadata.get_rnd_spatial_relations(k=1)[0]
            relations.append((obj1, relation, obj2))
        scene["spatial_relations"] = relations
        return scene

    def apply_size(self, _, scene):
        objects = scene["objects"]
        k = len(objects)
        if k < 2:
            scene["size_relations"] = []
            return scene
        relations = []
        used_pairs = set()
        num_relations = min(1, k-1)
        for _ in range(num_relations):
            while True:
                i, j = np.random.choice(k, 2, replace=False)
                if(i,j) not in used_pairs: 
                    used_pairs.add((i,j))
                    break
            obj1 = objects[i]["object"]
            obj2 = objects[j]["object"]
            relation = self.metadata.get_rnd_size_relations(k=1)[0]
            relations.append((obj1, relation, obj2))
        scene["size_relations"] = relations
        return scene

    def apply_emotion(self, _, scene):
        raise NotImplementedError

class PromptGeneration:
    def __init__(self, llm_interface):
        self.llm_interface = llm_interface

    def generate_prompt(self, meta_prompt, robustness_test=[], k=1):

        if not robustness_test:
            return [self.llm_interface.generate(meta_prompt) for _ in range(k)]

        if k == 1:
            raise ValueError("k must be > 1 when applying robustness tests.")

        prompt = self.llm_interface.generate(meta_prompt)
        prompts = [prompt]

        # Sélection du style de robustesse
        if RobustnessTests.TYPOS in robustness_test and RobustnessTests.CONSISTENCY in robustness_test:
            template = "Paraphrase this prompt and introduce typos:\n"
        elif RobustnessTests.TYPOS in robustness_test:
            template = "Introduce typos while keeping the meaning intact:\n"
        elif RobustnessTests.CONSISTENCY in robustness_test:
            template = "Paraphrase the prompt while keeping the meaning intact:\n"
        else:
            template = ""

        prompts.extend([self.llm_interface.generate(template + prompt) for _ in range(k - 1)])

        return prompts    