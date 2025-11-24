class Skills:
    COUNTING = "counting"
    COLOR = "color"
    SPATIAL = "spatial_relations"
    SIZE = "size_relations"
    EMOTION = "emotion"
    TEXT = "text"


    @staticmethod
    def from_string(s):
        s = s.lower()
        mapping = {
            "spatial_relations": Skills.SPATIAL,
            "color": Skills.COLOR,
            "counting": Skills.COUNTING,
            "size_relations": Skills.SIZE,
            "emotion": Skills.EMOTION,
        }
        return mapping[s]


class RobustnessTests:
    TYPOS = "typos"
    CONSISTENCY = "consistency"
