# T2I Evaluation Framework

A comprehensive framework for evaluating text-to-image generation models through systematic prompt generation and multi-modal assessment.

## Overview

The T2I Evaluation Framework enables researchers to:

1. **Generate Synthetic Prompts**: Create structured prompts that test specific visual skills including counting, color recognition, spatial relationships, size perception, and emotional expression.

2. **Multi-Modal Evaluation**: Assess generated images using multiple evaluation methods:
   - **Detectron2-based Object Detection**: Automatic detection and counting of objects, colors, and spatial relationships
   - **LLM Judge Evaluation**: Vision-language model assessment for complex visual understanding
   - **Manual Evaluation**: Human evaluation interface for ground truth validation

3. **Skill-Based Testing**: Systematically evaluate models across different difficulty levels (easy, medium, hard) and various visual understanding capabilities.

## Features

### Supported Skills
- **Counting**: Verify correct quantities of objects in images
- **Color**: Assess color accuracy and consistency
- **Spatial**: Evaluate spatial relationships and positioning
- **Size**: Test relative size understanding
- **Emotion**: Analyze emotional expression in generated content
- **Text**: Text rendering and integration (planned)

### Evaluation Methods
- **Detectron2 Object Detection**: Automated object detection and analysis
- **LLM Vision Judge**: GPT-4 Vision and similar models for semantic evaluation
- **Manual Evaluation**: Human assessment interface

### Robustness Testing
- **Typo Handling**: Test model resilience to text errors
- **Consistency**: Evaluate output consistency across similar prompts

## Installation

### Requirements

First, install the basic dependencies:

```bash
pip install -r requirements.txt
```

### Detectron2 Installation

Install Detectron2 for object detection evaluation:

```bash
pip install --no-build-isolation --no-deps git+https://github.com/facebookresearch/detectron2.git
```

For detailed Detectron2 installation instructions and troubleshooting, visit the [official Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

### Environment Setup

Create a `.env` file in the project root with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Quick Start

### 1. Generate Synthetic Prompts

Create prompts for testing specific visual skills:

```bash
python main.py generate --config config/prompt_generation/prompt_generation.yaml
```

This will generate structured prompts based on the configuration and save them to `outputs/prompts/`.

### 2. Run Evaluation

Evaluate generated images using the framework:

```bash
python main.py evaluate --config config/evaluation/evaluation_config.yaml
```

## Configuration

### Prompt Generation Configuration

Configure prompt generation in `config/prompt_generation/prompt_generation.yaml`:

```yaml
experiment:
  name: "your_experiment_name"

prompt_generation:
  metadata_config:
    name: "coco"  # Dataset source for objects
  skills:
    - level: "easy"
      skills: ["counting", "color"]
    - level: "medium"
      skills: ["spatial"]
    - level: "hard"
      skills: ["color", "counting", "spatial"]
  llm_model: "gpt4"
  samples_per_skill: 10

output:
  file: "outputs/prompts/your_experiment_prompts.json"
```

### Evaluation Configuration

Configure evaluation in `config/evaluation/evaluation_config.yaml`:

```yaml
experiment:
  name: "your_evaluation_experiment"

input:
  synthetic_prompts: "your_experiment_prompts.json"
  images_directory: "data/images/"
  image_metadata: "data/images_metadata/images_metadata.json"

evaluators:
  detectron:
    model: "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    device: "cuda"  # or "cpu"
    confidence_threshold: 0.5

  llm_judge:
    model: "gpt4_vision"
    temperature: 0.1

output:
  results_file: "outputs/results/your_evaluation_results.json"
```

## Project Structure

```
T2I_new/
├── main.py                     # Main CLI entry point
├── requirements.txt            # Python dependencies
├── config/                     # Configuration files
│   ├── prompt_generation/      # Prompt generation configs
│   └── evaluation/             # Evaluation configs
├── data/                       # Input data and metadata
│   ├── COCO/                   # COCO dataset objects
│   ├── images/                 # Generated images to evaluate
│   └── images_metadata/        # Image metadata
├── evaluation/                 # Evaluation modules
│   ├── detectron/              # Detectron2-based evaluation
│   └── llm_judge/              # LLM-based evaluation
├── experiments/                # Experiment orchestration
├── llm_interfaces/             # LLM API interfaces
├── metadata/                   # Dataset metadata handlers
├── prompt_generation/          # Synthetic prompt generation
├── utils/                      # Utility functions and skill definitions
└── outputs/                    # Generated prompts and results
    ├── prompts/                # Generated prompt files
    └── results/                # Evaluation results
```

## Usage Examples

### Basic Counting and Color Evaluation

1. Generate prompts that test counting and color recognition:
```bash
python main.py generate --config config/prompt_generation/prompt_generation.yaml
```

2. Place your generated images in `data/images/`

3. Run evaluation:
```bash
python main.py evaluate --config config/evaluation/evaluation_config.yaml
```

### Custom Skill Testing

Create custom configurations to test specific skills or combinations:

```yaml
skills:
  - level: "hard"
    skills: ["counting", "color", "spatial"]
    samples_per_skill: 20
```

## Extending the Framework

### Adding New Skills

1. Add skill definition to `utils/skills.py`:
```python
class Skills:
    YOUR_SKILL = "your_skill"
```

2. Implement skill logic in `prompt_generation/prompt_generation.py`

3. Add evaluation logic in appropriate evaluator modules

### Adding New Evaluators

1. Create evaluator class inheriting from `BaseEvaluator`
2. Implement the `evaluate_image` method
3. Add configuration support in evaluation configs

## Dependencies

- **PyTorch**: Deep learning framework
- **Detectron2**: Facebook's object detection library
- **OpenAI**: GPT API access for LLM evaluation
- **OpenCV**: Image processing
- **PyTorch Vision**: Computer vision utilities
- **COCO Tools**: COCO dataset utilities

## Contributing

This framework is designed for research in text-to-image model evaluation. Contributions are welcome for:

- Additional evaluation skills and methods
- New evaluator implementations
- Robustness testing improvements
- Documentation and examples

## License

[Specify your license here]

## Citation

If you use this framework in your research, please cite:

```bibtex
[Add citation information here]
```
