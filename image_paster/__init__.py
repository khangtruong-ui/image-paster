"""Image Paster: Diffusion-Free Semantic Image Generation using C++ Scene DSL and SAM 3."""

__version__ = "0.1.0"

from image_paster.dsl import (
    parse_dsl,
    SceneDSLParser,
    DSLSyntaxError,
    DSLValidator,
    DSLValidationError,
    SceneIR,
    ObjectIR,
)
from image_paster.llm.planner import (
    BaseScenePlanner,
    RuleBasedPlanner,
    LLMScenePlanner,
)
from image_paster.retrieval.base import (
    ImageRetriever,
    ImageCandidate,
    RetrievalResult,
)
from image_paster.retrieval.duckduckgo import DuckDuckGoRetriever
from image_paster.retrieval.mock import MockRetriever
from image_paster.segmentation.base import (
    Segmenter,
    SegmentationResult,
)
from image_paster.segmentation.sam3 import SAM3Segmenter
from image_paster.scene.layout import (
    SemanticLayoutSolver,
    LayoutPlan,
    ObjectLayout,
)
from image_paster.rendering.compositor import (
    SceneCompositor,
    CompositeResult,
)
from image_paster.verification.verifier import (
    SceneVerifier,
    SemanticVisualVerifier,
    VerificationResult,
)
from image_paster.pipeline.generator import (
    SemanticImageGenerator,
    GenerationResult,
)

__all__ = [
    "parse_dsl",
    "SceneDSLParser",
    "DSLSyntaxError",
    "DSLValidator",
    "DSLValidationError",
    "SceneIR",
    "ObjectIR",
    "BaseScenePlanner",
    "RuleBasedPlanner",
    "LLMScenePlanner",
    "ImageRetriever",
    "ImageCandidate",
    "RetrievalResult",
    "DuckDuckGoRetriever",
    "MockRetriever",
    "Segmenter",
    "SegmentationResult",
    "SAM3Segmenter",
    "SemanticLayoutSolver",
    "LayoutPlan",
    "ObjectLayout",
    "SceneCompositor",
    "CompositeResult",
    "SceneVerifier",
    "SemanticVisualVerifier",
    "VerificationResult",
    "SemanticImageGenerator",
    "GenerationResult",
]
