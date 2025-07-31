#!/usr/bin/env python3
"""
Advanced Custom Metrics Example

This example demonstrates sophisticated metric development patterns
including composite metrics, weighted scoring, and domain-specific evaluation.

Usage:
    python advanced_usage/01_custom_metrics.py
"""

import sys
import os
import re
import math
from typing import Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Add project root to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)

from sde_harness.core import Oracle, Generation, Workflow, Prompt


@dataclass
class MetricResult:
    """Structured result from a metric evaluation"""

    score: float
    details: Dict[str, Any]
    confidence: float = 1.0
    explanation: str = ""


class AdvancedMetric(ABC):
    """Abstract base class for advanced metrics"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        """Evaluate the metric and return structured result"""
        pass

    def __call__(self, prediction: str, reference: str, **kwargs) -> float:
        """Oracle-compatible interface"""
        result = self.evaluate(prediction, reference, **kwargs)
        return result.score * self.weight


class SemanticCoherenceMetric(AdvancedMetric):
    """Evaluates semantic coherence and logical flow"""

    def __init__(self, name: str = "semantic_coherence", weight: float = 1.0):
        super().__init__(name, weight)
        self.transition_words = {
            "addition": ["furthermore", "moreover", "additionally", "also", "besides"],
            "contrast": [
                "however",
                "nevertheless",
                "nonetheless",
                "conversely",
                "on the other hand",
            ],
            "cause": ["therefore", "consequently", "thus", "hence", "as a result"],
            "sequence": ["first", "second", "then", "next", "finally", "subsequently"],
        }

    def evaluate(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        """Evaluate semantic coherence"""
        if not prediction.strip():
            return MetricResult(0.0, {"error": "Empty prediction"})

        # Analyze sentence structure
        sentences = [s.strip() for s in prediction.split(".") if s.strip()]
        if len(sentences) < 2:
            return MetricResult(
                0.5,
                {
                    "sentences": len(sentences),
                    "note": "Too few sentences for coherence analysis",
                },
            )

        # Check for transition words
        transitions_found = 0
        transition_types = set()

        text_lower = prediction.lower()
        for category, words in self.transition_words.items():
            for word in words:
                if word in text_lower:
                    transitions_found += 1
                    transition_types.add(category)

        transition_score = min(1.0, transitions_found / max(1, len(sentences) - 1))

        # Check sentence length variation (good coherence has varied sentence lengths)
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        length_variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(
            sentence_lengths
        )
        length_score = min(1.0, length_variance / 50)  # Normalize variance

        # Check for repetitive patterns (bad coherence)
        words = prediction.lower().split()
        unique_words = len(set(words))
        repetition_score = unique_words / len(words) if words else 0

        # Combine scores
        coherence_score = (
            transition_score * 0.4 + length_score * 0.3 + repetition_score * 0.3
        )

        details = {
            "transitions_found": transitions_found,
            "transition_types": list(transition_types),
            "sentence_count": len(sentences),
            "avg_sentence_length": round(avg_length, 1),
            "length_variance": round(length_variance, 2),
            "word_repetition_ratio": round(repetition_score, 3),
        }

        explanation = f"Coherence analysis: {transitions_found} transitions, {len(sentences)} sentences, {round(repetition_score, 2)} word diversity"

        return MetricResult(
            score=coherence_score,
            details=details,
            confidence=0.8,  # Semantic analysis has inherent uncertainty
            explanation=explanation,
        )


class TechnicalAccuracyMetric(AdvancedMetric):
    """Evaluates technical accuracy in scientific/technical content"""

    def __init__(
        self,
        domain: str = "general",
        name: str = "technical_accuracy",
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        self.domain = domain
        self.technical_vocabularies = {
            "general": [
                "algorithm",
                "data",
                "system",
                "process",
                "method",
                "analysis",
                "model",
                "framework",
            ],
            "ml": [
                "machine learning",
                "neural network",
                "training",
                "validation",
                "overfitting",
                "feature",
                "model",
                "dataset",
            ],
            "chemistry": [
                "molecule",
                "atom",
                "bond",
                "reaction",
                "compound",
                "element",
                "ion",
                "catalyst",
            ],
            "physics": [
                "energy",
                "force",
                "momentum",
                "particle",
                "wave",
                "quantum",
                "relativity",
                "field",
            ],
        }

    def evaluate(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        """Evaluate technical accuracy"""
        domain_vocab = self.technical_vocabularies.get(
            self.domain, self.technical_vocabularies["general"]
        )

        prediction_lower = prediction.lower()

        # Count technical terms
        tech_terms_found = sum(1 for term in domain_vocab if term in prediction_lower)
        tech_term_ratio = tech_terms_found / len(domain_vocab)

        # Check for common technical patterns
        has_definitions = bool(
            re.search(
                r"\b(is defined as|refers to|means that|is the process of)\b",
                prediction_lower,
            )
        )
        has_examples = bool(
            re.search(r"\b(for example|such as|including|like)\b", prediction_lower)
        )
        has_quantification = bool(
            re.search(
                r"\b(\d+%|\d+\.\d+|approximately|roughly|about \d+)\b", prediction
            )
        )

        # Check for technical structure
        has_enumeration = bool(
            re.search(r"\b(first|second|third|1\.|2\.|3\.)\b", prediction_lower)
        )
        has_causation = bool(
            re.search(
                r"\b(because|due to|results in|leads to|causes)\b", prediction_lower
            )
        )

        # Calculate technical accuracy score
        components = {
            "technical_vocabulary": tech_term_ratio * 0.3,
            "definitions": 0.15 if has_definitions else 0,
            "examples": 0.15 if has_examples else 0,
            "quantification": 0.15 if has_quantification else 0,
            "structure": 0.1 if has_enumeration else 0,
            "causation": 0.15 if has_causation else 0,
        }

        accuracy_score = sum(components.values())

        details = {
            "domain": self.domain,
            "technical_terms_found": tech_terms_found,
            "technical_term_ratio": round(tech_term_ratio, 3),
            "has_definitions": has_definitions,
            "has_examples": has_examples,
            "has_quantification": has_quantification,
            "has_enumeration": has_enumeration,
            "has_causation": has_causation,
            "score_components": {k: round(v, 3) for k, v in components.items()},
        }

        explanation = f"Technical accuracy for {self.domain}: {tech_terms_found}/{len(domain_vocab)} terms, structure score: {round(sum(components.values()) - components['technical_vocabulary'], 2)}"

        return MetricResult(
            score=accuracy_score,
            details=details,
            confidence=0.9,
            explanation=explanation,
        )


class CompositionalCreativityMetric(AdvancedMetric):
    """Evaluates creativity through novel combinations and insights"""

    def __init__(self, name: str = "creativity", weight: float = 1.0):
        super().__init__(name, weight)
        self.common_phrases = [
            "in conclusion",
            "it is important",
            "on the other hand",
            "for example",
            "in summary",
            "as we can see",
            "it should be noted",
            "in general",
        ]

    def evaluate(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        """Evaluate compositional creativity"""
        if not prediction.strip():
            return MetricResult(0.0, {"error": "Empty prediction"})

        # Novelty indicators
        word_combinations = self._extract_word_combinations(prediction)
        novel_combinations = self._assess_novelty(word_combinations)

        # Metaphor and analogy detection
        metaphor_score = self._detect_metaphors(prediction)

        # Cliche avoidance
        cliche_penalty = self._calculate_cliche_penalty(prediction)

        # Conceptual bridging (connecting different domains)
        bridging_score = self._assess_conceptual_bridging(prediction)

        # Unexpected insights (surprising connections)
        insight_score = self._detect_insights(prediction)

        # Combine creativity components
        creativity_components = {
            "novelty": novel_combinations * 0.25,
            "metaphor": metaphor_score * 0.2,
            "cliche_avoidance": (1.0 - cliche_penalty) * 0.15,
            "conceptual_bridging": bridging_score * 0.2,
            "insights": insight_score * 0.2,
        }

        creativity_score = sum(creativity_components.values())

        details = {
            "novel_word_combinations": len(word_combinations),
            "metaphor_indicators": metaphor_score > 0,
            "cliche_penalty": round(cliche_penalty, 3),
            "conceptual_bridging": bridging_score > 0,
            "insight_indicators": insight_score > 0,
            "creativity_components": {
                k: round(v, 3) for k, v in creativity_components.items()
            },
        }

        explanation = f"Creativity analysis: {len(word_combinations)} novel combinations, metaphor score: {round(metaphor_score, 2)}, insight score: {round(insight_score, 2)}"

        return MetricResult(
            score=creativity_score,
            details=details,
            confidence=0.7,  # Creativity assessment is inherently subjective
            explanation=explanation,
        )

    def _extract_word_combinations(self, text: str) -> List[Tuple[str, str]]:
        """Extract novel word combinations"""
        words = re.findall(r"\b\w+\b", text.lower())
        combinations = []

        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:  # Skip short words
                combinations.append((words[i], words[i + 1]))

        return combinations

    def _assess_novelty(self, combinations: List[Tuple[str, str]]) -> float:
        """Assess novelty of word combinations (simplified)"""
        # In practice, this would use embeddings or language models
        # For now, we use heuristics
        return min(1.0, len(set(combinations)) / max(1, len(combinations)))

    def _detect_metaphors(self, text: str) -> float:
        """Detect metaphorical language"""
        metaphor_indicators = [
            "like",
            "as if",
            "reminds me of",
            "similar to",
            "acts as",
            "behaves like",
        ]
        text_lower = text.lower()

        metaphors_found = sum(
            1 for indicator in metaphor_indicators if indicator in text_lower
        )
        return min(1.0, metaphors_found / 3)  # Normalize

    def _calculate_cliche_penalty(self, text: str) -> float:
        """Calculate penalty for cliched expressions"""
        text_lower = text.lower()
        cliches_found = sum(1 for phrase in self.common_phrases if phrase in text_lower)

        return min(1.0, cliches_found / 5)  # Penalty increases with cliches

    def _assess_conceptual_bridging(self, text: str) -> float:
        """Assess connections between different conceptual domains"""
        # Simplified: look for words from different domains in close proximity
        domain_words = {
            "tech": ["algorithm", "data", "computer", "digital", "software"],
            "nature": ["organic", "natural", "biological", "ecosystem", "evolution"],
            "art": ["creative", "aesthetic", "beautiful", "artistic", "design"],
            "social": ["community", "society", "human", "cultural", "social"],
        }

        text_lower = text.lower()
        domains_present = []

        for domain, words in domain_words.items():
            if any(word in text_lower for word in words):
                domains_present.append(domain)

        return (
            min(1.0, (len(domains_present) - 1) / 3)
            if len(domains_present) > 1
            else 0.0
        )

    def _detect_insights(self, text: str) -> float:
        """Detect potentially insightful connections"""
        insight_indicators = [
            "surprisingly",
            "unexpectedly",
            "interestingly",
            "remarkably",
            "what if",
            "imagine if",
            "consider that",
            "it turns out",
        ]

        text_lower = text.lower()
        insights_found = sum(
            1 for indicator in insight_indicators if indicator in text_lower
        )

        return min(1.0, insights_found / 2)


class MetricComposer:
    """Composes multiple metrics into weighted combinations"""

    def __init__(self, metrics: List[AdvancedMetric], weights: Dict[str, float] = None):
        self.metrics = metrics
        self.weights = weights or {metric.name: 1.0 for metric in metrics}

    def evaluate_comprehensive(
        self, prediction: str, reference: str, **kwargs
    ) -> Dict[str, Any]:
        """Evaluate using all metrics and provide comprehensive analysis"""
        results = {}
        detailed_results = {}
        total_weight = 0
        weighted_score = 0

        for metric in self.metrics:
            try:
                result = metric.evaluate(prediction, reference, **kwargs)
                weight = self.weights.get(metric.name, 1.0)

                results[metric.name] = result.score
                detailed_results[metric.name] = {
                    "score": result.score,
                    "weighted_score": result.score * weight,
                    "details": result.details,
                    "confidence": result.confidence,
                    "explanation": result.explanation,
                }

                weighted_score += result.score * weight
                total_weight += weight

            except Exception as e:
                results[metric.name] = 0.0
                detailed_results[metric.name] = {"score": 0.0, "error": str(e)}

        composite_score = weighted_score / total_weight if total_weight > 0 else 0.0

        return {
            "composite_score": composite_score,
            "individual_scores": results,
            "detailed_analysis": detailed_results,
            "weights_used": self.weights,
        }


def example_advanced_metrics():
    """Demonstrate advanced metric usage"""
    print("🔷 Advanced Custom Metrics Example")
    print("=" * 60)

    # Create advanced metrics
    coherence_metric = SemanticCoherenceMetric(weight=0.3)
    accuracy_metric = TechnicalAccuracyMetric(domain="ml", weight=0.4)
    creativity_metric = CompositionalCreativityMetric(weight=0.3)

    # Create metric composer
    composer = MetricComposer([coherence_metric, accuracy_metric, creativity_metric])

    # Test texts
    test_cases = [
        {
            "name": "Technical ML Explanation",
            "text": """Machine learning algorithms learn patterns from training data to make predictions on new data. 
            For example, neural networks use backpropagation to adjust weights during training. 
            This process is surprisingly similar to how humans learn from experience, creating unexpected connections 
            between artificial and biological intelligence systems.""",
            "reference": "machine learning explanation",
        },
        {
            "name": "Creative Scientific Writing",
            "text": """Data flows through neural networks like water through a complex river system. 
            Each layer acts as a filter, transforming information in ways that mirror how our brains 
            process sensory input. This biological inspiration leads to fascinating innovations in AI.""",
            "reference": "creative explanation",
        },
        {
            "name": "Poor Quality Text",
            "text": """Machine learning is good. It is very good. Data is important. Very important. 
            In conclusion, machine learning with data is good for making predictions.""",
            "reference": "low quality explanation",
        },
    ]

    for case in test_cases:
        print(f"\n📝 Analyzing: {case['name']}")
        print("-" * 40)
        print(f"Text: {case['text'][:100]}...")
        print()

        # Get comprehensive evaluation
        evaluation = composer.evaluate_comprehensive(case["text"], case["reference"])

        print(f"🎯 Composite Score: {evaluation['composite_score']:.3f}")
        print()

        print("Individual Metrics:")
        for metric_name, details in evaluation["detailed_analysis"].items():
            score = details["score"]
            confidence = details.get("confidence", 1.0)
            explanation = details.get("explanation", "No explanation")

            print(f"  {metric_name}: {score:.3f} (confidence: {confidence:.2f})")
            print(f"    {explanation}")

        print()


def example_oracle_integration():
    """Show how to integrate advanced metrics with Oracle"""
    print("🔷 Oracle Integration with Advanced Metrics")
    print("=" * 60)

    oracle = Oracle()

    # Create and register advanced metrics
    coherence_metric = SemanticCoherenceMetric()
    accuracy_metric = TechnicalAccuracyMetric(domain="general")
    creativity_metric = CompositionalCreativityMetric()

    # Register with Oracle (Oracle expects functions that return floats)
    oracle.register_metric("coherence", coherence_metric)
    oracle.register_metric("accuracy", accuracy_metric)
    oracle.register_metric("creativity", creativity_metric)

    # Create a multi-round metric that tracks improvement in creativity
    def creativity_improvement(
        history: Dict, reference: Any, current_iteration: int, **kwargs
    ) -> float:
        """Track creativity improvement over iterations"""
        if "scores" not in history or len(history["scores"]) < 2:
            return 0.0

        creativity_scores = [
            score.get("creativity", 0.0) for score in history["scores"]
        ]
        if len(creativity_scores) < 2:
            return 0.0

        # Calculate improvement trend
        recent_scores = (
            creativity_scores[-3:] if len(creativity_scores) >= 3 else creativity_scores
        )

        if len(recent_scores) < 2:
            return 0.0

        # Linear trend (positive means improving)
        x = list(range(len(recent_scores)))
        y = recent_scores

        # Simple linear regression slope
        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (
            n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2
        )

        return max(0.0, slope)  # Return 0 if declining, positive value if improving

    oracle.register_multi_round_metric("creativity_improvement", creativity_improvement)

    # Test evaluation
    test_text = """Neural networks are fascinating computational systems that mirror biological processes. 
    Like rivers carving through landscape, data flows through layers, each transformation creating 
    new patterns and insights. This biological inspiration surprisingly leads to breakthrough innovations."""

    # Single evaluation
    print("Single Evaluation:")
    result = oracle.compute(test_text, "reference")
    for metric, score in result.items():
        print(f"  {metric}: {score:.3f}")

    print()

    # Multi-round evaluation simulation
    print("Multi-round Evaluation Simulation:")
    history = {"outputs": [], "scores": []}

    iterations = [
        "Machine learning is good for data analysis.",
        "Neural networks process data like biological systems, creating patterns.",
        "Neural networks flow like rivers through data landscapes, surprisingly creating biological-inspired breakthroughs.",
    ]

    for i, text in enumerate(iterations, 1):
        print(f"\nIteration {i}:")
        print(f"  Text: {text}")

        history["outputs"].append(text)

        # Compute scores including multi-round metrics
        if i == 1:
            scores = oracle.compute(text, "reference")
        else:
            scores = oracle.compute_with_history(text, "reference", history, i)

        history["scores"].append(scores)

        print(f"  Scores: {scores}")


def main():
    """Run advanced metrics examples"""
    print("🚀 SDE-Harness Advanced Custom Metrics")
    print("=" * 70)
    print()

    example_advanced_metrics()
    print("\n" + "=" * 70 + "\n")
    example_oracle_integration()

    print("\n" + "=" * 70)
    print("✅ Advanced metrics examples completed!")
    print("\n💡 Key Takeaways:")
    print("- Advanced metrics provide detailed, structured evaluation")
    print("- Metric composition enables multi-dimensional assessment")
    print("- Integration with Oracle maintains framework compatibility")
    print("- Multi-round metrics can track improvement trends")

    print("\n🎯 Next Steps:")
    print("- Create domain-specific metrics for your use case")
    print("- Experiment with different weighting schemes")
    print("- Try advanced_usage/02_multi_round_workflows.py")


if __name__ == "__main__":
    main()
