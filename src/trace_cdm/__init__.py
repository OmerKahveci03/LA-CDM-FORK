"""TRACE-CDM: evidence auditing and an adaptive Evidence Critic for LA-CDM.

The package hooks into ``src.environment.environment.Environment`` at the
decision boundary (after the Hypothesis Agent, before the Decision Agent). It
only imports torch/transformers when a local critic model is loaded, so the
audit, gate, controller and comparison code run on CPU without the GPU stack.
"""
