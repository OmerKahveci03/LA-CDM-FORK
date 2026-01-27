# Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning

📄 [**Paper**](https://arxiv.org/abs/2506.13474)

This repository contains the official code for the paper

> **Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning**
>
> David Bani-Harouni, Chantal Pellegrini, Ege Özsoy, Matthias Keicher, Nassir Navab<br>
> Technical University of Munich, Munich Center for Machine Learning 
> 
> to be published at ICLR 2026


## Abstract

Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency.


## Status

🚧 **Coming soon** 🚧

If you’re interested, feel free to ⭐ the repository to stay updated.


## Citation

If you find this work useful, please cite:

```bibtex
@article{bani2025language,
  title={Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning},
  author={Bani-Harouni, David and Pellegrini, Chantal and {\"O}zsoy, Ege and Keicher, Matthias and Navab, Nassir},
  journal={arXiv preprint arXiv:2506.13474},
  year={2025}
}
