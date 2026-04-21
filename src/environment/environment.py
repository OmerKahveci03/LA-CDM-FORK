import re
import random
from dataclasses import dataclass, field
from trl.data_utils import maybe_apply_chat_template


@dataclass
class EnvironmentOutput:
    """Typed container for the outputs of ``Environment.run()``."""

    completion_ids: list[list[int]]
    completion_mask: list[list[bool]]
    reward_function_kwargs: list[dict]
    ha_prompt_ids: list[list[int]]
    ha_label_ids: list[list[int]]
    ha_conf_cal_prompt_ids: list[list[int]] = field(default_factory=list)
    ha_conf_cal_completion_ids: list[list[int]] = field(default_factory=list)
    ha_hypothesis: str = "unknown"
    ece_calc_triplets: list[tuple] = field(default_factory=list)
    ha_hypotheses_lists: list[list[str]] = field(default_factory=list)
    ha_confidences_lists: list[list[float | None]] = field(default_factory=list)

class Trajectory:
    """State of a single patient episode through the environment.

    Tracks the evolving text prompt, model outputs, requested/provided tests,
    hypotheses, confidence scores, and the final diagnosis.
    """

    def __init__(self, text: str, test_results: dict[str, str], patient_history: str) -> None:
        self.text: str = text
        self.output = None  # latest vLLM RequestOutput
        self.token_mask: list[bool] = []
        self.completed: bool = False
        self.invalid: bool = False
        self.test_results: dict[str, str] = {k.lower(): v for k, v in test_results.items()}
        self.test_results_lower_to_original: dict[str, str] = {k.lower(): k for k, v in test_results.items()}
        self.patient_history: str = patient_history
        self.last_completion: str = ""
        self.diagnosis: str | None = None
        self.initial_prompt_token_ids: list[int] | None = None
        self.completion_ids: list[int] = []
        self.provided_tests: list[str] = []
        self.requested_tests: list[str] = []
        self.request_count: int = 0
        self.hypothesis: list[str] = []
        self.hypothesis_prompt: list[str] = []
        self.confidence: list[float | None] = []
        self.error: bool = False

    def update_output(self, output, tokenizer) -> None:
        self.output = output
        prompt_ids = output.prompt_token_ids
        # Extract raw completion token ids and text
        completion_ids_raw = list(output.outputs[0].token_ids)
        completion_text_raw = output.outputs[0].text

        # Detect whether the action is a diagnosis. In this case we *keep* a
        # trailing eos token (the eos marks the end of the whole dialogue), but
        # for test actions we remove it so that the environment can continue
        # appending text.
        is_diagnosis_action = bool(re.search(r"action:\s*diagnosis", completion_text_raw, flags=re.IGNORECASE))

        # Optionally strip eos for non-diagnosis actions
        if (
            not is_diagnosis_action
            and len(completion_ids_raw) > 0
            and completion_ids_raw[-1] == tokenizer.eos_token_id
        ):
            completion_ids = completion_ids_raw[:-1]
        else:
            completion_ids = completion_ids_raw

        # Remove a trailing "Observation:" stop string from the token ids, if present.
        # We keep any preceding newline tokens because they are reflected in the
        # text that was already saved. Generation stops immediately after the
        # word "Observation:", so no newlines follow it.
        if not is_diagnosis_action:
            obs_token_ids = tokenizer.encode("Observation:", add_special_tokens=False)
            if (
                len(completion_ids) >= len(obs_token_ids)
                and completion_ids[-len(obs_token_ids):] == obs_token_ids
            ):
                completion_ids = completion_ids[:-len(obs_token_ids)]

        # Save the processed completion ids for later use
        self.completion_ids = completion_ids

        # `vLLM` does not include the `<eos>` token in `text`, so we can simply
        # keep the original text. For non-diagnosis actions the eos token was
        # removed from `completion_ids` above, which does not affect the text.
        self.last_completion = completion_text_raw
        len_previous_tokens = len(self.token_mask)
        # For iterations > 0, we only append the token mask
        self.token_mask += [False] * (len(prompt_ids) - len_previous_tokens) + [True] * len(completion_ids)
        self.text += completion_text_raw
        if self.initial_prompt_token_ids is None:
            self.initial_prompt_token_ids = prompt_ids

    def get_test_result(self, test: str) -> str:
        test_result = self.test_results[test]
        self.request_count += 1
        if test not in self.provided_tests and test_result != "not available.\n":
            self.provided_tests.append(test)
        if test not in self.requested_tests:
            self.requested_tests.append(test)
        return test_result

    def complete(self, invalid: bool = False) -> None:
        self.completed = True
        self.invalid = invalid


class Environment:
    """Multi-turn ReAct environment for clinical decision making.

    Manages patient trajectories where the Decision Agent iteratively requests
    tests or gives a diagnosis, optionally guided by a Hypothesis Agent that
    provides hypotheses and confidence scores at each step.
    """

    def __init__(
        self,
        prompt_template: str,
        max_length: int,
        max_steps: int,
        disease_list: list[str],
        test_list: list[str],
        generate_hypothesis: bool = False,
        hypothesis_prompt_template: str | None = None,
        generate_confidence_calibration: bool = False,
        seed: int = 42,
    ) -> None:
        self.prompt_template = prompt_template
        self.generate_hypothesis = generate_hypothesis
        self.hypothesis_prompt_template = hypothesis_prompt_template
        self.max_length = max_length
        self.max_steps = max_steps
        self.possible_actions = ['Test', 'Diagnosis']
        self.disease_list = disease_list
        self.test_list = test_list
        self.ha_prompt_ids = []
        self.ha_label_ids = []
        self.ha_label_format = "Hypothesis: {hypothesis}{eos_token}" if not generate_confidence_calibration else "Hypothesis: {hypothesis}\nConfidence: {confidence}{eos_token}"
        self.generate_confidence_calibration = generate_confidence_calibration
        self._rng = random.Random(seed)

    def construct_prompt(self, inputs: list[dict]) -> list[dict]:
        """Construct chat-formatted prompts for each patient input.

        Args:
            inputs: List of dicts with keys ``patient_id``, ``patient_history``,
                ``label``, ``test_results``.

        Returns:
            The same list with a ``prompt`` key added to each dict.
        """
        for input in inputs:
            patient_history = input['patient_history']
            if self.generate_hypothesis:
                prompt = self.prompt_template.format(patient_history=patient_history, first_hypothesis="{first_hypothesis}")
            else:
                prompt = self.prompt_template.format(patient_history=patient_history)
            prompt = [{"role": "user", "content": prompt}]
            input['prompt'] = prompt
        return inputs
    
    def parse_actions_and_check_validity(self, trajectory: Trajectory) -> tuple[bool, str | None, str | None]:
        """Parse action and action input from the latest completion.

        Args:
            trajectory: The trajectory to parse.

        Returns:
            Tuple of ``(valid, action, action_input)``. When invalid, action
            and action_input are ``None``.
        """
        # 1. Check if the last completion made the text too long.
        if len(trajectory.token_mask) > self.max_length:
            return False, None, None
        
        # 2. Parse Action and Action Input from the last completion. We no longer
        #    expect the model to generate the "Observation:" keyword – that will
        #    be appended by the environment itself. Therefore a single pattern
        #    suffices for both Test and Diagnosis actions.
        pattern = r"Thought: (.*)\n{1,2}Action: (.*)\n{1,2}Action Input: (.*)"

        match = re.search(pattern, trajectory.last_completion)
        if match:
            _, action, action_input = match.groups()
            return True, action.lower(), action_input.lower()
        
        return False, None, None
      
    def run(self, prompts_text: list[str], inputs: list[dict], model, sampling_params, processing_class) -> EnvironmentOutput:
        """Run multi-turn episodes for a GRPO group (multiple rollouts of the same patient).

        Args:
            prompts_text: Chat-templated prompt strings, one per rollout.
            inputs: Patient dicts (duplicated ``num_generations`` times for GRPO).
            model: vLLM ``LLM`` instance for generation.
            sampling_params: vLLM ``SamplingParams``.
            processing_class: HuggingFace tokenizer.

        Returns:
            ``EnvironmentOutput`` with completion ids, masks, reward kwargs,
            HA SFT/conf-cal tensors, and per-trajectory hypothesis/confidence data.
        """
        # Initialize trajectories
        trajectories = [Trajectory(prompt, input['test_results'], input['patient_history']) for prompt, input in zip(prompts_text, inputs)]
        # clear ha_prompt_ids
        self.ha_prompt_ids.clear()

        # Loop while not all trajectories are completed
        step = 0
        while not all(trajectory.completed for trajectory in trajectories):
            active_trajectories = [trajectory for trajectory in trajectories if not trajectory.completed]
            if self.generate_hypothesis:
                active_prompts = self._generate_hypothesis(active_trajectories, model, sampling_params, processing_class)
            else:
                active_prompts = [trajectory.text for trajectory in active_trajectories]
            # Generate responses
            tokenizer = model.get_tokenizer()
            outputs = model.generate(active_prompts, sampling_params=sampling_params, use_tqdm=False)
            for output, trajectory in zip(outputs, active_trajectories):
                trajectory.update_output(output, tokenizer)

            # Step the environment forward
            for trajectory in active_trajectories:
                self.step(trajectory, step, model.get_tokenizer())
            step += 1
        
        # Get the completion_ids of the whole trajectory, i.e., all completions and test results, only excluding the initial prompt.
        # The completion_ids are partially within the prompt token_ids of the final output, since previous completions (and test results) are appended to the prompt for the next one.
        # Therefore, we need to concatenate the prompt_ids sliced to start after the initial prompt, with the completion_ids.
        prompt_ids = [trajectory.output.prompt_token_ids[len(trajectory.initial_prompt_token_ids):] for trajectory in trajectories]
        completion_ids = [trajectory.completion_ids for trajectory in trajectories]
        # Concatenate prompt_ids and completion_ids
        completion_ids = [prompt_id + completion_id for prompt_id, completion_id in zip(prompt_ids, completion_ids)]

        reward_function_kwargs = self._collect_reward_function_kwargs(trajectories)

        # Remove duplicates from ha_prompt_ids (ha_prompt_ids is a list of lists)
        if self.generate_hypothesis:
            self.ha_prompt_ids = list(list(prompt_ids) for prompt_ids in set(tuple(ids) for ids in self.ha_prompt_ids))
            if self.generate_confidence_calibration:
                self.ha_label_ids = processing_class([self.ha_label_format.format(hypothesis=inputs[0]['condition'], confidence="0", eos_token=processing_class.eos_token)] * len(self.ha_prompt_ids), add_special_tokens=False)['input_ids']
            else:
                self.ha_label_ids = processing_class([self.ha_label_format.format(hypothesis=inputs[0]['condition'], eos_token=processing_class.eos_token)] * len(self.ha_prompt_ids), add_special_tokens=False)['input_ids']

        # Generate confidence calibration group generations
        num_generations_ha = len(prompts_text)

        ha_conf_cal_prompt_ids = []
        ha_conf_cal_completion_ids = []
        # Sample a pair of hypothesis and prompt
        ha_pair = self._sample_pair(trajectories)
        if ha_pair is not None:
            # Generate a group of completions for the confidence calibration prompt
            ha_hypothesis, ha_prompt = ha_pair
            conf_cal_prompts = self._generate_conf_cal_prompts(ha_hypothesis, ha_prompt, num_generations_ha)
            ha_conf_cal_outputs = model.generate(conf_cal_prompts, sampling_params=sampling_params, use_tqdm=False)
            ha_conf_cal_prompt_ids = [output.prompt_token_ids for output in ha_conf_cal_outputs]
            ha_conf_cal_completion_ids = [list(output.outputs[0].token_ids) for output in ha_conf_cal_outputs]
        else:
            ha_hypothesis = "unknown"
        
        # Collect triplets for ECE calculation
        ece_calc_triplets = []
        ha_hypotheses_lists = []
        ha_confidences_lists = []
        for trajectory in trajectories:
            ha_hypotheses_list = []
            ha_confidences_list = []
            for hypothesis, confidence in zip(trajectory.hypothesis, trajectory.confidence):
                ha_hypotheses_list.append(hypothesis)
                if self.generate_confidence_calibration:
                    ha_confidences_list.append(confidence)
                    ece_calc_triplets.append((hypothesis, confidence, inputs[0]['condition']))
            ha_hypotheses_lists.append(ha_hypotheses_list)
            ha_confidences_lists.append(ha_confidences_list)

        return EnvironmentOutput(
            completion_ids=completion_ids,
            completion_mask=[trajectory.token_mask[len(trajectory.initial_prompt_token_ids):] for trajectory in trajectories],
            reward_function_kwargs=reward_function_kwargs,
            ha_prompt_ids=self.ha_prompt_ids,
            ha_label_ids=self.ha_label_ids,
            ha_conf_cal_prompt_ids=ha_conf_cal_prompt_ids,
            ha_conf_cal_completion_ids=ha_conf_cal_completion_ids,
            ha_hypothesis=ha_hypothesis,
            ece_calc_triplets=ece_calc_triplets,
            ha_hypotheses_lists=ha_hypotheses_lists,
            ha_confidences_lists=ha_confidences_lists,
        )
        

    def step(self, trajectory: Trajectory, step: int, tokenizer) -> None:
        """Advance a single trajectory by one turn.

        Parses the model's latest completion for an action (Test or Diagnosis),
        validates it, and either appends the observation or marks the trajectory
        as completed/invalid.

        Args:
            trajectory: The trajectory to advance.
            step: Current environment step index.
            tokenizer: HuggingFace tokenizer (for length checks).
        """
        # Check if too many steps
        if step > self.max_steps:
            trajectory.complete(invalid=True)
            return
        
        # 1. Check if the last completion was valid. Yes -> Get action and action input (in lower case). No -> set invalid flag and end.
        valid, action, action_input = self.parse_actions_and_check_validity(trajectory)
        
        if not valid:
            trajectory.complete(invalid=True)
            return
    
        # 2. If the action is unknown, add response asking for repeat.
        if action not in [action.lower() for action in self.possible_actions]:
            observation = "\nError: Unknown action. Valid actions are: " + ", ".join(self.possible_actions) + "." + "\n\n---\n"
            trajectory.error = True
        elif action == "test":
            # 3. If the action is known, check if the action input is valid.
            if action_input not in [test.lower() for test in self.test_list]:
                observation = "\nError: Unknown test. Only request tests as given above." + "\n\n---\n"
                trajectory.error = True
            else:
                # 4. If the action is test, add test results.
                result = trajectory.get_test_result(action_input)
                observation = "\nObservation: " + result + "\n\n---\n"
                if result == "not available.\n":
                    trajectory.error = True
        elif action == "diagnosis":
            # 3. If the action is known, check if the action input is valid.
            if action_input not in [disease.lower() for disease in self.disease_list]:
                observation = "\nError: Unknown diagnosis. Valid diagnoses are: " + ", ".join(self.disease_list) + "." + "\n\n---\n"
                trajectory.error = True
            else:
                # 5. If the action is diagnosis, set it andreturn true complete flag.
                trajectory.diagnosis = action_input
                trajectory.complete()
                observation = None

        # 6. Check length of new text. If too long, set invalid flag.  
        observation_token_length = len(tokenizer.encode(observation)) if observation else 0
        if len(trajectory.token_mask) + observation_token_length > self.max_length:
            trajectory.complete(invalid=True)
            return

        # 7. Add the observation to the trajectory.
        if observation:
            new_text = trajectory.text + observation
            trajectory.text = new_text

    def _generate_hypothesis(self, trajectories: list[Trajectory], model, sampling_params, processing_class) -> list[str]:
        """Generate a hypothesis (and optional confidence) for each active trajectory."""
        hypothesis_prompts = []
        for trajectory in trajectories:
            if trajectory.error:
                continue
            provided_test_results_string = f"Patient History: {trajectory.patient_history}\n\n"
            provided_test_results_string += "\n\n".join([f"{trajectory.test_results_lower_to_original[test]}: {trajectory.test_results[test]}" for test in trajectory.provided_tests])
            hypothesis_prompt = self.hypothesis_prompt_template.format(provided_test_results=provided_test_results_string)
            hypothesis_prompts.append({"prompt": [{"role": "user", "content": hypothesis_prompt}]})
        hypothesis_prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in hypothesis_prompts]

        outputs = model.generate(hypothesis_prompts_text, sampling_params=sampling_params, use_tqdm=False)
        idx = 0
        for trajectory in trajectories:
            if trajectory.error:
                hypothesis = trajectory.hypothesis[-1] if len(trajectory.hypothesis) > 0 else "unknown"
                confidence = trajectory.confidence[-1] if len(trajectory.confidence) > 0 else None
                trajectory.error = False
            else:
                output = outputs[idx]
                idx += 1
                # extract hypothesis from output with regex
                match = re.search(r"Hypothesis: (.*)", output.outputs[0].text)
                hypothesis = match.group(1) if match else "unknown"
                if hypothesis.lower() not in [d.lower() for d in self.disease_list]:
                    hypothesis = "unknown"

                confidence = None
                if self.generate_confidence_calibration:
                    conf_match = re.search(r"Confidence: (\d+)", output.outputs[0].text)
                    confidence = int(conf_match.group(1)) if conf_match else None
                    trajectory.confidence.append(confidence / 10 if confidence is not None else None)

                # add hypothesis to trajectory
                trajectory.hypothesis.append(hypothesis.lower())
                trajectory.hypothesis_prompt.append(output.prompt)

            # format the appended text
            if self.generate_confidence_calibration:
                suffix = f"The current hypothesis is: {hypothesis.lower()}"
                suffix += f" (confidence: {confidence})" if confidence is not None and hypothesis != "unknown" else ""
                suffix += "\n"
            else:
                suffix = f"The current hypothesis is: {hypothesis.lower()}\n"

            if "{first_hypothesis}" in trajectory.text:
                # inject at placeholder
                trajectory.text = trajectory.text.format(
                    first_hypothesis=f"\n\n{suffix}"
                )
            else:
                trajectory.text += suffix
        self.ha_prompt_ids.extend([output.prompt_token_ids for output in outputs])
            
        return [trajectory.text for trajectory in trajectories]
    

    def _sample_pair(self, trajectories: list[Trajectory], forbidden_value: str = 'unknown') -> tuple[str, str] | None:
        # Collect all valid (a[i], b[i]) pairs
        candidates = []
        for trajectory in trajectories:
            hypothesis_list = trajectory.hypothesis
            hypothesis_prompt_list = trajectory.hypothesis_prompt
            for hypothesis, hypothesis_prompt in zip(hypothesis_list, hypothesis_prompt_list):
                if hypothesis != forbidden_value:
                    candidates.append((hypothesis, hypothesis_prompt))

        if candidates:
            return self._rng.choice(candidates)
        return None

    def _generate_conf_cal_prompts(self, ha_hypothesis: str, ha_prompt: str, num_generations_ha: int) -> list[str]:
        return [ha_prompt + "Hypothesis: " + ha_hypothesis.lower() + "\n" + "Confidence: "] * num_generations_ha

    def _collect_reward_function_kwargs(self, trajectories: list[Trajectory]) -> list[dict]:
        return [
            {
                "provided_tests": trajectory.provided_tests,
                "request_count": trajectory.request_count, 
                "diagnosis": trajectory.diagnosis,
                "invalid": trajectory.invalid
            }
            for trajectory in trajectories
        ]
    