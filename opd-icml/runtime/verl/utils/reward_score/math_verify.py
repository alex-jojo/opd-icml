# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math_verify import parse, verify

# This scorer is the source of the training correctness monitor. Missing
# ``math-verify`` must abort before an expensive run starts; silently turning
# every mathematically correct answer into score 0 recreates the exact failure
# mode that previously made the training dashboard unusable.


def last_boxed_only_string(string):
    """Return only a syntactically complete final ``\\boxed{...}`` expression.

    ``Answer:``, ``\\fbox`` and brace-less ``\\boxed`` forms are deliberately
    rejected so training monitoring and formal evaluation use one protocol.
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def answers_equivalent(left: str, right: str) -> bool:
    """Compare two answer payloads with the one shared math verifier."""
    left = str(left)
    right = str(right)
    boxed_left = last_boxed_only_string(left)
    if boxed_left is not None:
        left = remove_boxed(boxed_left)
    boxed_right = last_boxed_only_string(right)
    if boxed_right is not None:
        right = remove_boxed(boxed_right)
    if left is None or right is None or len(left) > 300 or len(right) > 300:
        return False
    try:
        return bool(verify(parse("\\boxed{" + left + "}"), parse("\\boxed{" + right + "}")))
    except Exception:
        return False


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    # verify_func = math_metric(
    #     gold_extraction_target=(LatexExtractionConfig(),),
    #     pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    # )
    # ret_score = 0.0

    # # Wrap the ground truth in \boxed{} format for verification
    # ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    # try:
    #     ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    # except Exception:
    #     pass
    # except TimeoutException:
    #     ret_score = timeout_score

    # return ret_score

    answer = remove_boxed(last_boxed_only_string(model_output))
    if answer is None:
        return 0.0
    return float(answers_equivalent(ground_truth, answer))
