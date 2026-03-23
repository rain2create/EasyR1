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

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata
REWARD_NAME = "math"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:   ## 格式奖励
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)  ## 要求输出包含<think>...</think>和\boxed{...}，且<think>在\boxed前面，且两者之间可以有任意字符（包括换行）
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:  ## 用数学对错来做准确率奖励
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0



## 这里的奖励函数既考虑了输出的格式是否符合要求（format_reward），也考虑了输出的内容是否正确（accuracy_reward）。最后通过一个加权平均的方式将两者结合成一个overall分数，既鼓励模型输出正确的答案，也鼓励模型按照指定的格式输出。
def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:  ## 这里的reward_inputs是一个包含多个样本的列表，每个样本是一个字典，包含模型的response和对应的ground_truth。函数会对每个样本计算格式奖励和准确率奖励，并根据format_weight将两者结合成一个overall分数，最后返回一个包含每个样本分数的列表。
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
