import re
import string


class DistractorProcessor:
    def __init__(self):
        pass

    @staticmethod
    def remove_duplicates(items: list[str]) -> list[str]:
        seen = set()
        unique_items = []

        for item in items:
            normalized_item = DistractorProcessor._normalize(item)
            if normalized_item not in seen:
                seen.add(normalized_item)
                unique_items.append(item)

        return unique_items

    @staticmethod
    def remove_distractors_duplicate_with_correct_answer(correct: str, distractors: list[str]) -> list[str]:
        normalized_correct = DistractorProcessor._normalize(correct)
        return [
            distractor for distractor in distractors \
            if DistractorProcessor._normalize(distractor) != normalized_correct
        ]

    @staticmethod
    def _normalize(item: str) -> str:
        return ' '.join(
            re.sub(
                r'\b(a|an|the)\b', ' ', item.lower().translate(
                    str.maketrans('', '', string.punctuation)
                )
            ).split()
        )
