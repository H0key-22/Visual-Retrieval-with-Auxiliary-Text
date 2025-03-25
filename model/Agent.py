import os
from openai import OpenAI


class Agent:
    """Agent for Answer Retrieve System.

    This agent generates potential answers based on a given question, target object, and answer count.
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=open("api_key.txt").read().strip(),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate_answers(self, question, ans_num, target_obj):
        """Generates potential answers based on the given question and target object."""
        template = ("Given the question: {}, generate {} distinct potential answer frameworks.(No more than 10 words) "
                    "The target object in the question is {}. Replace the answer of the question with ___. "
                    "Example: (1) Question: What are the benefits of this herb?, "
                    "(2) Target Object: Ginseng, "
                    "(3) Generation: "
                    "(i)Ginseng can help ____(benefits of herb)"
                    "(ii)Ginseng is known for its ability to ____(benefits of herb)"
                    "(iii)One of the key benefits of using ginseng is that it can ____(benefits of herb)"
                    "(iv)Incorporating ginseng into your diet may help you to ____(benefits of herb)"
                    "(v)Regular consumption of ginseng has been shown to ____(benefits of herb)").format(question, ans_num, target_obj)

        response = self.client.chat.completions.create(
            model="qwen-max",
            messages=[
                {'role': 'user', 'content': template}
            ]
        )

        return response.choices[0].message.content if response.choices else "No response generated."


# Example usage:
if __name__ == "__main__":
    agent = Agent()
    answers = agent.generate_answers(question="How long does this animal take care of its children?",ans_num=3, target_obj="Cardisoma guanhumi")
    print(answers)
