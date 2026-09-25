# Iterative Refinement
# In this process, LLM refines its output on itself, what we do it, once the output is generated, once again we pass
# the generated output back to the llm, to reflect on the generated output and look for the possibilities of refinement


import os
from dotenv import load_dotenv
from openai import OpenAI
import dspy
from pydantic import BaseModel, Field
from typing import Optional

model_name=os.getenv("Model_Name")
model_key=os.getenv("API_KEY")

kode_model=dspy.LM(
    f'openai/{model_name}',
    api_key=model_key,
    base_url="https://api.ai.kodekloud.com/v1"

)

dspy.configure(lm=kode_model)

class JokeIdea(BaseModel):
    setup: str
    contradiction: str
    punchline: str

class QueryToIdea(dspy.Signature):
    """
    You are a funny comeidan and your goal is to generate a nice structure of a joke.
    """

    query: str = dspy.InputField()
    joke_idea: JokeIdea = dspy.OutputField()

class IdeaToJoke(dspy.Signature):
    """
    You are a funnu comedian who likes to tell stories before delivering a punchline.
    You are always funny and act on the input joke idea.
    """

    joke_idea: JokeIdea = dspy.InputField()
    draft_joke: Optional[str] = dspy.InputField(desc = "a draft joke")
    feedback: Optional[str] = dspy.InputField(desc="feedback on the draft joke")

    joke: str = dspy.OutputField(desc="The full joke delivery in the comedian's voice")

class Refinement(dspy.Signature):
    """
    Given a joke, is it funny? If not, suggest a change.
    """

    joke_idea: JokeIdea = dspy.InputField()
    joke: str = dspy.InputField()
    change: str = dspy.OutputField()

class IterativeJokeGenerator(dspy.Module):
    def __init__(self, n_attempts: int=3):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke= dspy.Predict(IdeaToJoke)
        self.refinement = dspy.ChainOfThought(Refinement)
        self.n_attempts = n_attempts

    def forward(self, query:str):
        joke_idea = self.query_to_idea(query=query)
        print(f"Joke Idea: \n{joke_idea}")

        draft_joke= None
        feedback= None

        for _ in range (self.n_attempts):
            print(f"------- Iteration {_ + 1} --------")
            joke = self.idea_to_joke(joke_idea= joke_idea, draft_joke=draft_joke, feedback=feedback)
            print(f"Joke:\ {joke}")

            feedback=self.refinement(joke_idea=joke_idea, joke=joke)
            print(f'Feedback:\n {feedback}')

            draft_joke=joke
            feedback=feedback.change
        return joke

joke_generator = IterativeJokeGenerator()
joke=joke_generator(query="Write a joke about science.")

print("------------")
print(joke.joke)



