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
    joke: str = dspy.OutputField(desc="The full joke delivery in the comedian's voice")

class JokeJudge(dspy.Signature):
    """
    Is this joke idea funny?
    """
    joke_idea: JokeIdea= dspy.InputField()
    joke_rating: int = dspy.OutputField(description= "Rating between 1 to 5", le=5, ge =1)

class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, max_attempts: int=3, good_idea_threshold=4):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke= dspy.Predict(IdeaToJoke)
        self.judge = dspy.ChainOfThought(JokeJudge)
        self.max_attempts = max_attempts
        self.good_idea_threshold= good_idea_threshold

    def forward(self, query:str):
        for _ in range (self.max_attempts):
            print(f"------- Iteration {_ + 1} --------")
            joke_idea = self.query_to_idea(query=query)
            print(f"Joke Idea:\ {joke_idea}")

            judge_score = self.judge(joke_idea=joke_idea).joke_rating

            print(f"\n\n---\nJudge Score: ", judge_score)

            if judge_score>=self.good_idea_threshold:
                print("Judge said it was awesome, skipping it")
                break

        joke = self.idea_to_joke(joke_idea=joke_idea)
        return joke

joke_generator = ConditionalJokeGenerator()
joke=joke_generator(query="Write a joke about science.")

print("------------")
print(joke.joke)



