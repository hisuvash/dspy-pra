
import os
from dotenv import load_dotenv
from openai import OpenAI
import dspy
from pydantic import BaseModel, Field
from typing import Optional
import asyncio

model_name=os.getenv("Model_Name")
model_key=os.getenv("API_KEY")

kode_model=dspy.LM(
    f'openai/{model_name}',
    api_key=model_key,
    base_url="https://api.ai.kodekloud.com/v1"

)

dspy.configure(lm=kode_model)

#it is important to disable cache DSPy’s current docs explicitly say that repeated identical calls return the cached output 
# unless you either disable caching or differentiate the calls with rollout_id
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

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
    You are a funny comedian who likes to tell stories before delivering a punchline.
    You are always funny and act on the input joke idea.
    """

    joke_idea: JokeIdea = dspy.InputField()
    joke: str = dspy.OutputField(desc="The full joke delivery in the comedian's voice")

class JokeJudge(dspy.Signature):
    """
    Rank each of the jokes between 1-N. Rank 1 as the most unique and the funniest. Then goes 2, 3 ...N, N being the least uniqe and funniest
    
    """
    joke_idea: list[JokeIdea]= dspy.InputField()
    joke_rankings: list[int] = dspy.OutputField(description= "Rank between 1, 2, 3, ... N, N being the list unique and funniest")
# Rank might not be suffice, because llm may not rank any of the jokes to 1. For a stricter rule we will use refinement in another lab/python file
class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, num_samples=5):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke= dspy.Predict(IdeaToJoke)
        self.judge = dspy.ChainOfThought(JokeJudge)
        self.num_samples = num_samples


    async def aforward(self, query: str):
        predictions = await asyncio.gather(
            *[
                self.query_to_idea.acall(query=query)
                for _ in range (self.num_samples)
            ]

        )
        joke_ideas = [
        prediction.joke_idea
        for prediction in predictions
]



        print ("Generated joke Ideas: \n", joke_ideas)
        judge_score = self.judge(joke_idea=joke_ideas).joke_rankings
        print("Judege Score for each: ", judge_score)

        best_joke_idea_idx = judge_score.index(1)
        print("Selected Index: ", best_joke_idea_idx)

        selected_joke_idea = joke_ideas[best_joke_idea_idx]
        print("Selected Joke Idea:\n", selected_joke_idea)

        joke=self.idea_to_joke(joke_idea=selected_joke_idea)
        return joke
async def main():
    joke_generator=ConditionalJokeGenerator()
    joke=await joke_generator.acall(query="Write a joke on a student who has a very good knowledge on science but no knowledge on geography")

if __name__ == "__main__":
    asyncio.run(main())