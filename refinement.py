
## This code refines the jokes

import os
from dotenv import load_dotenv
from openai import OpenAI
import dspy
from pydantic import BaseModel, Field
from typing import Optional
import asyncio

import mlflow
mlflow.dspy.autolog()
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Refinements")

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
    #  defines the input and output contract for the step that turns the user's topic into a structured joke idea.
    """
    You are a funny comeidan and your goal is to generate a nice structure of a joke.
    """
    # I added here style so that jokeIdea will get variation otherwise we might be generating the same joke idea because our query string is hardcoded and never changes.
    query: str = dspy.InputField()
    style: str = dspy.InputField()
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
    joke_ratings: list[int] = dspy.OutputField(description= "Rank between 1, 2, 3, ... N, N being the list unique and funniest")
# Refinement here

def check_score_goodness(args, pred):
    num_samples = len(args["joke_idea"])
    same_length = len(pred.joke_ratings) == num_samples
    all_ranks_present = all([(i+1) in pred.joke_ratings for i in range(num_samples)])
    print("\n--- REWARD DEBUG ---")
    print("Number of jokes:", num_samples)
    print("Received ratings:", pred.joke_ratings)
    print("--------------------\n")
    return  1.0 if (same_length and all_ranks_present) else 0.0

class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, num_samples=5):
        self.query_to_idea = dspy.ChainOfThought(QueryToIdea)
        self.idea_to_joke= dspy.ChainOfThought(IdeaToJoke)

        self.idea_to_joke.set_lm(lm=dspy.LM( f'openai/{model_name}',
            api_key=model_key,
            base_url="https://api.ai.kodekloud.com/v1",
            temperature=0.7))
        self.num_samples = num_samples

        #self.judge = dspy.ChainOfThought(JokeJudge)

        self.judge = dspy.Refine(
            module = dspy.ChainOfThought(JokeJudge), 
            N=3,
            reward_fn=check_score_goodness,
            threshold=1.0
        )


    async def aforward(self, query: str):
        # predictions = await asyncio.gather(
        #     *[
        #         self.query_to_idea.acall(query=query)
        #         for _ in range (self.num_samples)
        #     ]

        # )

        # I am adding this to get variations in the joke idea
        styles = [
            "wordplay",
            "absurd misunderstanding",
            "sarcasm",
            "exaggeration",
            "situational comedy"
            ]
        predictions = await asyncio.gather(
            *[
                self.query_to_idea.acall(
                    query=query,
                    style=styles[i],
                    config={
                        "rollout_id": i,  # Treat this generation as a different rollout, even if the input prompt is the same.
                        "temperature": 1.0
                    }
                )
                for i in range(self.num_samples)
            ]
        )
        joke_ideas = [
        prediction.joke_idea
        for prediction in predictions
        ]

        print("\n--- GENERATED JOKE IDEAS ---")

        # for i, idea in enumerate(joke_ideas, start=1):
        #     print(f"\nJoke Idea {i}")
        #     print("Setup:", idea.setup)
        #     print("Contradiction:", idea.contradiction)
        #     print("Punchline:", idea.punchline)

        # print("==="*30)
        print("*******"*40)
        print(kode_model.inspect_history(1))
        
        print ("\nGenerated joke Ideas: \n", joke_ideas)
        judge_score = self.judge(joke_idea=joke_ideas).joke_ratings
        print("Judege Score for each: ", judge_score)

        best_joke_idea_idx = judge_score.index(1)
        print("Selected Index: ", best_joke_idea_idx)

        selected_joke_idea = joke_ideas[best_joke_idea_idx]
        print("Selected Joke Idea:\n", selected_joke_idea)

        joke=self.idea_to_joke(joke_idea=selected_joke_idea)
        return joke
async def main():
    joke_generator=ConditionalJokeGenerator()
    joke=await joke_generator.acall(query="Write a joke on a student who has a very good knowledge on science but no knowledge on maths")

    print("---"*50)
    print(joke)

if __name__ == "__main__":
    asyncio.run(main())