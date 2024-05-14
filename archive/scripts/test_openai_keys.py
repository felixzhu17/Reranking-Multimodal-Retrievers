import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.Completion.create(
#     model="gpt-3.5-turbo",
#     prompt="Please answer the following question: who is the current president of the US?",
#     max_tokens=7,
#     temperature=0
# )
PROMPT = """
You are a question answering system. You will need to answer the questions based on the knowledge provided. You need to think and reason step by step. 
If the provided knowledge can be used to answer the question, output [yes].
If the knowledge is not relevant to the question, output [no].
In any case, you should guess the answer if you can not find the answer with the provided knowledge.
Your answer should be less than 5 words.

Here are some examples:
- Question: Which american president is most associated with the stuffed animal seen here?
- Caption:  a group of teddy bears sitting on a chair with wine bottles. 
- Knowledge: we've all played with one at some point; stuffed animal bears are rooted in our country's childhood.  theodore roosevelt, commonly known as teddy, became the 26th president of the united states in 1901.  significantly behind him with longino, decided to tie the bear up for roosevelt.
- Answer: [yes] theodore roosevelt.

- Question: What is on the round plate?
- Caption:  a tray of food with a bowl of fruit and a glass of orange juice. 
- Knowledge: fresh croissants on grey round plate on white wooden surface, top view.  from above, overhead, flat lay.
- Answer: [yes] croissant.

- Question: What's the name of this river?
- Caption:  a red and black boat is sitting on the water. 
- Knowledge: the mississippi river, derived from the old ojibwe word misi-ziibi meaning 'great river' (gichi-ziibi 'big river' at its headwaters), is the second-longest river in the united states; the longest is the missouri river, which flows into the mississippi.  the abandoned distributary diminishes in volume and forms what are known as bayous.  after flowing into cass lake, the name of the river again changed to miskwaawaakokaa-ziibi (red cedar river) and then to gichi-ziibi (big river) after flowing into lake winnibigoshish.
- Answer: [yes] mississippi.

- Question: Why do kids stare?
- Caption:  a woman and two girls sitting on a bench. 
- Knowledge: when my kids stare off into space, i am also annoyed.
- Answer: [no] curious.

- Question: What are the most popular countries for this sport?
- Caption:  a group of young men playing a game of soccer.
- Knowledge: football (soccer) is the nation’s most popular sport, and brazilians are highly enthusiastic fans.  for the country’s beaches, including rio de janeiro’s famous copacabana and ipanema.
- Answer: [yes] brazil.

- Question: What character is this umbrella designed after?
- Caption:  a group of children playing with an umbrella.
- Knowledge: a fantastic but more intense muppet adventure.  charles cassady jr., common sense media parents recommendpopular with kids common sense says age 10+ (i) a fantastic but more intense muppet adventure. , common sense media parents recommendpopular with kids common sense says age 10+ (i) a fantastic but more intense muppet adventure.
- Answer: [no] sock monkey.

- Question: How do you care for this equipment?
- Caption:  a man walking on the beach with a surfboard.
- Knowledge: perform hand hygiene with an alcohol-based hand rub (preferably) or soap and water whenever ungloved hands touch contaminated ppe items.e.3.  patient-care equipment that is required for use by other patients before use.place an appropriate container with a lid outside the door for equipment that requires disinfection or sterilization.keep adequate equipment required for cleaning or disinfection inside the isolation room or area, and ensure scrupulous daily cleaning of the isolation room or area.set up a telephone or other method of communication in the isolation room or area to enable patients, family members or visitors to communicate with health-care workers.
- Answer: [no] wax.

- Question: What company produces the most of the cooking appliance in this photo?
- Caption:  a kitchen with white cabinets and a stove and sink.
- Knowledge: this aerial photo shows appliance park in 1963.  of the domestic refrigerator market, and ge would eventually produce more than a million units. refrigerator innovation did not stop there.
- Answer: [yes] ge.
"""
question = """

- Question: What kind of store would sell this?
- Caption:  a green vase with flowers next to a box. 
- Knowledge: a gift shop or souvenir shop is a store primarily selling souvenirs, memorabilia, and other items relating to a particular topic or theme.  this article is about the type of store.  each will have different business strategies and will typically sell various product ranges that appeal to different customer groups, with gender, age, celebration or personal interest differentiation.
- Answer:
"""
# question = """

# - Question: Which part of this animal would be in use of it was playing the game that is played with the items the man is holding?
# - Caption:  a man holding two frisbees with a dog on his back. 
# - Knowledge: most parts of the caribou were used: flesh, marrow, sinew for thread, hide for clothing, antlers for bows and tools, tallow for lamp light, fat, blood, and the contents of the stomach and intestine [176].  small-scale hunting involved getting close to the animal through stalking or luring so that it could be killed [6].  layers of muscle were sliced off in thin strips and hung to dry.
# - Answer:
# """
# question = """

# - Question: What sport can you use this for?
# - Caption:  a black and silver motorcycle parked next to a building. 
# - Knowledge: riders must use the same motorcycle throughout a race, with repairs made between heats if necessary.  a cotton jersey, nylon pants padded at the knee and thigh, padded boots and gloves, a helmet and goggles, a plastic chest protector, and a kidney belt for support constitute the usual outfit of the motocross cyclist.motocross racingkinney jones britannica quiz sports fun facts quiz what sport's equipment was found in the tomb of an egyptian child buried about 3200 bce? what sport originated because businessmen found the new game of basketball to be too difficult? take this quiz to find out how many fun facts you know about the history and evolution of sports, some of them familiar and others definitely not.
# - Answer: 
# """
# question = """

# - Question: Name the type of plant this is?
# - Caption:  a bathroom with a plant in the middle of it. 
# - Knowledge: if you have a houseplant, but you don't know the exact name, this guide is made for you!1.  a succulent-cactus, a vine, a fern, or another type of herbaceous plant.3.  your plant's name, you can download a mobile app like picture this, plantsnap or garden answers.take a picture of the plant, and these apps will tell you the name right away.
# """
# question = """

# - Question: How many teeth does this animal use to have?
# - Caption:  a cat laying on top of a wooden table looking out a window. 
# - Knowledge: one of the many ways humans differ from others in the animal kingdom goes directly back to our habits as omnivores.  a typical adult mouth will have 32 teeth, which will be the second and the last set in the mouth.  dogs will use these teeth to secure items in their mouth as well as to defend themselves when threatened.
# - Answer:
# """
# question = """

# - Question: How many teeth does this animal use to have?
# - Caption:  a cat laying on top of a wooden table looking out a window. 
# - Knowledge: for example, a meat-eating animal, such as a cat, has quite different teeth compared to a grass-eating animal, such as a horse.  most cats have 26 deciduous teeth and 30 permanent teeth.
# - Answer:
# """
print(PROMPT + question)
res = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.2,
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": PROMPT + question},
        # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        # {"role": "user", "content": "Where was it played?"}
    ]
)
print(res)