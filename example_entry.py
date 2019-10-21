"""
Each entry in personachat is a dict with two keys personality and utterances, the dataset is a list of entries.
personality:  list of strings containing the personality of the agent
utterances: list of dictionaries, each of which has two keys which are lists of strings.
    candidates: [next_utterance_candidate_1, ..., next_utterance_candidate_19]
        The last candidate is the ground truth response observed in the conversational data
    history: [dialog_turn_0, ... dialog_turn N], where N is an odd number since the other user starts every conversation.
Preprocessing:
    - Spaces before periods at end of sentences
    - everything lowercase
"""
EXAMPLE_ENTRY = {
    "personality": ["i like to remodel homes .", "i like to go hunting .",
                    "i like to shoot a bow .", "my favorite holiday is halloween ."],
    "utterances": [
        {"candidates": [
            "my mom was single with 3 boys , so we never left the projects .",
            "i try to wear all black every day . it makes me feel comfortable .",
            "well nursing stresses you out so i wish luck with sister",
            "yeah just want to pick up nba nfl getting old",
            "i really like celine dion . what about you ?", "no . i live near farms .",
            "i wish i had a daughter , i'm a boy mom . they're beautiful boys though still "
            "lucky",
            "yeah when i get bored i play gone with the wind my favorite movie .",
            "hi how are you ? i'm eating dinner with my hubby and 2 kids .",
            "were you married to your high school sweetheart ? i was .",
            "that is great to hear ! are you a competitive rider ?",
            "hi , i'm doing ok . i'm a banker . how about you ?", "i'm 5 years old",
            "hi there . how are you today ?",
            "i totally understand how stressful that can be .",
            "yeah sometimes you do not know what you are actually watching",
            "mother taught me to cook ! we are looking for an exterminator .",
            "i enjoy romantic movie . what is your favorite season ? mine is summer .",
            "editing photos takes a lot of work .",
            "you must be very fast . hunting is one of my favorite hobbies ."
        ],
        "history": [
            "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .",
        ]
        },
        {"candidates": ["hello i am doing well how are you ?",
                        "ll something like that . do you play games ?",
                        "does anything give you relief ? i hate taking medicine "
                        "for mine .",
                        "i decorate cakes at a local bakery ! and you ?",
                        "do you eat lots of meat",
                        "i am so weird that i like to collect people and cats",
                        "how are your typing skills ?",
                        "yeah . i am headed to the gym in a bit to weight lift .",
                        "yeah you have plenty of time",
                        "metal is my favorite , but i can accept that people "
                        "listen to country . haha",
                        "that's why you desire to be controlled . let me control "
                        "you person one .",
                        "two dogs they are the best , how about you ?",
                        "you do art ? what kind of art do you do ?",
                        "i love watching baseball outdoors on sunny days .",
                        "oh i see . do you ever think about moving ? i do , "
                        "it is what i want .",
                        "sure . i wish it were winter . the sun really hurts my "
                        "blue eyes .",
                        "are we pretending to play tennis",
                        "i am rich and have all of my dreams fulfilled already",
                        "they tire me so , i probably sleep about 10 hrs a day "
                        "because of them .",
                        "i also remodel homes when i am not out bow hunting ."],
         "history": [
            "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .",
             "you must be very fast . hunting is one of my favorite hobbies .",
             "i am ! for my hobby i like to do canning or some whittling .",]
         },
        {"candidates": ["yes they do but i say no to them lol",
                        "i have trouble getting along with family .",
                        "i live in texas , what kind of stuff do you do in "
                        "toronto ?",
                        "that's so unique ! veganism and line dancing usually "
                        "don't mix !",
                        "no , it isn't that big . do you travel a lot",
                        "that's because they are real ; what do you do for "
                        "work ?",
                        "i am lazy all day lol . my mom wants me to get a job "
                        "and move out",
                        "i was born on arbor day , so plant a tree in my name",
                        "okay , i should not tell you , its against the rules "
                        "but my name is sarah , call me o",
                        "hello how are u tonight",
                        "cool . . . my parents love country music that's why i "
                        "hate it",
                        "i am an accountant . what do you do ?",
                        "what do your parents do ? my dad is a mechanic .",
                        "how are you liking it ?",
                        "i really am too . great talking to you too .",
                        "cool . whats it like working there ?",
                        "one daughter . she's pre med",
                        "no and all men is taller than me why can't i find a "
                        "man to dance with",
                        "i live in utah , and my family live in england , "
                        "so i understand",
                        "that's awesome . do you have a favorite season or "
                        "time of year ?"],
         "history": [
             "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .",
             "you must be very fast . hunting is one of my favorite hobbies .",
             "i am ! for my hobby i like to do canning or some whittling .",
             "i also remodel homes when i am not out bow hunting .",
             "that's neat . when i was in high school i placed 6th in 100m dash !", ]
         },
    ]
}
