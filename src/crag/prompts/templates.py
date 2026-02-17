# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

INSTRUCTIONS = """Assume you are an expert in grading predictions given by a model.
You are given a Question, a Ground Truth answer (always correct), and a Prediction.
Your job is to output one label: correct, missing, or incorrect.

# Definitions:
## Correct
- The prediction answers the question and matches the ground truth in meaning.
- Minor formatting differences are acceptable (e.g., case, punctuation, units formatting, or ordering when order is not meaningful).
- If the ground truth is numeric, the prediction must provide a numerically equivalent value
- For comparison, choice, or identification questions (e.g., “which has more”, “which is larger”, “which came first”):
  - The prediction is correct if it selects the same option or comparative outcome as the ground truth
  - DO NOT evaluate the correctness of any supporting numeric values, counts, or facts used in the prediction.
  - Incorrect or inconsistent supporting details do not affect correctness.

## Missing
- The prediction does not provide an answer.
- This includes:
  - explicit abstentions (e.g., “I don't know”, “not enough information”)
  - refusals to answer
  - empty or near-empty responses
  - responses that clearly avoid answering
- If the prediction provides information that is correct but does not answer the question at all, it is also considered missing.

## Incorrect:
- The prediction attempts to answer, but the final decision, selection, or comparative outcome does not match the ground truth.
- Self-contradictory answers are incorrect.
- Incorrect supporting facts, numbers, or explanations alone are NOT sufficient to label an answer incorrect.

# Grading Rules
- Treat the Ground Truth as the only source of truth.
- If the prediction meets the definition of missing, output missing.
- Otherwise, if the required answer, choice, or comparison matches the ground truth, output correct.
- Otherwise, output incorrect.

# Special Cases
## Sets (lists of items)
- The prediction must contain exactly the same items as the ground truth to be correct:
  - Extra items ⇒ incorrect
  - Missing items ⇒ incorrect
  - Same items in different order ⇒ correct if order is not important, else incorrect
- Ignore trivial formatting differences.
- DO NOT treat duplicates as extra.

## Numbers
- If the ground truth is numeric and the question is not comparative, the prediction must provide the same number or a numerically equivalent value.
- Acceptable differences include formatting variations (e.g., “2.5 million” vs “2,500,000” vs “~2.5M”).

# Output
- Return ONLY a JSON object with two fields:
  - "explanation": a short explanation ≤ 300 chars)
  - "label": one of "correct", "missing", "incorrect"
"""

IN_CONTEXT_EXAMPLES = """Here are some examples to help you make the judgment.
Examples:
Question: "which company has higher eps, btu or cma?"
Ground Truth: "cma"
Prediction: "it is not possible to determine which company has a higher eps."
Output: {"label": "missing", "explanation": "The prediction explicitly abstains (cannot determine)."}

Question: "who is taller, a or b?"
Ground Truth: "a"
Prediction: "The answer is a. a is of height 1.75 m and b is of height 1.82 m. So b is taller."
Output: {"label": "incorrect", "explanation": "The prediction is self-contradictory."}

Question: "who among patricia place and lana clarkson is younger?"
Ground Truth: "lana clarkson"
Prediction: "lana clarkson (born april 5, 1962) is younger than patricia clarkson (born december 29, 1959)."
Output: {"label": "correct", "explanation": "The prediction gives the correct answer."}

Question: "did lakers win more games than warriors in 2024-4?"
Ground Truth: "yes"
Prediction: "yes, lakers won more games than warriors in 2023-2024 season."
Output: {"label": "incorrect", "explanation": "It answers a different timeframe than asked."}

Question: "what is the height of the building where citigroup is headquartered?"
Ground Truth: "151 m"
Prediction: "the height is 915 feet (279 m)."
Output: {"label": "incorrect", "explanation": "The predicted height does not match the ground truth."}

Question: "who were the last 3 prime ministers of uk?"
Ground Truth: "rishi sunak, liz truss, and boris johnson"
Prediction: "1. liz truss, 2. boris johnson, 3.theresa may"
Output: {"label": "incorrect", "explanation": "The set is wrong: missing rishi sunak and includes theresa may."}

Question: "who authored the taming of the shrew?"
Ground Truth: "william shakespeare"
Prediction: "william shakespeare"
Output: {"label": "correct", "explanation": "Exact match."}

Question: "who authored The Taming of the Shrew?"
Ground Truth: "william shakespeare"
Prediction: "w shakespeare"
Output: {"label": "correct", "explanation": "Abbreviation still identifies William Shakespeare."}

Question: "what is the state bird of california?"
Ground Truth: "california quail"
Prediction: "california valley quail"
Output: {"label": "correct", "explanation": "Synonym/alternate name for the same bird."}

Question: "what country is maharashtra metro rail corporation limited located in?"
Ground Truth: "india"
Prediction: "indianapolis"
Output: {"label": "incorrect", "explanation": "The prediction does not match the ground truth."}

Question: "what was wolves's record in their last game in eng-premier league?"
Ground Truth: "w"
Prediction: "wolves's record in their last game in the eng-premier League was a 2-1 win over fulham."
Output: {"label": "correct", "explanation": "It correctly indicates a win."}

Question: "how many seconds is 3 minutes 15 seconds?"
Ground Truth: "195 seconds"
Prediction: "3 minutes 15 seconds is 195 seconds."
Output: {"label": "correct", "explanation": "Correct numeric conversion."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-24, 2023-04-25, 2023-07-25, 2023-10-24"
Prediction: "the company distributed dividends on october 24, 2023 and january 24, 2023."
Output: {"label": "incorrect", "explanation": "Set is incomplete (missing dates)."}

Question: "on which days did xxx distribute dividends in the last year?"
Ground Truth: "2023-01-13, 2023-03-25, 2023-11-21"
Prediction: "xxx distributed dividends on 1. 2023-01-13, 2. 2023-03-25, 3. 2023-10-21."
Output: {"label": "incorrect", "explanation": "Set contains an extra/wrong date not in ground truth."}

Question: "who are current band members of maroon 5?"
Ground Truth: "adam Levine, jesse carmichael, james valentine, matt flynn, pj morton, sam farrar"
Prediction: "they are jesse carmichael, matt flynn, adam levine, pj morton, sam farrar, and james valentine."
Output: {"label": "correct", "explanation": "Same set of members (order differs)."}

Question: "which movies comprise the matrix franchise?"
Ground Truth: "the matrix, the matrix reloaded, the matrix revolutions, the matrix resurrections"
Prediction: "the matrix, the matrix reloaded, the matrix revolutions, the animatrix, and the matrix resurrections."
Output: {"label": "incorrect", "explanation": "Set includes an extra item (the animatrix)."}

Question: "how deep is the deepest lake of new york?"
Ground Truth: "618 ft"
Prediction: "the deepest lake in new york is seneca lake, with a depth of 618.23 feet."
Output: {"label": "correct", "explanation": "Numerically equivalent after rounding."}

Question: "what is the closing price of meta yesterday?"
Ground Truth: "$310.17"
Prediction: "310.2"
Output: {"label": "correct", "explanation": "Matches after rounding."}

Question: "what is the current market cap of appl?"
Ground Truth: "2.81 trillion"
Prediction: "2.667 trillion"
Output: {"label": "incorrect", "explanation": "Numeric value does not match."}

Question: "what is the current pe ratio of appl?"
Ground Truth: "28.3"
Prediction: "the current pe ratio of apple is 26.66"
Output: {"label": "incorrect", "explanation": "Numeric value does not match."}

Question: "how much is tesla's stock price down from its all-time high?"
Ground Truth: "$221.83"
Prediction: "209.52"
Output: {"label": "incorrect", "explanation": "Numeric value does not match."}

Question: "what is the length of amazon river?"
Ground Truth: "over 4000 miles"
Prediction: "the length of amazon river is 4,000 miles"
Output: {"label": "incorrect", "explanation": "Does not satisfy the 'over 4000' condition."}

Question: "how many copies x were sold?"
Ground Truth: "2 million."
Prediction: "it is over 2 million"
Output: {"label": "incorrect", "explanation": "Does not match the specific ground truth value."}

Question: "what is the population of country x?"
Ground Truth: "3,576,873"
Prediction: "the population of country x is 3.3 million."
Output: {"label": "incorrect", "explanation": "Approximation does not match the ground truth number."}

Question: "what is the current market value of stock x?"
Ground Truth: "$2,237,578,268"
Prediction: "$2.16 billion."
Output: {"label": "incorrect", "explanation": "Numeric value does not match ground truth."}

Question: "how old is michael jordan?"
Ground Truth: "61"
Prediction: "mj was born on February 17, 1963."
Output: {"label": "missing", "explanation": "Gives the correct birthdate but does not answer the asked age."}

Question: "how long did amy teach as a teacher?"
Ground Truth: "24 years and 255 days"
Prediction: "25 years."
Output: {"label": "incorrect", "explanation": "Duration does not exactly match ground truth."}

Question: "at what age did usher perform in the half time show of superbowl?"
Ground Truth: "45 years old"
Prediction: "usher has not performed at a super bowl halftime show."
Output: {"label": "incorrect", "explanation": "Contradicts the ground truth."}

Question: "what year did olivia rodrigo win the grammys best new artist award?"
Ground Truth: "2022"
Prediction: "sure! here's an answer in three words or fewer: 2022 grammy awards."
Output: {"label": "correct", "explanation": "Contains the correct year."}

Question: "how many times have zendaya and tom holland starred in a film together?"
Ground Truth: "3"
Prediction: "they have starred in 3 films together: spider-man: homecoming, spider-man: far from home, and spider-man: no way home."
Output: {"label": "correct", "explanation": "Correct count."}

Question: "who is younger, timothee chalamet or tom holland?"
Ground Truth: "tom holland"
Prediction: "timothée chalamet is younger than tom holland."
Output: {"label": "incorrect", "explanation": "Chooses the wrong person."}

Question: "who had more number one hits on the us billboard, a or b?"
Ground Truth: "a had more number one hits on the us billboard than b, with 20 number one hits compared to b's 15."
Prediction: "a"
Output: {"label": "correct", "explanation": "Concise answer matches the ground truth meaning."}

Question: "what is xxx's birthdate?"
Ground Truth: "1996-01-01."
Prediction: "02/01/1996"
Output: {"label": "incorrect", "explanation": "Date format is ambiguous and does not exactly match the ground truth."}

Question: "what was the worldwide box office haul for movie x?"
Ground Truth: "101756123."
Prediction: "102 million"
Output: {"label": "correct", "explanation": "Matches after rounding."}

Question: "how much has spotify's user base increased by since 2020 in na?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "spotify's north american user base increased from 36 million in 2020 to 85 million by 2021"
Output: {"label": "incorrect", "explanation": "Does not answer the asked increase since 2020."}

Question: "how much has spotify's user base increased by since 2020 in na?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "I don't have access to the required information, can you provide more details?"
Output: {"label": "missing", "explanation": "The prediction explicitly abstains from answering the question."}

Question: "how much has spotify's user base increased by since 2020 in na?"
Ground Truth: "spotify's user base increased by 34 million since 2020."
Prediction: "I didn't get that. Could you please rephrase?"
Output: {"label": "missing", "explanation": "The prediction is empty and does not answer the question."}

Question: "Did Spotify's user base increased between 2020 and 2021?"
Ground Truth: "Yes, Spotify's user base increased from 100M in 2020 to 110M in 2021."
Prediction: "Yes, Spotify's user base increased from 130M in 2020 to 150M in 2021."
Output: {"label": "correct", "explanation": "The final answer is correct, even though the explanation does not answer the ground truth exactly."}
"""
