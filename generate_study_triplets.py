
###########################################################
## IMPROVED PROMPT ##
###########################################################

############################################################
# SET THE WORKING DIRECTORY 
############################################################

BASE_DIR = r"C:/Users/paka0001/Desktop/IQ_differences_analyses/batched_triplets/triplet_res_new/triplets_2"

############################################################
# IMPORT PACKAGES
############################################################

import os
import random
import pandas as pd
import json
import itertools
from openai import OpenAI
import numpy as np

############################################################
# SET SEED (REPRODUCIBILITY)
############################################################

SEED = 12345

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

set_seed()

############################################################
# SET API CLIENT
############################################################

client = OpenAI(
    api_key="XXX", 
    base_url="https://openwebui.uni-freiburg.de/api"
)

############################################################
# SETTINGS FOR SPLITTING INTO BATCHES
############################################################

BATCH_SIZE = 100
PROGRESS_FILE = os.path.join(BASE_DIR, "progress_new.json")
SELECTED_FILE = os.path.join(BASE_DIR, "selected_triplets_new.json")
FAILED_FILE   = os.path.join(BASE_DIR, "failed_triplets_new.json")

############################################################
# LOAD STUDIES
############################################################

## specify the path to upload the dataset
csv_path = "C:/Users/paka0001/Desktop/IQ_differences_analyses/batched_triplets/triplet_res_new/triplets_2/preterm_dataset.csv" 

df = pd.read_csv(csv_path)

print("Number of studies in CSV:", len(df))

study_texts = {}

for _, row in df.iterrows():
    study_id = row["study"]
    
    text = f"""
study: {row['study']}
Median_by: {row['Median.by']}
Country: {row['Country']}
Quality: {row['Quality.']}
Matched: {row['Matched']}
Mean_ga: {row['Mean..ga']}
Mean_bw: {row['Mean.bw']}
Mean_age: {row['Mean.age']}
n_EPT.VPT: {row['n_EPT.VPT']}
n_FT: {row['n_FT']}
mean_EPT.VPT: {row['mean_EPT.VPT']}
mean_FT: {row['mean_FT']}
sd_EPT.VPT: {row['sd_EPT.VPT']}
sd_FT: {row['sd_FT']}
Birth_level: {row['Birth_level']}
IQ_type: {row['IQ_type']}
"""
    study_texts[study_id] = text

studies = list(study_texts.keys())
print("Number of studies loaded:", len(studies))

############################################################
# GENERATE TRIPLETS
############################################################

all_triplets = list(itertools.combinations(studies, 3)) ## itertool.combinations(studies, 3) selects only the unique triplets
                                                        ## for each triplet we have the following situations based on the relative distance of B.
                                                        ## 1. A < B < C , then these triplets: (A,B,C) = (B,A,C) = (C,B,A) are all the same, so it chooses one of them.
                                                        ## or 2. B < A < C, then these triplets: (A,B,C) = (B,A,C) = (C,A,B) are all the same, so it chooses one of them.
                                                        ## With 58 studies, the total possible unique triplets is “58 choose 3” = 58×57×56 / 6 = 30,856.
                                                        ## 5However, all the combinations (not only the unique ones) are 58 × 57 × 56 = 185,136 combinations.
print("Total possible triplets:", len(all_triplets))

num_triplets = 5000

if num_triplets > len(all_triplets):
    raise ValueError("Too many triplets requested")

# LOAD THE SAVED STUDIES FROM THE PREVIOUS BATCH
if os.path.exists(SELECTED_FILE):
    with open(SELECTED_FILE, "r") as f:
        selected = json.load(f)
    print("Loaded existing triplet selection")
else:
    random.shuffle(all_triplets)
    selected = all_triplets[:num_triplets]

    with open(SELECTED_FILE, "w") as f:
        json.dump(selected, f)

    print("Saved new triplet selection")

############################################################
# GO ON USING THE PREVIOUS BATCH'S TRIPLETS OUTPUTS
############################################################

## check whether a previous run exists
## if it exists you run the script before
if os.path.exists(PROGRESS_FILE):
    ## load where you left off
    with open(PROGRESS_FILE, "r") as f:
        progress = json.load(f)

    start_idx = progress["last_index"]

    ## load the already generated results
    TRIPLETS_JSON = os.path.join(BASE_DIR, "triplets.json")
    if os.path.exists(TRIPLETS_JSON):
        with open(TRIPLETS_JSON, "r") as f:
            triplets = json.load(f)
    else:
        triplets = []

    ### load the already generated results with absolute paths
    #if os.path.exists("C:/Users/paka0001/Desktop/IQ_differences_analyses/batched_triplets/triplets1000.json"):
    #    with open("C:/Users/paka0001/Desktop/IQ_differences_analyses/batched_triplets/triplets1000.json", "r") as f:
    #        triplets = json.load(f)
    #else:
    #    triplets = []

    ## keep track of what arleady failed
    ## load the failed triplets
    if os.path.exists(FAILED_FILE):
        with open(FAILED_FILE, "r") as f:
            failed = json.load(f)
    else:
        failed = []

    print(f"🔄 Resuming from triplet {start_idx}")

## if a previous run does not exist => fresh start
else:
    start_idx = 0
    triplets = []
    failed = []
    print("🚀 Starting fresh run")

############################################################
# LLM FUNCTION 
############################################################

import time
from openai import OpenAI

def ask_llm(textA, textB, textC, A, B, C, max_retries=3, timeout=120):
### the prompt
    prompt = f"""
You are an expert in epidemiology and meta-analysis. 

Your task is to determine which study is more similar to the anchor study by comparing all the following key epidemiological characteristics:

-mean_bw
-mean_ga 
-Birth_level 
-Median_by 
-IQ_type 

IMPORTANT RULES:
- Output JSON only - no other text
- In reasoning:
  - refer to Study A as "anchor"
  - refer to the more similar study as "positive"
  - refer to the less similar study as "negative"
  - do NOT use labels A, B, or C in reasoning
  -report the true values for each characteristic

OUTPUT FORMAT:

{{
 "positive": "B or C",
 "negative": "B or C",
 "reasons": "Short explanation in the form of short formal report using anchor,positive,negative to refer to studies instead of A,B,C"
}}

STUDIES:

Study A:
{textA}

Study B:
{textB}

Study C:
{textC}
"""

    for attempt in range(max_retries):  ## run multiple times to find positive and negative if one triplet fails
        try:
            start_time = time.time()
            
            ### sends the prompt to the model
            response = client.chat.completions.create(
                model = "openai/gpt-5.4-llmlb",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,   ### slight variability in triplets positives and negatives but the model does not get stuck
                timeout=timeout
            )

            result = response.choices[0].message.content.strip()   ## text response

            ## make the output valied JASON format
            if result.startswith("```"):
                result = result.replace("```json", "").replace("```", "").strip()

            data = json.loads(result)
            
            ## if the LLM respondeds but the output is not udable
            ## if the required triplet context is wrong/incomplete (e.g. positive was not fount etc.) return "error"
            if not all(k in data for k in ["positive", "negative", "reasons"]):
                print(f"  ⚠️ Invalid JSON structure on attempt {attempt + 1}")
                ## LLM re-tries
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                ## give up if the last attempt fail
                return None
            
            ## compute the response time
            elapsed = time.time() - start_time
            print(f"  ⏱️ Response time: {elapsed:.1f}s")
            ## return the triplets with their explanation
            return data
        
        ## "Except" triggers when something breaks. When something crushed and we coundl't process the LLM response 
        ## try all the previous and if triplets positive, geatives and reasons fail quit
        except Exception as e:
            print(f"  ⚠️ Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"  ⏳ Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Max retries reached")
    ## if everything failed return None
    return None

############################################################
# PROCESS TRIPLET 
############################################################

def process_triplet(A, B, C):
    ## ask LLM to decide which study is more similar to the reference (A)
    result = ask_llm(study_texts[A], study_texts[B], study_texts[C], A, B, C)
 
    ## label mapping
    if result is None:
        return None

    label_map = {"A": A, "B": B, "C": C}    ## B is just the label not the actual study's name

    pos = label_map[result["positive"]]
    neg = label_map[result["negative"]]

    ## THE MAIN OUTPUT
    explanation = f"{pos} is more similar to {A} than {neg}. "
 
    ## if you found many reasons then the output is a "list" case
    if isinstance(result["reasons"], list):
        explanation += " ".join(result["reasons"])
    ## if you only found one reason then the output is a sinlge "string" not a list
    elif isinstance(result["reasons"], str) and result["reasons"]:
        explanation += result["reasons"]

    return {
        "anchor": A,
        "positive": pos,
        "negative": neg,
        "explanation": explanation,
        "A": A,
        "B": B,
        "C": C,
        "seed": SEED
    }

############################################################
# SAVE OUTPUTS
############################################################

def save_outputs(triplets, failed_calls):

    TRIPLETS_JSON = os.path.join(BASE_DIR, "triplets_new.json")
    TRIPLETS_CSV  = os.path.join(BASE_DIR, "triplets_new.csv")

    with open(TRIPLETS_JSON, "w") as f:
        json.dump(triplets, f, indent=2)

    pd.DataFrame(triplets).to_csv(TRIPLETS_CSV, index=False)

    print("Saved JSON:", TRIPLETS_JSON)
    print("Saved CSV:", TRIPLETS_CSV)

    ## or using the absolute paths
    #json_path = "C:/Users/paka0001/Desktop/IQ_differences_analyses/batched_triplets/triplets1000.json"
    #csv_path = "C:/Users/paka0001/Desktop/IQ_differences_analyses/batched_triplets/triplets1000.csv"

    #with open(json_path, "w") as f:
    #    json.dump(triplets, f, indent=2)

    #pd.DataFrame(triplets).to_csv(csv_path, index=False)

    #print("Saved JSON:", json_path)
    #print("Saved CSV:", csv_path)

    if failed_calls:
        with open("failed_triplets.json", "w") as f:
            json.dump(failed_calls, f, indent=2)

    print("\nTotal successful:", len(triplets))
    print("Total failed:", len(failed_calls))

############################################################
# MAIN LOOP 
############################################################

print(f"\nGenerating {num_triplets} triplets...\n")

## loop over the triplets and start from where you left
for i in range(start_idx, len(selected)):

    A, B, C = selected[i]

    print(f"Triplet {i+1}: {A}, {B}, {C}")

    ## generate the triplet with the process_triplet function
    result = process_triplet(A, B, C)

    if result is None:
        failed.append((A, B, C))
        print("❌ Failed\n")
    ## print the output in the format you want
    else:
        triplets.append(result)
        print(f"✅ Pos: {result['positive']} | Neg: {result['negative']}")
        print(f"   {result['explanation']}\n")
    
    ## BATCH SAVE 
    if (i + 1) % BATCH_SIZE == 0:

        print(f"\n💾 Saving batch at triplet {i+1}...\n")

        save_outputs(triplets, failed)

        with open(PROGRESS_FILE, "w") as f:
            json.dump({"last_index": i + 1}, f)

        with open(FAILED_FILE, "w") as f:
            json.dump(failed, f, indent=2)

    # Delay between requests
    ## Svoids hitting API rate limits and reduces server's overload
    if i < len(selected) - 1:
        time.sleep(1)

############################################################
# FINAL SAVE
############################################################

save_outputs(triplets, failed)

with open(PROGRESS_FILE, "w") as f:
    json.dump({"last_index": len(selected)}, f)

with open(FAILED_FILE, "w") as f:
    json.dump(failed, f, indent=2)

print("\nFINAL SUMMARY")
print(f"Success: {len(triplets)}")
print(f"Failed: {len(failed)}")
