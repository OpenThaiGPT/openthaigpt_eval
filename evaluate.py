
import sys
sys.path.append("./models")
from tqdm import tqdm
import os
import pandas as pd
import re
exam_path = './exams'
answer_pattern = r"\([1-5]\)"

# current date time
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%Y-%m-%d_%H-%M-%S")

# Check_Answer function
# For each row in test set, check if the model's answer is correct.
# - answer:str - Model's answer
# - sol:str - Ground truth answer
#
# Return True if the model's answer is correct, otherwise False.
def check_answer(answer:str, sol:str):
    assert isinstance(answer, str)
    assert isinstance(sol, str)
    
    answer = answer.strip()
    sol = sol.strip()
    print(f"Model's answer > {answer}")
    print(f"Ground truth answer > {sol}")
    
    # For xnli2.0_th_200, some model was trained to output entailment, neutral, contradiction directly.
    if answer == "entailment":
        answer = "(1)"
    elif answer == "neutral":
        answer = "(2)"
    elif answer == "contradiction":
        answer = "(3)"
    
    # If there is the following prefix, most likely the answer will be after this part.
    if "คำตอบที่ถูกต้องคือ" in answer:
        answer = answer.split("คำตอบที่ถูกต้องคือ")[-1]
        
    elif "คำตอบคือ" in answer: 
        answer = answer.split("คำตอบคือ")[-1]
        
    elif "ตอบว่า:" in answer:
        answer = answer.split("ตอบว่า:")[-1]
   
    # Extract a correct choice from ground truth answer
    search_sol = re.search(answer_pattern, sol)
    assert search_sol is not None, f"Can not found candidates in solution: {sol}"
    sol_choice = str(search_sol.group())
    
    # Extract a choice from model's answer
    search_answer = re.search(answer_pattern, answer)
    if search_answer:
        ans_choice = str(search_answer.group())
        print(f"Answer Choice: {ans_choice}")
        print(f"Solution Choice: {sol_choice}")
        print(f"Is Correct: {ans_choice == sol_choice}")
        return ans_choice == sol_choice
    
    # If a choice is not found model's answer, it is wrong.
    return False


def main(init, inference, model_name, model_path_or_api_key=None):
    # Init model
    init(model_name, model_path_or_api_key)
    
    model_name_safe_filepath = model_name.replace("/", "_")
    
    filename = f'outputs/{current_time}_{model_name_safe_filepath}_answer.tsv'
    stat_filename = f'outputs/{current_time}_{model_name_safe_filepath}_stat.tsv'
    stat_by_year_filename = f'outputs/{current_time}_{model_name_safe_filepath}_stat_by_year.tsv'
    
    if os.path.exists(filename):
        print(f"'{filename}' exists.")
        is_existed = True
    else:
        print(f"'{filename}' does not exist.")
        is_existed = False

    # Answer file name
    with open(filename, 'a') as f:
        if (not is_existed):
            f.write("Exam name" + "\t"+ "Year" + "\t" + "Question No" + "\t" +  "Question" + "\t" + "Choices" + "\t" + "Model Answer" + "\t" + "Solution" + "\t" + "Is Correct?" + '\n')
            f.flush()
        
        exam_paths = os.listdir(exam_path)
        exam_paths.sort()
        
        for name in exam_paths:
            if name[-4:] == ".csv":
                print("Evaluating exam:", name)
                num_correct = {}
                num_question = {}
                df = pd.read_csv(os.path.join(os.getcwd(), exam_path, name))

                for row in tqdm(df.itertuples()):
                    
                    year = str(row.year)
                    
                    if (not (row.isAnswerable and row.isMultipleChoice and row.isSingleChoiceSolution)):
                        continue
                    
                    instruction = str(row.instruction).replace("\t"," ").strip()
                    input = str(row.input).replace("\t"," ").strip()
                    result = str(row.result).replace("\t"," ").strip()
                    
                    instruction = f"ตอบคำถามดังต่อไปนี้โดยการเลือกคำตอบตาม Choice ที่กำหนดให้เท่านั้น ไม่ต้องอธิบายเพิ่ม อาทิเช่น 'คำตอบที่ถูกต้องคือ (1)'\nคำถาม: {instruction}\nChoice: {input}"
                    answer = inference(instruction)
                    correct = check_answer(answer, result)
                    
                    if year not in num_correct:
                        num_correct[year] = 0
                        num_question[year] = 0
                        
                    if ("_all" not in num_correct):
                        num_correct["_all"] = 0
                        num_question["_all"] = 0
                        
                    num_question[year] +=1
                    num_question["_all"] +=1
                    if correct:
                        num_correct[year] += 1
                        num_correct["_all"]+=1
                    f.write(str(name[:-4]) + "\t"+ str(year) + "\t" + str(row.no) + "\t" + instruction + "\t" + input + "\t" + answer + "\t" + str(row.result) + "\t" + str(correct) + '\n')
                    f.flush()

            
            # Stat file name
            if os.path.exists(stat_filename):
                print(f"'{stat_filename}' exists.")
                stat_is_existed = True
            else:
                print(f"'{stat_filename}' does not exist.")
                stat_is_existed = False
                
            # Stat file name
            if os.path.exists(stat_by_year_filename):
                print(f"'{stat_by_year_filename}' exists.")
                stat_by_year_is_existed = True
            else:
                print(f"'{stat_by_year_filename}' does not exist.")
                stat_by_year_is_existed = False
            
            with open(stat_filename, 'a') as fs:
                if (not stat_is_existed):
                    fs.write("Exam name"+ "\t" + "Year" + "\t" + "Score" + "\t" + "Total" + "\t" + "Score (%)" + '\n')
                    fs.flush()
                
                with open(stat_by_year_filename, 'a') as fy:
                    if (not stat_by_year_is_existed):
                        fy.write("Exam name"+ "\t" + "Year" + "\t" + "Score" + "\t" + "Total" + "\t" + "Score (%)" + '\n')
                        fy.flush()

                    # Get list of key in from object num_correct and sort alphabetically
                    keys = list(num_correct.keys())
                    keys.sort()
                    
                    for key in keys:
                        if (key == "_all"):
                            fs.write(str(name[:-4]) + "\t"+ "All" + "\t" + str(num_correct[key])+"\t"+str(num_question[key])+"\t{:.2%}".format(num_correct[key]/num_question[key])+"\n")
                            fs.flush()
                        else:
                            fy.write(str(name[:-4]) + "\t"+ str(key) + "\t" + str(num_correct[key])+"\t"+str(num_question[key])+"\t{:.2%}".format(num_correct[key]/num_question[key])+"\n")
                            fy.flush()

if __name__ == "__main__":
    support_models = {
        "openthaigpt/openthaigpt-1.0.0-beta-7b-chat-ckpt-hf":"openthaigpt_hf_7b_2023",
        "openthaigpt/openthaigpt-1.0.0-beta-13b-chat-hf":"openthaigpt_hf_13b_2023",
        "openthaigpt/openthaigpt-1.0.0-7b-chat":"openthaigpt_hf_2024",
        "openthaigpt/openthaigpt-1.0.0-13b-chat":"openthaigpt_hf_2024",
        "openthaigpt/openthaigpt-1.0.0-70b-chat":"openthaigpt_hf_2024",
        "sail/Sailor-7B-Chat":"sailor",
        "pythainlp/wangchanglm-7.5B-sft-enth":"wangchanglm",
        "aisingapore/sea-lion-7b-instruct":"sealion",
        "SeaLLMs/SeaLLM-7B-v1":"seallm_v1",
        "SeaLLMs/SeaLLM-7B-v2":"seallm_v2",
        "claude-3-opus-20240229":"claude",
        "claude-3-sonnet-20240229":"claude",
        "claude-3-haiku-20240307":"claude",
        "typhoon-instruct":"typhoongpt",
        "gpt-3.5-turbo":"openai",
        "gpt-4":"openai",
        "gemini-pro-1.5":"gemini"
    }
    
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python evaluate.py <model_name> [model_path/api_key]")
        print("Description: Evaluate the model with all exams in exams folder with the given model name.")
        print("Available models:")
        for key in support_models:
            print(f" - {key}")
        sys.exit(1)

    if sys.argv[1] not in support_models:
        print(f"Model '{sys.argv[1]}' is not supported.")
        sys.exit(1)
    
    # Load the correct model
    model_name = sys.argv[1]
    
    if len(sys.argv) == 3:
        model_path_or_api_key = sys.argv[2]
    else:
        model_path_or_api_key = None
    
    model_class = support_models[model_name]
    if (model_class == "openthaigpt_hf_2024"):
        from models.openthaigpt_hf_2024 import init, inference
    elif (model_class == "openthaigpt_hf_7b_2023"):
        from models.openthaigpt_hf_7b_2023 import init, inference
    elif (model_class == "openthaigpt_hf_13b_2023"):
        from models.openthaigpt_hf_13b_2023 import init, inference
    elif (model_class == "sailor"):
        from models.sailor import init, inference
    elif (model_class == "wangchanglm"):
        from models.wangchanglm import init, inference
    elif (model_class == "sealion"):
        from models.sealion import init, inference
    elif (model_class == "seallm_v1"):
        from models.seallm_v1 import init, inference
    elif (model_class == "seallm_v2"):
        from models.seallm_v2 import init, inference
    elif (model_class == "claude"):
        from models.claude import init, inference
    elif (model_class == "typhoongpt"):
        from models.typhoongpt import init, inference
    elif (model_class == "openai"):
        from models.openai import init, inference
    elif (model_class == "gemini"):
        from models.gemini import init, inference
    
    # Check exam path
    if (not os.path.exists(exam_path)):
        print(f"'{exam_path}' does not exist.")
        sys.exit(1)
    
    # List all exams in exam_path
    exam_paths = os.listdir(exam_path)
    exam_paths.sort()
    for name in exam_paths:
        if name[-4:] == ".csv":
            print("Found exam:", name)
            
    main(init, inference, model_name, model_path_or_api_key)
