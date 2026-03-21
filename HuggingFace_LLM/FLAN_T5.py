################################################################################# 
# 
# FLAN-T5 Model (with Joblib support) 
# 
################################################################################# 
import os 
import joblib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from Hugging Face Transformers 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Choose a instruction-tuned model 
MODEL_NAME = "google/flan-t5-small" 

print(f"FLAN-T5_Summarizer_Q&A_Assistant {MODEL_NAME} model loading...") 

################################################################################# 
# 
# Load model & tokenizer with joblib caching 
# 
################################################################################# 

###############################################################################################################
# Function name :- load_model_and_tokenizer()
# Description :- Load FLAN-T5 model and tokenizer from cache if available, else download and cache them
# Author :- Om Ravindra Wakhare
# Date :- 18/09/2025
###############################################################################################################
def load_model_and_tokenizer():
    model_path = "flan_model.joblib"
    tokenizer_path = "flan_tokenizer.joblib"

    # Check if already saved locally
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        print("Loading model & tokenizer from local joblib cache...")
        model = joblib.load(model_path)
        tokenizer = joblib.load(tokenizer_path)
    else:
        print("Downloading model & tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) 
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME) 
        
        # Save them for reuse
        joblib.dump(model, model_path)
        joblib.dump(tokenizer, tokenizer_path)

    return model, tokenizer


# Load once at start
model, tokenizer = load_model_and_tokenizer()

################################################################################# 
# 
# Function: Run FLAN 
# 
#################################################################################

###############################################################################################################
# Function name :- Marvellous_run_flan()
# Description :- Run FLAN-T5 model inference on given prompt with configurable parameters
# Author :- Om Ravindra Wakhare
# Date :- 18/09/2025
###############################################################################################################
def Marvellous_run_flan(prompt: str, max_new_tokens: int = 128) -> str: 
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True) 
 
    outputs = model.generate( 
        **inputs,           
        max_new_tokens=max_new_tokens,  
        do_sample=True,     
        top_p=0.9,          
        temperature=0.7     
    ) 
 
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

################################################################################# 
# 
# Function: Summarisation 
# 
################################################################################# 

###############################################################################################################
# Function name :- Marvellous_summarize_text()
# Description :- Summarize given text using FLAN-T5 model in 4-6 bullet points
# Author :- Om Ravindra Wakhare
# Date :- 18/09/2025
###############################################################################################################
def Marvellous_summarize_text(text: str) -> str: 
    prompt = f"Summarize the following text in 4-6 bullet points:\n\n{text}" 
    return Marvellous_run_flan(prompt, max_new_tokens=160)

################################################################################# 
# 
# Function: Load context from file 
# 
################################################################################# 

###############################################################################################################
# Function name :- Marvellous_load_context()
# Description :- Load context text from file for question answering
# Author :- Om Ravindra Wakhare
# Date :- 18/09/2025
###############################################################################################################
def Marvellous_load_context(path: str = "context.txt") -> str: 
    try: 
        with open(path, "r", encoding="utf-8") as f: 
            return f.read() 
    except FileNotFoundError: 
        return "" 

################################################################################# 
# 
# Function: Answer from context 
# 
#################################################################################

###############################################################################################################
# Function name :- Marvellous_answer_from_context()
# Description :- Answer questions based on provided context using FLAN-T5 model
# Author :- Om Ravindra Wakhare
# Date :- 18/09/2025
###############################################################################################################
def Marvellous_answer_from_context(question: str, context: str) -> str: 
    if not context.strip(): 
        return "Context file not found or empty. Create 'context.txt' first." 
 
    prompt = ( 
        "You are a helpful assistant. Answer the question ONLY using the context.\n" 
        "If the answer is not in the context, reply exactly: Not found.\n\n" 
        f"Context:\n{context}\n\n" 
        f"Question: {question}\nAnswer:" 
    ) 
 
    return Marvellous_run_flan(prompt, max_new_tokens=120) 

################################################################################# 
# 
# Entry point 
# 
################################################################################# 

###############################################################################################################
# Function name :- main()
# Description :- Entry point of the application with interactive menu for FLAN-T5 operations
# Author :- Om Ravindra Wakhare
# Date :- 18/09/2025
###############################################################################################################
def main(): 
    print("--------------------------------------------------------------------------") 
    print("\n---------------------- FLAN-T5 Model ------------------------") 
    print("1. Summarize the data") 
    print("2. Questions & Answers over local context.txt") 
    print("0. Exit") 
    print("--------------------------------------------------------------------------") 
    
    while True: 
        choice = input("\nChoose an option (1/2/0): ").strip() 
        
        if choice == "0": 
            print("Thank you for using FLAN-T5 Model") 
            break 
        
        elif choice == "1": 
            print("You have selected Summarisation option...") 
            print("\nPaste text to summarize. End with a blank line:") 
    
            lines = [] 
            while True: 
                line = input() 
                if not line.strip(): 
                    break
                lines.append(line)

            text = "\n".join(lines).strip() 
            
            if not text: 
                print("FLAN-T5 says : No text received.") 
                continue

            print("\nSummary generated by Marvellous FLAN model : ") 
            print(Marvellous_summarize_text(text))  
        
        elif choice == "2": 
            ctx = Marvellous_load_context("context.txt") 
            if not ctx.strip(): 
                print("Missing 'context.txt'. Create it in the same folder and try again.") 
                continue 
            
            q = input("\nAsk a question about your context to Marvellous FLAN model : ").strip() 
            if not q: 
                print("No question received.") 
                continue 

            print("\nAnswer from Marvellous FLAN model : ") 
            print(Marvellous_answer_from_context(q, ctx)) 
        
        else: 
            print("Please choose 1, 2, or 0.") 

################################################################################# 
# 
# Starter 
# 
################################################################################# 
if __name__ == "__main__": 
    main()
