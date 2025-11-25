import random
import csv
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold

### functions to get few-shot list

def get_random_few_shot_list(dataset, loaded_data_train_val, loaded_data_test):
    """
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    loaded_data: the loaded csv file in format [ [first-row requirement, first-row category, .. ], [second-row requirement, second-row category, .. ], [], ... ]
    output
    few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    """
  
    num_k_closest_examples = 160
    few_shot_list = []
    k_closest_examples = []

    # create the list of k closest few-shot examples
    values = list(range(0, len(loaded_data_train_val)))
    random.shuffle(values)
    for i in range(num_k_closest_examples):
        index = values.pop()
        k_closest_examples.append(loaded_data_train_val[index])

    # repeat and reuse this list of k closest few-shot examples for each software requirement, i.e. same list for each requirement
    for i in range(len(loaded_data_test)):
        few_shot_list.append(k_closest_examples)
    
    return few_shot_list


def get_embedding_few_shot_list(dataset, loaded_data_train_val, loaded_data_test):
    """
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    loaded_data: the loaded csv file in format [ [first-row requirement, first-row category, .. ], [second-row requirement, second-row category, .. ], [], ... ]
    output
    few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    """
    few_shot_list = []
    num_k_closest_examples = 160

    # collect all textual requirements
    loaded_data_train_val_requirement = [sublist[0] for sublist in loaded_data_train_val] 
    loaded_data_test_requirement = [sublist[0] for sublist in loaded_data_test] 
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding_train_val = model.encode(loaded_data_train_val_requirement)
    embedding_test = model.encode(loaded_data_test_requirement)

    # for each single test_requirement sample, rank the top relevant requirements in training dataset and obtain their index 
    index = find_most_relevant_vectors(embedding_test, embedding_train_val, num_k_closest_examples)

    for i in range(len(loaded_data_test)):
        k_closest_examples = []
        for j in range(num_k_closest_examples):
            k_closest_examples.append(loaded_data_train_val[index[i][j]])
        few_shot_list.append(k_closest_examples)

    return few_shot_list


def find_most_relevant_vectors(embedding_test, embedding_train_val, num_k_closest_examples):  
    """for each single test_requirement sample, rank the top relevant requirements in training dataset and obtain their index 
    embedding_test: embedding matrix, the number of test samples x embedding dimensions
    embedding_train_val: embedding matrix, the number of test samples x embedding dimensions
    output
    index: 2-dimensional list,  [ [the indexes of the most relevant training samples for the first requirement in test dataset], [the indexes of the most relevant training samples for requirement 2 in test dataset], ...]
    """
    index = []
    for i, vec1 in enumerate(embedding_test):  
        distances = []  
        for j, vec2 in enumerate(embedding_train_val):  
            distance = np.linalg.norm(vec1 - vec2)  # Euclidean distance  
            distances.append((distance, i, j))  
        distances.sort()  # Sort distances in ascending order  

        index_i_requirement = [ distances[k][2] for k in range(num_k_closest_examples)]   # iterate over the top K indices of the most relevant vectors 
        index.append(index_i_requirement)

    return index  


def get_tfidf_few_shot_list(dataset , loaded_data_train_val, loaded_data_test):
    """
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    loaded_data: the loaded csv file in format [ [first-row requirement, first-row category, .. ], [second-row requirement, second-row category, .. ], [], ... ]
    output
    few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    """
    few_shot_list = []
    num_k_closest_examples = 160

    # collect all textual requirements
    loaded_data_train_val_requirement = [sublist[0] for sublist in loaded_data_train_val] 
    loaded_data_test_requirement = [sublist[0] for sublist in loaded_data_test] 

    #transform to tfidf vectors
    v = TfidfVectorizer(min_df=0.001, max_df=0.8, analyzer='word')
    fitted_corpus = v.fit(loaded_data_train_val_requirement)
    tfidf_train_val = v.transform(loaded_data_train_val_requirement)
    tfidf_test = v.transform(loaded_data_test_requirement)
    # transform sparse matrix to numpy array
    tfidf_train_val = np.float32(  tfidf_train_val.toarray() )   
    tfidf_test = np.float32(  tfidf_test.toarray() )

    # for each single test_requirement sample, rank the top relevant requirements in training dataset and obtain their index 
    index = find_most_relevant_vectors(tfidf_test, tfidf_train_val, num_k_closest_examples)
    
    for i in range(len(loaded_data_test)):
        k_closest_examples = []
        for j in range(num_k_closest_examples):
            k_closest_examples.append(loaded_data_train_val[index[i][j]])
        few_shot_list.append(k_closest_examples)

    return few_shot_list


######## functions for prompt construction

def get_all_prompt( few_shot_list, df_test, dataset, method, num_shot, bi_classification):
    """ loaded_data is a N x 3 array for promise dataset, N x 2 array for nfr_so dataset
    input
        df_test: the dataframe we get from cross validation
        dataset, string name of the evaluated dataset
        method, string name of the used method   
        num_shot, int, the number of few-shot examples in the prompt
        few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    output
        the list of prompts, string list
    """
    prompt_list = []

    for requirement_index in range (len(df_test)):
        input_requirement = df_test[requirement_index][0]
        prompt = prompt_construction( few_shot_list, dataset, method, num_shot, bi_classification, requirement_index, input_requirement )
        prompt_list.append(prompt)

    return prompt_list


def prompt_construction(few_shot_list, dataset, method, num_shot, bi_classification, requirement_index, input_requirement):
    """ construct the prompt for each single input sentence/requirement
    input
        dataset, string name of the evaluated dataset
        method, string name of the used method    
        num_shot, int, the number of few-shot examples in the prompt
        few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    output
        string_prompt: the string prompt with each corresponding input sentence/requirement.

    """

    if bi_classification:
        prompt_base = ("Please classify the given software requirements into functional requirement or non-functional requirement. "
                #"The answer should be in format {the given requirement: functional requirement or non-functional requirement}."
                "The answer should be one word, i.e. Functional or Non-functional. \n"
                )
    elif bi_classification == False:    
        prompt_base = ("Please classify the given software requirements into the following categories: "
                       "Functional, Availability, Fault Tolerance, Legal, Look and Feel, Maintainability, Operational, "
                       "Performance, Portability, Scalability, Security, Usability. "
                       #"The answer should be in format {the given requirement: the name of classified category}."
                       "The answer should be very concise and short, i.e. only one of the above-mentioned categories."
                        )
    else: 
        raise ValueError("bi_classification must be True or False.")    

    if dataset == "nfr":
        if bi_classification:
            raise ValueError("nfr does not has binary classification data'.")
        
        prompt_base = ("Please classify the given nonfunctional software requirements into the following categories: "
                       "Availability, Fault Tolerance, Maintainability, "
                       "Performance, Portability, Scalability, Security. "
                       #"The answer should be in format {the given requirement: the name of the classified category}."
                       "The answer should be very concise and short, i.e. only one of the above-mentioned categories."

                        ) 

    prompt = ""
    if num_shot == 0:
        prompt = prompt_base + "The given requirement: " + input_requirement
    elif num_shot > 0:    
        prompt = prompt_base + get_str_few_shot_examples(few_shot_list, requirement_index, dataset, method, num_shot, bi_classification) + "\nNow, classify the following given requirement: " + input_requirement
    else: 
        raise ValueError("num_shot has to be a decimal number.")    
 

    return prompt       



def get_str_few_shot_examples(few_shot_list, requirement_index, dataset, method, num_shot, bi_classification):
    """ get the part of textual few-shot examples for the prompt
    input
        few_shot_list, few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
        requirement_index, the index of the requirement sentence, which means we construct the prompt for each single test requirement sentence one by one.
        dataset, string name of the evaluated dataset
        method, string name of the used method          
        num_shot, int, the number of few-shot examples in the prompt
        few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    output
        string_few_shot_examples: the string of selected few_shot examples, which will be integrated into the formal prompt.

    """
    example_str = ""

    for i in range(num_shot):
        if bi_classification:       # the requirement                               # the category of this requirement
            example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][2] + "\n"
        else: 
            example_str = example_str + few_shot_list[requirement_index][i][0] + ': ' + few_shot_list[requirement_index][i][1] + "\n"

    return "\nBelow are some demonstration examples for you to learn, which consist of a software requirement and its category: \n" + example_str





def get_cross_validation_prompt(dataset, method, num_shot, bi_classification):
    """ get the part of textual few-shot examples for the prompt
    input
        dataset, string name of the evaluated dataset
        method, string name of the used method          
        num_shot, int, the number of few-shot examples in the prompt
        bi_classification, bool
        few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    output
        the list of prompts, string list
    """
    
    fold = 1  

    # read the processed dataset
    if dataset == "promise":
        df = pd.read_csv('processed_promise.csv' )  
    elif dataset == "pure":
        df = pd.read_csv('processed_pure.csv' )  
 
    
    # start 10x10 cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # shuffle=True to shuffle the data before splitting  
    for train_index, test_index in kf.split(df):  
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]  
        df_train = df_train.values.tolist()  
        df_test = df_test.values.tolist() 
        
        if method == "random":
            few_shot_list = get_random_few_shot_list(dataset, df_train, df_test)
        elif method == "embedding":
            few_shot_list = get_embedding_few_shot_list(dataset, df_train, df_test)
        elif method == "tfidf":
            few_shot_list = get_tfidf_few_shot_list(dataset, df_train, df_test)

        # construct the prompt based on the few-shot list
        prompt_list = get_all_prompt( few_shot_list, df_test, dataset, method, num_shot, bi_classification)
        

        if bi_classification:
            path_prompt = './cross_validation_prompt_label/' + 'prompt_' + dataset + '_' + method + '_' + str(num_shot) + '_' + str(fold) + '_' + 'bi.txt'
            path_label = './cross_validation_prompt_label/' + 'label_' + dataset + '_' + method + '_' + str(num_shot) + '_' + str(fold) + '_' + 'bi.txt'
            label_list =  [sublist[2] for sublist in df_test] 
        else:
            path_prompt = './cross_validation_prompt_label/' + 'prompt_' + dataset + '_' + method + '_' + str(num_shot) + '_' + str(fold) + '_' + 'mul.txt'
            path_label = './cross_validation_prompt_label/' + 'label_' + dataset + '_' + method + '_' + str(num_shot) + '_' + str(fold) + '_' + 'mul.txt'
            label_list =  [sublist[1] for sublist in df_test] 

        save_prompt_list(path_prompt, prompt_list)
        save_prompt_list(path_label, label_list)

        print("fold: {}".format(fold) )
        fold = fold + 1







def save_prompt_list(path, prompt_list):
    """save the constructed prompts with few-shot examples in a list 
    input
        path, string path to save the list
        prompt_list, list of constructed prompts from the first requirement/sentence in test dataset to the last one
    """
    with open(path, 'w', newline='\n') as file:  
        for i, prompt in enumerate(prompt_list):
            if i+1 == len(prompt_list):
                file.write(prompt)
            else:     
                file.write(prompt + "\n\n\n")



def read_prompt_list(path):
    """read the saved list of prompts 
    input
        path, string path to save the list
    """
    with open(path, 'r') as file:  
        content = file.read()  
    prompt_list_read = content.split('\n\n\n')   

    return prompt_list_read



def precision_recall_f1(ground_truth, predictions):  
    '''
    given two list of binary values, e.g. functional, non-functional,
    calculate their precision, recall, and F1 score respectively.
    '''

    assert len(ground_truth) == len(predictions), "The length of ground truth and predictions must be the same."  
  
    def calculate_metrics(true_positive, false_positive, false_negative):  
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0  
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0  
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0  
        return precision, recall, f1_score  
  
    # Functional (1)  
    TP_functional = sum((gt == 1 and pred == 1) for gt, pred in zip(ground_truth, predictions))  
    FP_functional = sum((gt == 0 and pred == 1) for gt, pred in zip(ground_truth, predictions))  
    FN_functional = sum((gt == 1 and pred == 0) for gt, pred in zip(ground_truth, predictions))  
      
    precision_functional, recall_functional, f1_functional = calculate_metrics(TP_functional, FP_functional, FN_functional)  
  
    # Non-functional (0)  
    TP_non_functional = sum((gt == 0 and pred == 0) for gt, pred in zip(ground_truth, predictions))  
    FP_non_functional = sum((gt == 1 and pred == 0) for gt, pred in zip(ground_truth, predictions))  
    FN_non_functional = sum((gt == 0 and pred == 1) for gt, pred in zip(ground_truth, predictions))  
      
    precision_non_functional, recall_non_functional, f1_non_functional = calculate_metrics(TP_non_functional, FP_non_functional, FN_non_functional)  
  
    return {  
        "functional": {  
            "precision": precision_functional,  
            "recall": recall_functional,  
            "f1_score": f1_functional  
        },  
        "non_functional": {  
            "precision": precision_non_functional,  
            "recall": recall_non_functional,  
            "f1_score": f1_non_functional  
        }  
    }  
                                    