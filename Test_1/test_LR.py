import sys
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
import pickle
import time
import statistics
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import os
import fnmatch
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import LabelEncoder
    
def main(percentage):
    print("percentage received:", percentage, flush=True)
    # Convert the percentage to an integer
    try:
        percentage_int = int(percentage)
        print("percentage as an integer:", percentage_int, flush=True)
    except ValueError:
        print("percentage is not a valid integer.", flush=True)
        
    test_path = "./Train_" + percentage + "/test_datasets/"    
    num_files = 0
    num_files = len(fnmatch.filter(os.listdir(test_path), 'test_fold_*.csv'))
    print('Test file count:', num_files)
    num_files += 1         

    
    # Initialize lists to store evaluation metrics
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    accuracies = []
    time_list = [] 
    
    output = "./Train_" + percentage + "/models/LR/Results/testing_numeralia.txt"
    with open(output, "w") as file:     
         file.write("Testing data\n")
         
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep'])                
    
    # Iterate through each fold
    for i in range(1, num_files):
        print("Fold: ", i, flush = True)        
        # Load test data for this fold
        filename = "./Train_" + percentage + "/test_datasets/test_fold_{}.csv".format(i)
        data = pd.read_csv(filename)
        data = data[["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]]
        y_true = data['action']
        y_true = encoder.transform(y_true) 
        X_test = data.drop(columns=['action'])      

        # Load the model
        filename = "./Train_" + percentage + "/models/LR/LR_{}.lr".format(i)
        model = pickle.load(open(filename, 'rb')) 
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        end_time = time.time()

        testing_time = end_time - start_time
        #print("testing time:", testing_time, flush = True)
        time_list.append(testing_time) 
        
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        # Append metrics to lists
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        supports.append(support)
        accuracies.append(accuracy)
        
        with open(output, "a") as file:     
           # Confusion matrix
           string = "Fold : " + str(i) + "\n" + str(confusion_matrix(y_true, y_pred)) + "\n"
           file.write(string)
           string = str(classification_report(y_true, y_pred)) + "\n"
           file.write(string)           

    # Calculate average metrics
    average_precision = statistics.mean(precisions)
    stdev_precision = statistics.stdev(precisions)
    average_recall = statistics.mean(recalls)
    stdev_recall = statistics.stdev(recalls)
    average_f1 = statistics.mean(f1_scores)
    stdev_f1 = statistics.stdev(f1_scores)
    average_accuracy = statistics.mean(accuracies)
    stdev_accuracy = statistics.stdev(accuracies)
    average_time = statistics.mean(time_list)
    stdev_time = statistics.stdev(time_list)
    
    with open(output, "a") as file:        

      string = 'Precisions: ' + str(precisions) + "\n"
      file.write(string) 
      string = 'Recalls: ' + str(recalls) + "\n"
      file.write(string) 
      string = 'F1_scores: ' + str(f1_scores) + "\n"
      file.write(string) 
      string = 'Accuracies: ' + str(accuracies) + "\n"
      file.write(string) 

      string = "Average Precision: " + str(average_precision) + "\nstd dev Precision: " + str(stdev_precision) + "\n"
      file.write(string) 
      
      string = "Average Recall: " + str(average_recall) + "\nstd dev recall: " + str(stdev_recall) + "\n"
      file.write(string)       
      
      string = "Average F1-score: " + str(average_f1) + "\nstd dev F1-score:" + str(stdev_f1) + "\n"
      file.write(string)   
      
      string = "Average Accuracy: " + str(average_accuracy) + "\nstd dev Accuracy: " + str(stdev_accuracy) + "\n"
      file.write(string)                
            
      string = "Avg. testing time: " +  str(average_time) + "\n"
      file.write(string)    
      string = "Stdev. testing time: " + str(stdev_time) + "\n"
      file.write(string) 
      

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_LR.py <percentage>")
        sys.exit(1)
    percentage = sys.argv[1]
    main(percentage)
    
    
    
