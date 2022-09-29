# Activity 1 - Introduction to Artificial Intelligence



Selected Data Set: [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/car+evaluation)

## Relevant Informations

- Attributes (in lowercase):
  
     ```
     CAR             car acceptability
     . PRICE         overall price
     . . buying      buying price
     . . maint       price of the maintenance
     . TECH          technical characteristics
     . . COMFORT     comfort
     . . . doors     number of doors
     . . . persons   capacity in terms of persons to carry
  . . . lug_boot  the size of luggage boot
     . . safety      estimated safety of the car
  ```
  
- Number of instances: 1728

- Number of Attributes: 6

- Attribute Values:

     ```
     buying       v-high, high, med, low
     maint        v-high, high, med, low
     doors        2, 3, 4, 5-more
     persons      2, 4, more
     lug_boot     small, med, big
     safety       low, med, high
     ```

- Class Distribution (number of instances per class):

  | Class  | Number of Instances | Number of Instances % |
  | ------ | ------------------- | --------------------- |
  | unacc  | 1210                | (70.023 %)            |
  | acc    | 384                 | (22.222 %)            |
  | good   | 69                  | ( 3.993 %)            |
  | v-good | 65                  | ( 3.762 %)            |

  

  ## Results of machine learning algorithms

  | Algorithm Name | Accuracy |
  | -------------- | -------- |
  | Naive Bayes    | 61%      |
  | Decision Tree  | 97%      |
  | Random Forest  | 97%      |
  | KNN            | 92%      |
  | SVM            | 97%      |
  | Neural Network | 97%      |