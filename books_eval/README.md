## Search Recall Evaluation with Azure Cosmos DB and GraphRAG

A public dataset based on unstrcutured data across various book cateogies of books, wines and new articles was ingested into Azure Cosmos DB Vector Database. A graph based retrieval was built based on [GraphRAG](https://github.com/microsoft/graphrag). The recall evaluation was done for both global and local searches. The evaluation was done on 8 different questions with a total of 16K book files and 1K noise files. The evaluation was done using the following metrics:


## Evaluation Results

### Question 0
**Question**: Give me a list of books published in the year 2000  
- **Missing Files**: 92  
- **Recall**: 71.78%  
- **Precision**: 61.26%  
- **F1 Score**: 66.10%

---

### Question 1
**Question**: Give me a list of book of travel category  
- **Missing Files**: 26  
- **Recall**: 36.59%  
- **Precision**: 20.00%  
- **F1 Score**: 25.86%

---

### Question 2
**Question**: Give me a list of books from author Agatha Christie  
- **Missing Files**: 7  
- **Recall**: 81.08%  
- **Precision**: 40.54%  
- **F1 Score**: 54.05%

---

### Question 3
**Question**: Give me some haunted incidents from California state  
- **Missing Files**: 18  
- **Recall**: 79.55%  
- **Precision**: 70.71%  
- **F1 Score**: 74.87%

---

### Question 4
**Question**: Give me some wines found in Italy  
- **Missing Files**: 15  
- **Recall**: 91.71%  
- **Precision**: 46.24%  
- **F1 Score**: 61.48%

---

### Question 5
**Question**: Give me wines tasted by Roger Voss  
- **Missing Files**: 86  
- **Recall**: 12.24%  
- **Precision**: 6.12%  
- **F1 Score**: 8.16%

---

### Question 6
**Question**: Give me some wines in the variety of Red Blend  
- **Missing Files**: 72  
- **Recall**: 84.07%  
- **Precision**: 54.68%  
- **F1 Score**: 66.26%

---

### Question 7
**Question**: Give me some business category news  
- **Missing Files**: 285  
- **Recall**: 43.11%  
- **Precision**: 43.20%  
- **F1 Score**: 43.16%

---

### Question 8
**Question**: Give me a list of students graduated in the year 2025  
- **Missing Files**: 0  
- **Recall**: 100.00%  
- **Precision**: 72.73%  
- **F1 Score**: 84.21%
