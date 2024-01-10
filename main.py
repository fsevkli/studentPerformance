from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

