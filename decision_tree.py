from sklearn import tree
from base_model import BaseModel

class DecisionTree(BaseModel):
    def train_model(self):
        """
        Train a Decision Tree model on the given features and labels.

        Returns:
        model (DecisionTreeRegressor): The trained decision tree model.
        """
        self.model = tree.DecisionTreeRegressor(random_state=500)
        self.model.fit(self.X_train, self.y_train)
        return self.model