CMPS 142 - Assignment 4
Lewis Bell, Eimear Cosgrave and Carlos del Rey.
This submission consist three files: LogisticRegression.java, LogisticRegression_withBias.java and LogisticRegression_withRegularization.java.
To make the files work, they should be a part of a java project and they should be placed in a package called cmps142_hw4.

Next, we are going to clarify certain aspects of our implementation for a Logistic Regression classifier.

- General
	We have changed the type of the variables "TP", "FP", "TN" and "FN" to double to avoid problem with integer division.
- LogisticRegression.java

- LogisticRegression_withBias.java
	The bias term is included in the feature vector for each instance in the first position (e.g. [bias, feature1, feature2,..., featuren]). Also, the corresponding virtual/auxiliary feature.
- LogisticRegression_withRegularization.java