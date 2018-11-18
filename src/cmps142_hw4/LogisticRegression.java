package cmps142_hw4;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class LogisticRegression {

        /** the learning rate */
        private double rate=0.01;

        /** the weights to learn */
        private double[] weights;

        /** the number of iterations */
        private int ITERATIONS = 200;

        /** TODO: Constructor initializes the weight vector. Initialize it by setting it to the 0 vector. **/
        public LogisticRegression(int n) { // n is the number of weights to be learned
        	//We set the length of the weights
        	weights = new double [n];
        }

        /** TODO: Implement the function that returns the L2 norm of the weight vector **/
        private double weightsL2Norm(){
        	double sum = 0;
        	//For each weight in the vector, we add it to the power of two
        	for (int i = 0; i < weights.length; i++) {
        		sum += weights[i] * weights[i];
        	} 
        	//We return the square root of the sum of the sum of squares
        	return Math.sqrt(sum);
        }

        /** TODO: Implement the sigmoid function **/
        private static double sigmoid(double z) {
        	//We return the sigmoid for the parameter
        	return (1/( 1 + Math.pow(Math.E,(-1*z))));
        }

        /** TODO: Helper function for prediction **/
        /** Takes a test instance as input and outputs the probability of the label being 1 **/
        /** This function should call sigmoid() **/
        private double probPred1(double[] x) {
        	double dot_prod = 0;
        	//We dod the dot product of w and x for an instance
        	for(int i = 0;i < weights.length;i++) {
        		dot_prod += weights[i] * x[i];
        	}
        	//We return the sigmoid of the dot product
        	return sigmoid(dot_prod);
        }

        /** TODO: The prediction function **/
        /** Takes a test instance as input and outputs the predicted label **/
        /** This function should call probPred1() **/
        public int predict(double[] x) {
        	//If the sigmoid value for the instance is greater or equal than 0.5, we return 1.
        	if (probPred1(x) >= 0.5) {
        		return 1;
        	}
        	//If it is not, we return 0
        	return 0;
        }

        /** This function takes a test set as input, call the predict() to predict a label for it, and prints the accuracy, P, R, and F1 score of the positive class and negative class and the confusion matrix **/
        public void printPerformance(List<LRInstance> testInstances) {
            double acc = 0;
            double p_pos = 0, r_pos = 0, f_pos = 0;
            double p_neg = 0, r_neg = 0, f_neg = 0;
            int TP=0, TN=0, FP=0, FN=0; // TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives

            // TODO: write code here to compute the above mentioned variables
            //For each instance
            for (LRInstance instance: testInstances) {
            	int prediction = predict(instance.x);
            	//If the label is 1
            	if (instance.label == 1) {
            		//And the predicted label is equal to the actual label
					if (prediction == instance.label) {
						//We add a True positive
						TP++;
						//If not
					} else {
						//We add a False positive
						FN++;
					}
				//If the label is 0
				} else {
					//And the predicted label is equal to the actual label
					if (prediction == instance.label) {
						//We add a True negative
						TN++;
					} else {
						//We add a False negative
						FP++;
					}
				}
            }
            //Once we have checked all the instances we calculate some values:
            //Accuracy
            acc = (TN + TP)/(TP + TN + FP + FN);
            //Precision of positives
            p_pos = TP/(TP + FP);
            //Precision of negatives
            p_neg = TN/(TN + FN);
            //Recall of positives
            r_pos = TP/(TP + FN);
            //Recall of negatives
            r_neg = TN/(TN + FP);
            //F score of positives
            f_pos = (2 * p_pos * r_pos)/(p_pos + r_pos);
            //F score of negatives
            f_neg = (2 * p_neg * r_neg)/(p_neg + r_neg);
            
            System.out.println("Accuracy="+acc);
            System.out.println("P, R, and F1 score of the positive class=" + p_pos + " " + r_pos + " " + f_pos);
            System.out.println("P, R, and F1 score of the negative class=" + p_neg + " " + r_neg + " " + f_neg);
            System.out.println("Confusion Matrix");
            System.out.println(TP + "\t" + FN);
            System.out.println(FP + "\t" + TN);
        }


        /** Train the Logistic Regression using Stochastic Gradient Ascent **/
        /** Also compute the log-likelihood of the data in this function **/
        public void train(List<LRInstance> instances) {
            for (int n = 0; n < ITERATIONS; n++) {
                double lik = 0.0; // Stores log-likelihood of the training data for this iteration
                for (int i=0; i < instances.size(); i++) {
                	// TODO: Train the model
                	for(int j = 0;j < weights.length;j++) {
                		weights[j] = weights[j] + (rate * instances.get(i).x[j] * (instances.get(i).label - probPred1(instances.get(i).x)));
                	}
                    // TODO: Compute the log-likelihood of the data here. Remember to take logs when necessary
                	double dot_prod = 0;
                	for(int k = 0;k < weights.length;k++) {
                		dot_prod += weights[k] * instances.get(i).x[k];
                	}
                	lik += instances.get(i).label * dot_prod - Math.log(1 + Math.pow(Math.E,(dot_prod)));
				}
                System.out.println("iteration: " + n + " lik: " + lik);
            }
        }

        public static class LRInstance {
            public int label; // Label of the instance. Can be 0 or 1
            public double[] x; // The feature vector for the instance

            /** TODO: Constructor for initializing the Instance object **/
            public LRInstance(int label, double[] x) {
            	this.label = label;
            	this.x = x.clone();
            }
        }

        /** Function to read the input dataset **/
        public static List<LRInstance> readDataSet(String file) throws FileNotFoundException {
            List<LRInstance> dataset = new ArrayList<LRInstance>();
            Scanner scanner = null;
            try {
                scanner = new Scanner(new File(file));

                while(scanner.hasNextLine()) {
                    String line = scanner.nextLine();
                    if (line.startsWith("ju")) { // Ignore the header line
                        continue;
                    }
                    String[] columns = line.replace("\n", "").split(",");

                    // every line in the input file represents an instance-label pair
                    int i = 0;
                    double[] data = new double[columns.length - 1];
                    for (i=0; i < columns.length - 1; i++) {
                        data[i] = Double.valueOf(columns[i]);
                    }
                    int label = Integer.parseInt(columns[i]); // last column is the label
                    LRInstance instance = new LRInstance(label, data); // create the instance
                    dataset.add(instance); // add instance to the corpus
                }
            } finally {
                if (scanner != null)
                    scanner.close();
            }
            return dataset;
        }


        public static void main(String... args) throws FileNotFoundException {
            List<LRInstance> trainInstances = readDataSet("HW4_trainset.csv");
            List<LRInstance> testInstances = readDataSet("HW4_testset.csv");

            // create an instance of the classifier
            int d = trainInstances.get(0).x.length;
            LogisticRegression logistic = new LogisticRegression(d);

            logistic.train(trainInstances);

            System.out.println("Norm of the learned weights = "+logistic.weightsL2Norm());
            System.out.println("Length of the weight vector = "+logistic.weights.length);

            // printing accuracy for different values of lambda
            System.out.println("-----------------Printing train set performance-----------------");
            logistic.printPerformance(trainInstances);

            System.out.println("-----------------Printing test set performance-----------------");
            logistic.printPerformance(testInstances);
        }

    }
