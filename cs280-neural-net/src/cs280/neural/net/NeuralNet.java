package cs280.neural.net;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *  Artificial Neural Network - Backpropagation Algorithm
 *  @author Isabelle Tingzon, MS CS
 *  Department of Computer Science
 *  University of the Philippines - Diliman
 *  Built in Netbeans IDE using Java Compiler
 *  OS: Windows 8.0
 */

public final class NeuralNet {
    
    //Define data dimension
    private final int datasize = 3486;
   
    //Data partitions 
    private int part_train = datasize;
    private int part_test = 0;
    
    //Define Architecture of neural network
    private final static int n_in = 354; //input
    private static int n_h1 = 10; //hidden layer 1
    private static int n_h2 = 5; //hidden layer 2
    private final static int n_out = 8; //output
    private static double eta = 0.5; //learning rate
    
    // Pre-allocate storage for weights and biases
    private double[] x_in = new double[n_in];
    private double[][] w_h1 = new double[n_h1][n_in];
    private double[] b_h1 = new double[n_h1];
    private double[][] w_h2 = new double[n_h2][n_h1];
    private double[] b_h2 = new double[n_h2];
    private double[][] w_out = new double[n_out][n_h2];
    private double[] b_out = new double[n_out];
    private double[] d_out = new double[n_out];
        
    public NeuralNet() throws IOException{}
        
    @SuppressWarnings("empty-statement")
        
    // K-fold Cross Validation
    // Finds the average error and accuracy over k rounds of partitioning
    public void crossValidate() throws IOException{
        int kinit = 3;
        int kmax = 10;
        
        //int[] candidate_hnodes = {5, 10 ,20, 30, 40, 50, 60, 70, 80, 90, 100};
        //double[] candidate_eta = {0.05, 0.1, 0.5, 1.0};
        //int max = candidate_eta.length;
        //int max = candidate_hnodes.length;
        
        int max = 1;
        double [] acc = new double[max];
        double [] err = new double[max];
        double [] eta_ = new double[max];
              
        for (int iter = 0; iter < max; iter++){
            
            //increment no. of nodes in hidden layers
            //n_h1 = candidate_hnodes[iter]; 
            //n_h2 = candidate_hnodes[iter];
            
            //increment learning rate
            //eta = candidate_eta[iter]; 
           
            double mean_accuracy = 0;    
            double mean_error = 0;       
            
            for (int k = kinit; k < kmax + 1; k++){
                this.initializeWeights();
                                
                System.out.println("K = " + k);
                double kfold = (double)(k-1)/k;
                this.partitionData(kfold, true);
                List res = this.train(false);
                
                //Writes epoch, training error, validation error to file
                /*List epoch = (List) res.get(0);
                List train_err = (List) res.get(1);
                List val_err = (List) res.get(2);
                this.errorplot(epoch, train_err, val_err, k);*/
                                
                String val_file = "dataset/validation_set.csv";
                String val_label = "dataset/validation_labels.csv";
                
                List<double[][]> vlist = getData(val_file, val_label, part_test);
                List result = this.test(vlist, false);
                mean_accuracy = mean_accuracy + (double)result.get(1);
                mean_error = mean_error + (double)result.get(0);
            }
            
            acc[iter] = mean_accuracy/((double)(kmax + 1) - (double)kinit);
            err[iter] = mean_error/((double)(kmax + 1) - (double)kinit);
            eta_[iter] = eta;
            
            System.out.println("n_h1=" + n_h1 + " nh_2=" + n_h2 + " eta=" + eta);
            System.out.println("Average Accuracy: " + acc[iter] );
            System.out.println("Average Error: " + err[iter] );
        }       
        
        //Print error results
        /*double[] sorted = this.sortArray(err);
        System.out.println();
        for (int i = 0; i < err.length; i++){
            System.out.print(err[i] + "  ");
        }  
        //sorted = this.sortArray(acc);
        //Print accuracy results
        System.out.println();
        for (int i = 0; i < acc.length; i++){
            System.out.print(acc[i] + "  ");
        } */        
    }  
    
    // Writes epoch, training error, validation error (per iteration) to file (for each k partition)
    // The file is later graphed in Python using matplotlib
    public void errorplot(List epoch, List train, List val, int k){
        int size = epoch.size();
                
        FileWriter writer = null;
        try{
            writer = new FileWriter("dataset/" + Integer.toString(k) + "_error_plot.csv");            
            for (int i = 0; i < size; i++) {
                writer.append(Integer.toString((int)epoch.get(i)) + "," + 
                       Double.toString((double)train.get(i)) + "," +
                       Double.toString((double)val.get(i)) + "\n");
            }
        }catch (Exception e) {
	} finally {
            try {
                writer.flush();
		writer.close();
            } catch (IOException e) {
            }
	}
        
    }
    
    // Sorts an array of errors (or accuracies) from least to greatest 
    public double[] sortArray(double[] arr){
        int[] sorted = new int[arr.length];
        for (int i = 0; i < sorted.length; i++){
            sorted[i] = i;
        }
        for (int m = arr.length; m >= 0; m--) {
            for (int i = 0; i < arr.length - 1; i++) {
                int j = i + 1;
                if (arr[i] > arr[j]) {
                    double temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        return arr;
    }    
    
    // Classifies data set test_set.csv
    public void testSet() throws IOException{      
        String test_file = "dataset/test_set.csv";
        int test_len = 701;
        part_test = test_len;
        
        double[][] X = this.readData(test_file, test_len, n_in);
        double[][] Y = null;
        List<double[][]> list = new ArrayList<>();
        list.add(X);
        list.add(Y);
        
        //this.train(true);
        this.test(list, true);
    }
    
    // Writes predicted labels for test_set.csv to predicted_ann.csv
    public void writeTestLabel(double[][] output) throws IOException{
        int N = part_test;        
        FileWriter labelWriter = new FileWriter("dataset/predicted_ann.csv");
        try{                        
            for (int i = 0; i < N; i++) {
                double sum = 0;
                for (int index = 0; index < n_out; index++){
                    sum = sum + output[i][index];
                    if (output[i][index] == 1.0){
                        labelWriter.append(Integer.toString(index + 1) + "\n");
                    }
                }
                // classifier did not classify the sample ( i.e. y is zero vector)
                if (sum == 0){  
                    labelWriter.append(sum + "\n");
                }
            }
        }catch (Exception e) {
	} finally {
            try {
                labelWriter.flush();
		labelWriter.close();
            } catch (IOException e) {
            }
	}
    }
    
    // Reads data from .csv file 
    // Input: len = no. of instances per data set
    //        size = no. of attributes/features (354); for labels, size = no of classes (8)
    public double[][] readData(String filename, int len, int size) throws FileNotFoundException, IOException{
        BufferedReader dataReader = new BufferedReader(new FileReader(filename));
        double[][] data = new double[len][size];
        
        int c = 0;
        String line = dataReader.readLine();
        while(line != null){
            String[] inputArr = line.replace("\n","").split(",");
            if (size == n_out){
                for(int i = 0; i < size; i++){
                    if (i == (Integer.parseInt(inputArr[0]) - 1))
                        data[c][i] = 1;
                    else
                        data[c][i] = 0;
                }
            }
            else if (size == n_in){
                for(int i = 0; i < size; i++){
                    data[c][i] = Double.parseDouble(inputArr[i]);
                }
            }
            c++;
            line = dataReader.readLine();
        }
        return data;
    }
    
    //Initializes values for weights + biases
    public void initializeWeights(){
        Random generator = new Random();   
        w_h1 = new double[n_h1][n_in];
        b_h1 = new double[n_h1];
        for (int i = 0; i < n_h1; i++){ 
            b_h1[i] = -0.1+(0.1+0.1)*generator.nextDouble();
            for(int j = 0; j < n_in; j++){
                w_h1[i][j] = -0.1+(0.1+0.1)*generator.nextDouble(); 
                x_in[j] = 0; 
            }
        }
        w_h2 = new double[n_h2][n_h1];
        b_h2 = new double[n_h2];
        for (int i = 0; i < n_h2; i++){
            b_h2[i] = -0.1+(0.1+0.1)*generator.nextDouble();
            for(int j = 0; j < n_h1; j++){
                w_h2[i][j] = -0.1+(0.1+0.1)*generator.nextDouble();
            }
        }
        w_out = new double[n_out][n_h2];
        for (int i = 0; i < n_out; i++){
            b_out[i] = -0.1+(0.1+0.1)*generator.nextDouble();
            d_out[i] = 0;
            for(int j = 0; j < n_h2; j++){
                w_out[i][j] = -0.1+(0.1+0.1)*generator.nextDouble();
            }
        }
    }
    
    //random permute using Fisher-Yates Shuffle (integer arrays)
    public int[] shuffle(int[] arr){
        Random rnd = new Random();
        for (int i = arr.length - 1; i > 0; i--){
            int index = rnd.nextInt(i + 1);
            int a = arr[index];
            arr[index] = arr[i];
            arr[i] = a;
        }
        return arr;
    }
    
    // Random permute using Fisher-Yates Shuffle (String arrays)
    public List shuffle(String[] data, String[] label){
        Random rnd = new Random();
        for (int i = data.length - 1; i > 0; i--){
            int index = rnd.nextInt(i + 1);
            //swap data
            String a = data[index];
            data[index] = data[i];
            data[i] = a;
            //swap label
            String b = label[index];
            label[index] = label[i];
            label[i] = b;
        }
        List list = new ArrayList();
        list.add(data);
        list.add(label);
        return list;
    }
    
    // K-fold partitioning of data - for Cross Validation
    // Creates training_set.csv, validationset.csv, training_labels.csv, validation_labels.csv per partition
    // Input: k = number of partitions
    //        is_cross_val = specifies whether or not partitioning is necessary 
    //                       (i.e. no need to partition data to classify the test set test_set.csv
    //                        the whole of data.csv will be used as the training set)
    public void partitionData(double kfold, boolean is_cross_val) throws IOException{
        String data_file = "dataset/data.csv";
        String label_file = "dataset/data_labels.csv";
        
        // Data partition
        if (is_cross_val == true){   
            part_train = (int)(datasize*kfold);
            part_test = datasize - part_train;
        }
        
        //Shuffle initial data.csv
        String[] data_set = new String[datasize];
        String[] label_set = new String[datasize];
        BufferedReader trainReader = new BufferedReader(new FileReader(data_file));
        BufferedReader labelReader = new BufferedReader(new FileReader(label_file));
        for (int i = 0; i < datasize; i++) {
            String data = trainReader.readLine().trim() + "\n";
            String label = labelReader.readLine().trim() + "\n";
            data_set[i] = data;
            label_set[i] = label;
        }
        List l = this.shuffle(data_set, label_set);
        data_set = (String[]) l.get(0);
        label_set = (String[]) l.get(1);
        
        FileWriter trainWriter = null;
        FileWriter validWriter = null;
        try {
            trainWriter = new FileWriter("dataset/training_set.csv");
            validWriter = new FileWriter("dataset/validation_set.csv");
            //BufferedReader trainReader = new BufferedReader(new FileReader(data_file));        
            for (int i = 0; i < datasize; i++) {
                if (i < part_train){
                    trainWriter.append(data_set[i]);
                }
                else{
                    validWriter.append(data_set[i]);
                }
            }
        }catch (Exception e) {
	} finally {
            try {
                trainWriter.flush();
		trainWriter.close();
                validWriter.flush();
		validWriter.close();
            } catch (IOException e) {
            }
	}
        
        FileWriter labelWriter = null;
        FileWriter vlabelWriter = null;
        try{
            labelWriter = new FileWriter("dataset/training_labels.csv");
            vlabelWriter = new FileWriter("dataset/validation_labels.csv");
            //BufferedReader labelReader = new BufferedReader(new FileReader(label_file));
            for (int i = 0; i < datasize; i++) {
                //String label = labelReader.readLine().trim() + "\n";
                if (i < part_train){
                    labelWriter.append(label_set[i]);
                }
                else{
                    vlabelWriter.append(label_set[i]);
                }
            }
        }catch (Exception e) {
	} finally {
            try {
                labelWriter.flush();
		labelWriter.close();
                vlabelWriter.flush();
		vlabelWriter.close();
            } catch (IOException e) {
            }
	}
    }
  
    // Returns X (features) and Y (labels) components of the data 
    // Input:  data_file = file name of the file containing features (.csv)
    //         label_file = file name of the file containing labels (.csv)
    //         size = size of the data set
    // Output: List = [X, Y] 
    //         X = attributes/ features
    //         Y = labels
    public List<double[][]> getData(String data_file, String label_file, int size) throws IOException{        
        double[][] X = this.readData(data_file, size, n_in);
        double[][] Y = this.readData(label_file, size, n_out);
        List<double[][]> list = new ArrayList<>();
        list.add(X);
        list.add(Y);
        return list;
    }
    
    
    public List<double[][]> randomHybridResample(List<double[][]> data_set){
        double[][] X = data_set.get(0);
        double[][] Y = data_set.get(1);
        
        int target_samp = (int) Y.length/n_out;        
        
        //Set initial frequencies to zero
        int [] freq = new int[n_out];
        for (int i = 0; i< n_out; i++){
           freq[i] = 0;
        }        
        // Construct a list of instances per class (ipc)
        List<List> ipc = new ArrayList(n_out);
        for (int i = 0; i < n_out; i++){
            ipc.add(null);
        }        
        // Initialize ipc set
        for (int i = 0; i < n_out; i++){
            ipc.set(i, new ArrayList());
        }
        
        // Associates a group of instances to a particular class
        //   (i.e. a pool of instances is created per class label)
        for (int i = 0; i < Y.length; i++){
            for (int index = 0; index < n_out; index ++){
                if (Y[i][index] == 1){
                    freq[index] = freq[index] + 1;
                    List l = ipc.get(index); //index = class label
                    List xy = new ArrayList(2); 
                    xy.add(X[i]); 
                    xy.add(Y[i]);
                    l.add(xy); // add instance to class
                }
            }
        }
        
        int max_class = 0;
        int max_freq = 0;
        for (int i = 0; i < n_out; i++){
            if (max_freq < freq[i]){
                max_freq = freq[i];
                max_class = i;
            }
        }
        
        //System.out.println("Max freq = " + max_freq + " max class=" + max_class);
        
        double[][] new_X = new double[X.length][X[0].length];
        double[][] new_Y = new double[Y.length][Y[0].length];
        
        // Constructs the new sample by oversampling/ undesampling 
        //      from the pool of instances (per class label)
        int ind = 0;
        for (int i = 0; i < n_out; i++){ //for each class
            List obj = ipc.get(i);
            if (obj.toArray().length > target_samp){
                for (int k = 0; k < target_samp; k++){
                    List xy = (List) obj.get(k);
                    new_X[ind] = (double[]) xy.get(0);
                    new_Y[ind] = (double[]) xy.get(1);
                    ind = ind + 1;
                }
            }
            else{
                int j = 0;
                while (j < target_samp){
                    Random randomGenerator = new Random();
                    int randomInt = randomGenerator.nextInt(obj.toArray().length);
                    List xy = (List) obj.get(randomInt); 
                    new_X[ind] = (double[]) xy.get(0);
                    new_Y[ind] = (double[]) xy.get(1);
                    ind = ind + 1;
                    j = j + 1;
                }
            }
        } 
        
        freq = new int[n_out];
        for (int i = 0; i < new_Y.length; i++){
            for (int index = 0; index < n_out; index ++){
                if (new_Y[i][index] == 1){
                    freq[index] = freq[index] + 1;
                }
            }
        }
       
        //Print new proportions (should be equal for all classes)
        double total = 0;
        for (int i = 0; i < n_out; i++){
            System.out.print("Freq " + freq[i] + " ");
            double prop = ((double)(freq[i])/(new_Y.length))*100;
            System.out.print((i+1) + ".) " + prop + " ");
            total = total + prop;
        }
        System.out.println();
        System.out.println(total);
        
        List<double[][]> ret= new ArrayList<>();
        ret.add(new_X);
        ret.add(new_Y);
        return ret;
    }
    
    // Random Oversampling of the data set - for handing unbalanced data set
    // Goal: to have a proportional number of samples in all classes
    public List<double[][]> randomOversampling(List<double[][]> data_set){
        double[][] X = data_set.get(0);
        double[][] Y = data_set.get(1);
                
        //Set initial frequencies to zero
        int [] freq = new int[n_out];
        for (int i = 0; i< n_out; i++){
           freq[i] = 0;
        }        
        // Construct a list of instances per class (ipc)
        List<List> ipc = new ArrayList(n_out);
        for (int i = 0; i < n_out; i++){
            ipc.add(null);
        }        
        // Initialize ipc set
        for (int i = 0; i < n_out; i++){
            ipc.set(i, new ArrayList());
        }
        
        // Associates a group of instances to a particular class
        //   (i.e. a pool of instances is created per class label)
        for (int i = 0; i < Y.length; i++){
            for (int index = 0; index < n_out; index ++){
                if (Y[i][index] == 1){
                    freq[index] = freq[index] + 1;
                    List l = ipc.get(index); //index = class label
                    List xy = new ArrayList(2); 
                    xy.add(X[i]); 
                    xy.add(Y[i]);
                    l.add(xy); // add instance to class
                }
            }
        }
        
        int max_class = 0;
        int max_freq = 0;
        for (int i = 0; i < n_out; i++){
            if (max_freq < freq[i]){
                max_freq = freq[i];
                max_class = i;
            }
        }
        
        //System.out.println("Max freq = " + max_freq + " max class=" + max_class);
        
        double[][] new_X = new double[max_freq*n_out + 1][X[0].length];
        double[][] new_Y = new double[max_freq*n_out + 1][Y[0].length];
        
        // Constructs the new sample by oversampling/ undesampling 
        //      from the pool of instances (per class label)
        int ind = 0;
        for (int i = 0; i < n_out; i++){ //for each class
            List obj = ipc.get(i);
            int j = 0;
            for (int k = 0; k < obj.toArray().length; k++){
                List xy = (List) obj.get(k);
                new_X[ind] = (double[]) xy.get(0);
                new_Y[ind] = (double[]) xy.get(1);
                ind = ind + 1;
                j = j + 1;
            }
            while (j < max_freq){
                Random randomGenerator = new Random();
                int randomInt = randomGenerator.nextInt(obj.toArray().length);
                List xy = (List) obj.get(randomInt); 
                new_X[ind] = (double[]) xy.get(0);
                new_Y[ind] = (double[]) xy.get(1);
                ind = ind + 1;
                j = j + 1;
            }
        } 
        
        freq = new int[n_out];
        for (int i = 0; i < new_Y.length; i++){
            for (int index = 0; index < n_out; index ++){
                if (new_Y[i][index] == 1){
                    freq[index] = freq[index] + 1;
                }
            }
        }
       
        //Print new proportions (should be equal for all classes)
        /*double total = 0;
        for (int i = 0; i < n_out; i++){
            System.out.print("Freq " + freq[i] + " ");
            double prop = ((double)(freq[i])/(new_Y.length))*100;
            System.out.print((i+1) + ".) " + prop + " ");
            total = total + prop;
        }
        System.out.println();
        System.out.println(total);*/
        
        List<double[][]> ret= new ArrayList<>();
        ret.add(new_X);
        ret.add(new_Y);
        return ret;
    }
    
    // Training Phase with Backpropagation Algorithm
    // Parameters:   istestset - boolean (passed to function test()) 
    //                  necessary for monitoring prediction error
    // Returns list containing the epoch, training error, and validation errors per iteration
    public List train(boolean istestset) throws IOException{
        int max_epoch = 30000;
        
        //Set training set data
        String train_file = "dataset/training_set.csv";
        String label_file = "dataset/training_labels.csv";
        List<double[][]> list = getData(train_file, label_file, part_train);
        // Resample highly unbalanced data
        list = this.randomOversampling(list);
        //list = this.randomHybridResample(list);
        double[][] X = list.get(0);
        double[][] Y = list.get(1);
        int N = X.length;
        
                
        double[] totalerr = new double[max_epoch];
        for (int i =0; i< max_epoch; i++){
            totalerr[i] = 0;
        }
        
        List epoch_ = new ArrayList();
        List train_errors = new ArrayList();
        List val_errors = new ArrayList();
        
        int q;
        for (q = 0; q < max_epoch; q++){
            int[] p = new int[N];
            for (int i = 0; i< N; i++){
                p[i] = i;
            }
            //shuffle patterns
            p = this.shuffle(p); 
            double[] err = new double[n_out];
            for (int n = 0; n < N; n++){
                int nn = p[n];
                x_in = X[nn];
                d_out = Y[nn];
                //Forward Pass
                // hidden layer 1
                double[] v_h1 = new double[n_h1];
                double[] y_h1 = new double[n_h1];
                for (int i = 0; i < n_h1; i++){
                    v_h1[i] = 0;
                    for (int j = 0; j < n_in; j++){
                        v_h1[i] = v_h1[i] + w_h1[i][j]*x_in[j];
                    }
                    v_h1[i] = v_h1[i] + b_h1[i];
                    y_h1[i] = 1/(1 + Math.exp(-v_h1[i]));
                }
                //hidden layer 2
                double[] v_h2 = new double[n_h2];
                double[] y_h2 = new double[n_h2];
                for (int i = 0; i < n_h2; i++){
                    v_h2[i] = 0;
                    for (int j = 0; j < n_h1; j++){
                        v_h2[i] = v_h2[i] + w_h2[i][j]*y_h1[j];
                    }
                    v_h2[i] = v_h2[i] + b_h2[i];
                    y_h2[i] = 1/(1 + Math.exp(-v_h2[i]));
                }
                //output layer
                double[] v_out = new double[n_out];
                double[] out = new double[n_out];
                for (int i = 0; i < n_out; i++){
                    v_out[i] = 0;
                    for (int j = 0; j < n_h2; j++){
                        v_out[i] = v_out[i] + w_out[i][j]*y_h2[j];
                    }
                    v_out[i] = v_out[i] + b_out[i];
                    out[i] = 1/(1 + Math.exp(-v_out[i]));
                }
                //Error Propagation
                double[] delta_out = new double[n_out];
                //compute gradient in output layer
                for (int i = 0; i < n_out; i++){
                    err[i] = d_out[i] - out[i];
                    delta_out[i] = err[i]*out[i]*(1 - out[i]);
                }
                //compute gradient in hidden layer 2
                double[] delta_h2 = new double[n_h2];
                double[] temp = new double[n_h2];
                for (int i = 0; i < n_h2; i++){
                    temp[i] = 0;
                    for (int j = 0; j < n_out; j++){
                        temp[i] = temp[i] + w_out[j][i]*delta_out[j];
                    }
                }
                for (int i = 0; i < n_h2; i++){                    
                    delta_h2[i] = y_h2[i]*(1-y_h2[i])*(temp[i]);
                }
                //compute gradient in hidden layer 1
                double[] delta_h1 = new double[n_h1];
                double[] temp2 = new double[n_h1];
                for (int i = 0; i < n_h1; i++){
                    temp2[i] = 0;
                    for (int j = 0; j < n_h2; j++){
                        temp2[i] = temp2[i] + w_h2[j][i]*delta_h2[j];
                    }
                }
                for (int i = 0; i < n_h2; i++){                    
                    delta_h1[i] = y_h1[i]*(1-y_h1[i])*(temp2[i]);
                }
                //update weights and biases in output layer
                for (int i =0; i < n_out; i++){
                    for (int j= 0; j < n_h2; j++){
                        w_out[i][j] = w_out[i][j] + eta*delta_out[i]*y_h2[j];
                    }  
                    b_out[i] = b_out[i] + eta*delta_out[i];
                }
                //update weights and biases in hidden layer 2
                for (int i =0; i < n_h2; i++){
                    for (int j= 0; j < n_h1; j++){
                        w_h2[i][j] = w_h2[i][j] + eta*delta_h2[i]*y_h1[j];
                    }  
                    b_h2[i] = b_h2[i] + eta*delta_h2[i];
                }
               //update weights and biases in hidden layer 1
                for (int i =0; i < n_h1; i++){
                    for (int j= 0; j < n_in; j++){
                        w_h1[i][j] = w_h1[i][j] + eta*delta_h1[i]*x_in[j];
                    }  
                    b_h1[i] = b_h1[i] + eta*delta_h1[i];
                }
            }
            double sum = 0;
            for (int i = 0; i < n_out; i++){
                sum = sum + err[i]*err[i]; 
            }
            totalerr[q] = totalerr[q] + sum;
            if ((q % 10) == 0) {
                System.out.println("Iteration: " + q + " Training Error: " + totalerr[q]);
                if (istestset == false){
                    epoch_.add(q);
                    train_errors.add(totalerr[q]);
                    //Monitor validation errors
                    String val_file = "dataset/validation_set.csv";
                    String val_label = "dataset/validation_labels.csv";
                    List<double[][]> vlist = getData(val_file, val_label, part_test);
                    List vres =  this.test(vlist, istestset);
                    val_errors.add(vres.get(0)); 
                }
            }
            if(totalerr[q] < 0.0001){
                break;
            }
        }
        //System.out.println("Total epochs: " + q);
        //System.out.println("Network Error at termination:" + totalerr[q]);
        
        //Returns list = [epoch, training errors, validation errors]
        List ret = new ArrayList();
        ret.add(epoch_);
        ret.add(train_errors);
        ret.add(val_errors);
        return ret;
    }
    
    // Testing Phase 
    // Parameters: list - contains the X (features) and Y (labels) componenets of the data set
    //              istestset - boolean; 
    //                  if true, writes to predicted_ann.csv
    //                  if false, returns prediction error and accuracy
    public List test(List<double[][]> list, boolean istestset) throws IOException{
        int N = part_test;
        double[][] X = list.get(0);
        double[][] nn_output = new double[N][n_out];
        double[][] output = new double[N][n_out];
        
        for (int n = 1; n < N; n++){
            double[] x_in = X[n];
            //hidden layer 1
            double[] v_h1 = new double[n_h1];
            double[] y_h1 = new double[n_h1];
            for (int i = 0; i < n_h1; i++){
                v_h1[i] = 0;
                for (int j = 0; j < n_in; j++){
                    v_h1[i] = v_h1[i] + w_h1[i][j]*x_in[j];
                }
                v_h1[i] = v_h1[i] + b_h1[i];
                y_h1[i] = 1/(1+Math.exp(-v_h1[i]));
            }
            //hidden layer 2
            double[] v_h2 = new double[n_h2];
            double[] y_h2 = new double[n_h2];
            for (int i = 0; i < n_h2; i++){
                v_h2[i] = 0;
                for (int j = 0; j < n_h1; j++){
                    v_h2[i] = v_h2[i] + w_h2[i][j]*y_h1[j];
                }
                v_h2[i] = v_h2[i] + b_h2[i];
                y_h2[i] = 1/(1+Math.exp(-v_h2[i]));
            }
            //output layer
            double[] v_out = new double[n_out];
            double[] out = new double[n_out];
            for (int i = 0; i < n_out; i++){
                v_out[i] = 0;
                for (int j = 0; j < n_h2; j++){
                    v_out[i] = v_out[i] + w_out[i][j]*y_h2[j];
                }
                v_out[i] = v_out[i] + b_out[i];
                out[i] = 1/(1+Math.exp(-v_out[i]));
            }
            for (int i = 0; i < n_out; i++){
                nn_output[n][i] = out[i]; 
            }
        }
        
        //Activation Function 
        for(int i = 0; i < N; i++){
            for(int j = 0; j < n_out; j++){
                if (nn_output[i][j] > 0.5)
                    output[i][j] = 1;                     
                else
                    output[i][j] = 0;                                       
            }
        }
        
        //Validation Error and Accuracy of the model
        if (istestset == true){
            writeTestLabel(output); 
        }
        else{
            double[][] Y = list.get(1); 
            double[] err = new double[N]; // error 
            double[] acc = new double[N]; // accuracy
            
            for(int i = 0; i < N; i++){
                double[] d_out = Y[i];
                double[] nn_out = nn_output[i];
                int a = 0;
                for(int j = 0; j < n_out; j++){
                    err[j] = (d_out[j] - nn_out[j]); //measures predictive error (SSE)                
                    if(Y[i][j] == 1 && output[i][j] == 1)
                        a = a + 1; //accuracy increments for every correctly classified instance
                }
                acc[i] = a;
            }
                                    
            //Calculate total accuracy and error
            double e_sum = 0;
            double a_sum = 0;
            for (int i = 0; i < N; i++){
                e_sum = e_sum + err[i]*err[i];
                a_sum = a_sum + acc[i];
            }
            double accuracy = ((a_sum/N))*100;
            double total_err = e_sum;
            
            //Print and return error and accuracy
            System.out.println("Prediction Error: " + total_err + "; Accuracy: " + accuracy);
            List rlist = new ArrayList<>();
            rlist.add(total_err);
            rlist.add(accuracy);
            return rlist;
        }
        return null;
    }
    
    public static void main(String[] args) throws IOException {
        NeuralNet nn = new NeuralNet();
        nn.crossValidate();
        nn.testSet();
    }
}