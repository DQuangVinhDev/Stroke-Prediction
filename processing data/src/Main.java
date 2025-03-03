import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instances;
import weka.classifiers.functions.Logistic;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.io.File;
import java.io.IOException;
import weka.filters.Filter;


public class Main {
    public static void main(String[] args) throws IOException, Exception {

        //-------------------------------------------------------------------//
        // load CSV test
        CSVLoader loader_test = new CSVLoader();
        loader_test.setSource(new File("C:/Users/ACER/Documents/ITDataMining---Project/Preprocessing/DataTest.csv"));
        Instances csv_Data_test = loader_test.getDataSet();


        //write test data after convert to arff
        ArffSaver saver_test_data_convert = new ArffSaver();
        saver_test_data_convert.setInstances(csv_Data_test);
        saver_test_data_convert.setFile(new File("C:/Users/ACER/Documents/ITDataMining---Project/Preprocessing/DataTest.arff"));
        saver_test_data_convert.writeBatch();


        //-------------------------------------------------------------------//
        // load CSV train
        CSVLoader loader_train = new CSVLoader();
        loader_train.setSource(new File("C:/Users/ACER/Documents/ITDataMining---Project/Preprocessing/DataTrain.csv"));
        Instances csv_Data = loader_train.getDataSet();

       //save train
        ArffSaver saver = new ArffSaver();
        saver.setInstances(csv_Data);
        saver.setFile(new File("C:/Users/ACER/Documents/ITDataMining---Project/Preprocessing/DataTrain.arff"));
        saver.writeBatch();

        //load arff
        DataSource source = new DataSource("C:/Users/ACER/Documents/ITDataMining---Project/Preprocessing/DataTrain.arff");
        Instances data_train = source.getDataSet();
        data_train.setClassIndex(data_train.numAttributes() - 1);


        //-------------------------------------------------------------------//
        // Train the model with NaiveBayes
        NaiveBayesUpdateable nb = new NaiveBayesUpdateable();
        nb.buildClassifier(data_train);
        weka.core.SerializationHelper.write("NB_model.model", nb);

        // Train the model with logistic regression
        Logistic classifier = new Logistic();
        classifier.buildClassifier(data_train);
        weka.core.SerializationHelper.write("logistic_model.model", classifier);



        //---------------------------------------------------------------------------//
        // evaluate tren nb model (danh gia thu)

        //load NaiveBayes model
        NaiveBayesUpdateable nb1 = (NaiveBayesUpdateable) weka.core.SerializationHelper
                .read("NB_model.model");

        //load data test
        DataSource data_test = new DataSource("C:/Users/ACER/Documents/ITDataMining---Project/Preprocessing/DataTest.arff");
        Instances test = data_test.getDataSet();
        test.setClassIndex(test.numAttributes() - 1);

        //danh gia
        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(nb1, test);
        System.out.println(eval.toSummaryString("\nResults_nb\n\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        Logistic log = (Logistic) weka.core.SerializationHelper
                .read("logistic_model.model");

        //danh gia
        eval = new Evaluation(test);
        eval.evaluateModel(log, test);
        System.out.println(eval.toSummaryString("\nResults_logistic\n\n", false));
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());


    }
}
