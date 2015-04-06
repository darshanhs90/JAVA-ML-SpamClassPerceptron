

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeSet;

public class PerceptronTrainWOStopWords {
	public static TreeSet<String> tree;
	public static ArrayList<String> featureList;
	public static int noOfFeatures;
	public static ArrayList<ArrayList<String>> data;
	public static ArrayList<Integer> classValues;
	public static int[][] matrix;
	public static double[] weightMatrix;
	public static double weights0;
	public static double learningRate;
	private static int iter;

	private static String hamtrainFolder;
	private static String spamtrainFolder;
	private static String hamtestFolder;
	private static String spamtestFolder;
	private static String stopWordsFile;
	private static String learningRateStr;
	private static String iterations;

	private static HashMap<String,String> stopText=new HashMap<String,String>();
	public static void main(String[] args) throws Exception {


		hamtrainFolder=args[0];
		spamtrainFolder=args[1];
		hamtestFolder=args[2];
		spamtestFolder=args[3];
		stopWordsFile=args[4];
		learningRateStr=args[5];
		iterations=args[6];
		learningRate=Double.parseDouble(learningRateStr);
		iter=Integer.parseInt(iterations);
		System.out.println("Started");




		ArrayList<String[]> retValue=new ArrayList<String[]>();
		classValues=new ArrayList<Integer>();
		tree=new TreeSet<String>();
		featureList=new ArrayList<String>();
		retValue=(extractVocab());
		getFeatures(retValue);
		noOfFeatures=featureList.size();
		/*System.out.println("No Of Features is :"+noOfFeatures);
		System.out.println(classValues.size());	*/
		matrix=new int[classValues.size()][noOfFeatures];
		Arrays.fill(matrix[0],0);
		createMatrix();
		weightMatrix=new double[featureList.size()];
		createWeightMatrix();
		trainPerceptron();
		//System.out.println("done");
		testPerceptron();
	}


	private static void testPerceptron() throws Exception {
		int posCount=0,negCount=0;
		StringBuilder sb;
		//File hamfolder = new File("C:/Users/darshan/Desktop/MS/Text Books/Machine Learning/Assignment/Assignment 3/hw2/test/ham");
		File hamfolder = new File(hamtestFolder);

		File[] hamlistOfFiles = hamfolder.listFiles();
		for (int i = 0; i < hamlistOfFiles.length; i++) {
			FileInputStream en=new FileInputStream(new File(hamlistOfFiles[i].toString()));
			BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(hamlistOfFiles[i].toString()))));
			String str=x.readLine();
			sb=new StringBuilder();
			while (str!=null) {
				str=str.toLowerCase();
				sb.append(str);
				str=x.readLine();
			}
			str=sb.toString();
			//do  calculation
			str=str.replaceAll("[^a-zA-Z]"," ");
			str=str.replaceAll("\\s+", " ");
			String s[]=str.split(" ");
			int tempWeights[]=new int[weightMatrix.length];
			Arrays.fill(tempWeights,0);
			for (int j = 0; j < s.length; j++) {
				String word=s[j];
				if(!(stopText.containsKey(word.toLowerCase()))){
					int index=featureList.indexOf(word);
					if(index!=-1)
						tempWeights[index]++;
				}
			}
			double sum=0;
			for (int j = 0; j < tempWeights.length; j++) {
				sum+=tempWeights[j]*weightMatrix[j];
			}
			sum+=weights0;
			//System.out.println("ham"+i+"||"+sum);
			if(sum>0)
				posCount++;
			else 
				negCount++;

		}
		System.out.println("Ham accuracy is:"+(100*((double)(posCount))/((double)(posCount+negCount))));
		/*System.out.println(posCount);
		System.out.println(negCount);*/
		int posCount1=posCount;
		int negCount1=negCount;

		posCount=0;
		negCount=0;
		//For Spam
		//File spamfolder = new File("C:/Users/darshan/Desktop/MS/Text Books/Machine Learning/Assignment/Assignment 3/hw2/test/spam");
		File spamfolder = new File(spamtestFolder);

		File[] spamlistOfFiles = spamfolder.listFiles();
		for (int i = 0; i < spamlistOfFiles.length; i++) {
			FileInputStream en=new FileInputStream(new File(spamlistOfFiles[i].toString()));
			BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(spamlistOfFiles[i].toString()))));
			String str=x.readLine();
			sb=new StringBuilder();
			while (str!=null) {
				str=str.toLowerCase();
				sb.append(str);
				str=x.readLine();
			}
			str=sb.toString();
			//do  calculation
			str=str.replaceAll("[^a-zA-Z]"," ");
			str=str.replaceAll("\\s+", " ");
			String s[]=str.split(" ");
			int tempWeights[]=new int[weightMatrix.length];
			Arrays.fill(tempWeights,0);
			for (int j = 0; j < s.length; j++) {
				String word=s[j];
				if(!(stopText.containsKey(word.toLowerCase()))){
					int index=featureList.indexOf(word);
					if(index!=-1)
						tempWeights[index]++;
				}
			}
			double sum=0;
			for (int j = 0; j < tempWeights.length; j++) {
				sum+=tempWeights[j]*weightMatrix[j];
			}
			sum+=weights0;
			//System.out.println("spam"+i+"||"+sum);
			if(sum<=0)
				posCount++;
			else 
				negCount++;

		}
		System.out.println("Spam accuracy is:"+(100*((double)(posCount))/((double)(posCount+negCount))));
		/*System.out.println(posCount);
		System.out.println(negCount);*/
		System.out.println("Total accuracy is:"+(100*((double)(posCount+posCount1))/((double)(posCount+posCount1+negCount+negCount1))));
		System.out.println("Learning rate:"+learningRate);
		System.out.println("Iterations :"+iter);

	}


	private static void trainPerceptron() {
		for (int iter1 = 0; iter1 < iter; iter1++) {
			for (int i = 0; i < classValues.size(); i++) {//get each file or row data from here
				double sum=0;
				int output=0;
				for (int j = 0; j <featureList.size(); j++) {//for each feature
					sum+=weightMatrix[j]*matrix[i][j];
				}
				sum+=weights0;
				if(sum>0)
					output=1;
				else 
					output=-1;

				if(output!=classValues.get(i)){
					int cv=classValues.get(i);
					//System.out.println("output"+output+"cv"+cv);
					double[] deltaW=new double[weightMatrix.length];
					Arrays.fill(deltaW,0);
					double diff=cv-output;
					for (int j = 0; j < weightMatrix.length; j++) {
						deltaW[j]=learningRate*(diff)*matrix[i][j];
						weightMatrix[j]+=deltaW[j];
					}
				}
			}
		}
	}


	private static void createWeightMatrix() {
		Random r=new Random();
		weights0=-1 + r.nextDouble() * (1 - (-1));
		for (int i = 0; i < weightMatrix.length; i++) {
			double x=r.nextDouble();
			weightMatrix[i]=(double) x;
			weightMatrix[i]=-1 + r.nextDouble() * (1 - (-1));
		}
	}

	private static void createMatrix() {
		for (int i = 0; i < data.size(); i++) {
			ArrayList<String> rows=data.get(i);
			for (int j = 0; j < rows.size(); j++) {
				String str=rows.get(j);
				//System.out.println(str);
				if(!(stopText.containsKey(str.toLowerCase()))){
					int index=featureList.indexOf(str);
					matrix[i][index]++;
				}
			}
		}
	}


	private static void getFeatures(ArrayList<String[]> str) {
		data=new ArrayList<ArrayList<String>>();
		ArrayList<String> al;
		for (int i = 0; i < str.size(); i++) {
			String [] strArray=str.get(i);
			for (int j = 0; j < strArray.length; j++) {
				String line=strArray[j];
				al=new ArrayList<String>();
				line=line.toString().replaceAll("[^a-zA-Z]"," ");
				line=line.toString().replaceAll("\\s+", " ");
				String[] tempArray=line.split(" ");
				for (int k = 0; k < tempArray.length; k++) {
					String s=tempArray[k];
					//System.out.println(s);
					if(!(stopText.containsKey(s.toLowerCase()))){
						al.add(s.toLowerCase());
						boolean b=tree.add(s.toLowerCase());
						if(b==true)
							featureList.add(s.toLowerCase());
					}
				}
				data.add(al);
			}	
		}
	}

	private static ArrayList<String[]> extractVocab() throws Exception{
		//File stopfile = new File("stopwords.txt");
		File stopfile = new File(stopWordsFile);

		StringBuilder builder=new StringBuilder();
		BufferedReader x=new BufferedReader(new InputStreamReader(new FileInputStream(stopfile)));
		String str=x.readLine();
		while (str!=null) {
			builder.append(str);
			str=str.replaceAll("[^a-zA-Z]"," ");
			str=str.replaceAll("\\s+", "");
			stopText.put(str, str);
			str=x.readLine();
		}








		//File hamfolder = new File("C:/Users/darshan/Desktop/MS/Text Books/Machine Learning/Assignment/Assignment 3/hw2/train/ham");
		File hamfolder = new File(hamtrainFolder);
		
		File[] hamlistOfFiles = hamfolder.listFiles();
		builder=new StringBuilder(); 
		String[] returningObj=new String[hamlistOfFiles.length];
		for (int i = 0; i < hamlistOfFiles.length; i++) {
			classValues.add(1);
			builder=new StringBuilder();
			FileInputStream en=new FileInputStream(new File(hamlistOfFiles[i].toString()));
			x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(hamlistOfFiles[i].toString()))));
			str=x.readLine();
			while (str!=null) {
				builder.append(str.toLowerCase());
				str=x.readLine();
			}
			returningObj[i]=builder.toString();
		}

		//File spamfolder = new File("C:/Users/darshan/Desktop/MS/Text Books/Machine Learning/Assignment/Assignment 3/hw2/train/spam");
		File spamfolder = new File(spamtrainFolder);
		File[] spamlistOfFiles = spamfolder.listFiles();
		String[] returningObj1=new String[spamlistOfFiles.length];
		for (int i = 0; i < spamlistOfFiles.length; i++) {
			classValues.add(-1);
			builder=new StringBuilder();
			FileInputStream en=new FileInputStream(new File(spamlistOfFiles[i].toString()));
			x=new BufferedReader(new InputStreamReader(new FileInputStream(new File(spamlistOfFiles[i].toString()))));
			str=x.readLine();
			while (str!=null) {
				builder.append(str.toLowerCase());
				str=x.readLine();
			}
			returningObj1[i]=builder.toString();
		}


		ArrayList<String[]> ret=new ArrayList<String[]>();
		ret.add(returningObj);
		ret.add(returningObj1);
		return ret;
	}

}
