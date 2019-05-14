import java.util.ArrayList;
import java.io.File;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;

import java.util.Arrays;
public class NeuralNetwork {
	public double[][] nodes;
	public double[][][] weights;
	public double[][] errors;
	public double[][] bias_weights;
	public double[][] bias_errors;
	public double learning_rate = 0.1;
	private int[] sizes;
	
	public NeuralNetwork(int[] topology) {
		nodes = new double[topology.length][];
		errors = new double[topology.length][];
		bias_weights = new double[topology.length][];
		
		for(int layer = 0; layer < nodes.length; layer++) {
			nodes[layer] = new double[topology[layer]];
			errors[layer] = new double[topology[layer]];
			bias_weights[layer] = new double[topology[layer]];
			for(int w = 0; w < bias_weights[layer].length; w++) {
				bias_weights[layer][w] = Math.random();
			}
		}
		
		weights = new double[topology.length-1][][];
		for(int layer = 0; layer < weights.length; layer++) {
			weights[layer] = new double[nodes[layer+1].length][];
			for(int toNode = 0; toNode < weights[layer].length; toNode++) {
				weights[layer][toNode] = new double[nodes[layer].length];
				for(int weight = 0; weight < weights[layer][toNode].length; weight++) {
					weights[layer][toNode][weight] = Math.random();
				}
			}
		}
		
		sizes = topology;
	}
	
	public void save(String filename) {
		try {
			File f = new File(filename);
			FileWriter writer = new FileWriter(f,false);
			for(int layer = 0; layer < sizes.length; layer++) {
				if(layer == sizes.length-1) {
					writer.write(Integer.toString(sizes[layer]));
				}
				else {
					writer.write(sizes[layer] + ",");
				}
			}
			writer.write("\n");
			
			for(int layer = 0; layer < weights.length; layer++) {
				for(int node = 0; node < weights[layer].length; node++) {
					for(int weight = 0; weight < weights[layer][node].length; weight++) {
						if(weight == weights[layer][node].length-1) {
							writer.write(Double.toString(weights[layer][node][weight]));
						}
						else {
							writer.write(weights[layer][node][weight] + ",");
						}
					}
					if(node != weights[layer].length-1) {
						writer.write("/");
					}
				}
				writer.write("\n");
			}
			
			for(int layer = 0; layer < bias_weights.length; layer++) {
				System.out.println(Arrays.toString(bias_weights[layer]));
				for(int node = 0; node < bias_weights[layer].length; node++) {
					if(node == bias_weights[layer].length-1) {
						writer.write(Double.toString(bias_weights[layer][node]));
					}
					else {
						writer.write(bias_weights[layer][node] + ",");
					}
				}
				writer.write("/");
			}
			
			writer.close();
		}
		catch(IOException e) {
			System.out.println(e);
		}
	}
	
	public void load(String filename) {
		try {
			File f = new File(filename);
			FileReader freader = new FileReader(f);
			BufferedReader reader = new BufferedReader(freader);
			reader.readLine();
			for(int layer = 0; layer < weights.length; layer++) {
				String line = reader.readLine();
				System.out.println(line);
				String[] weightLayers = line.split("/");
				for(int node = 0; node < weights[layer].length; node++) {
					String[] weightArrays = weightLayers[node].split(",");
					for(int weight = 0; weight < weights[layer][node].length; weight++) {
						weights[layer][node][weight] = Double.parseDouble(weightArrays[weight]);
					}
				}
			}
			
			String line = reader.readLine();
			for(int layer = 0; layer < bias_weights.length; layer++) {
				String[] weightLayers = line.split("/");
				for(int node = 0; node < bias_weights[layer].length; node++) {
					String[] weightArrays = weightLayers[node].split(",");
					bias_weights[layer][node] = Double.parseDouble(weightArrays[node]);
				}
			}
			
		}
		catch(IOException e) {
			System.out.println(e);
		}
	}
	
	
	private double multiplySum(double[] a, double[] b) {
		double sum = 0;
		for(int i = 0; i < a.length; i++) {
			sum += (a[i] * b[i]);
		}
		return sum;
	}
	
	
	public void calculateLayer(int layer) {
		if(layer == nodes.length-1) {
			for(int node = 0; node < nodes[layer].length; node++) {
				nodes[layer][node] = Formulas.sigmoid(multiplySum(nodes[layer-1],weights[layer-1][node]) + bias_weights[layer][node],false);
			}
		}
		else {
			for(int node = 0; node < nodes[layer].length; node++) {
				nodes[layer][node] = Formulas.relu(multiplySum(nodes[layer-1],weights[layer-1][node]) + bias_weights[layer][node],false);
			}
		}
	}
	
	public void calculateOutputError(double[] expected) {
		for(int node = 0; node < nodes[nodes.length-1].length; node++) {
			errors[nodes.length-1][node] = Formulas.sigmoid(nodes[nodes.length-1][node],true) * (expected[node] - nodes[nodes.length-1][node]);
		}
	}
	
	public void calculateError(int layer) {
		for(int node = 0; node < nodes[layer].length; node++) {
			double[] tempWeights = new double[nodes[layer+1].length];
			for(int weight = 0; weight < nodes[layer+1].length; weight++) {
				tempWeights[weight] = weights[layer][weight][node];
			}
			errors[layer][node] = Formulas.relu(nodes[layer][node],true) * multiplySum(tempWeights,errors[layer+1]);
		}
	}
	
	public void propagate(double[] inputs) {
		for(int i = 0; i < nodes[0].length;i++) {
			nodes[0][i] = inputs[i];
		}
		for(int layer = 1; layer < nodes.length; layer++) {
			calculateLayer(layer);
		}
	}
	
	public void backpropagate(double[] expected) {
		calculateOutputError(expected);
		for(int layer = nodes.length-2; layer > 0; layer--) {
			calculateError(layer);
		}
		
		for(int layer = 1; layer-1 < weights.length; layer++) {
			for(int node = 0; node < weights[layer-1].length; node++) {
				for(int weight = 0; weight < weights[layer-1][node].length; weight++) {
					weights[layer-1][node][weight] += (learning_rate * nodes[layer-1][weight] * errors[layer][node]);
				}
			}
		}
		
		for(int layer = 1; layer < bias_weights.length; layer++) {
			for(int node = 0; node < bias_weights[layer].length; node++) {
				bias_weights[layer][node] += (learning_rate * errors[layer][node]);
			}
		}
		
	}
	
	public void train(double[][] data) {
		propagate(data[0]);
		System.out.println(Arrays.deepToString(nodes));
		backpropagate(data[1]);
		double sum = 0;
		for(double e : errors[errors.length-1]) {
			sum += e;
		}
		System.out.println("Error: " + sum/errors[errors.length-1].length);
	}
	
	
	public static void main(String[] args) {
		int[] topology = {2,2,1};
		double[][][] datas = {{{0,0},{0}},{{1,0},{1}},{{0,1},{1}},{{1,1},{1}}};
		NeuralNetwork test = new NeuralNetwork(topology);
		test.load("network1.txt");
		System.out.println(Arrays.deepToString(test.weights));
		int r = (int)(Math.random()*4);
		for(int i = 0; i < 10000; i++) {
			r = (int)(Math.random()*4);
			test.train(datas[r]);
		}
		test.save("network1.txt");
		test.train(datas[0]);
		
	}
	
	
	
	
	
	

	
	
	
	
	
}