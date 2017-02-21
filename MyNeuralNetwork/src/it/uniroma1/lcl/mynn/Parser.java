package it.uniroma1.lcl.mynn;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;

/**
 * The parser.
 * Load a neural network schema from a file and create the neural network
 * object with all components specified into the loaded file.  
 * The network schema is explained into MyNN documentation.
 *  
 * @author      Nunzio Castelli
 * @since       1.0
 * 
 */
public class Parser {

	private File file;
	private LinkedList<Map<String, String>> lines = new LinkedList<Map<String, String>>();
	private Set<Entry<String, String>> entries;
	private Map<String, String> tokensMap;
	
	/**
	 * Create the parser object and load the file.
	 * On error an exception will be raised.
	 *
	 * @param	filename  file to be loaded from file system.
	 */
	public Parser(String filename) throws ParserException {

		if (filename == null) {
			throw new ParserException("Invalid filename");
		}

		file = new File(filename);
		if (file.exists() == false) {
			throw new ParserException("File " + filename + " not found");
		}
		
		verifyTemplate();
	}
	
	private void newLayer() {
		
		tokensMap = new HashMap<String, String>();
		
		if (lines.size() == 0) {
			tokensMap.put(ParserTokens.NETWORK_NAME.toString(), "");
		} else {
			for (ParserTokens tok : ParserTokens.values()) {
				tokensMap.put(tok.toString(), null);
			}			
		}
		
		lines.add(tokensMap);
	}
	
	private void verifyTemplate() throws ParserException {
		
		try {
			Scanner in = new Scanner(this.file);
			
			while (in.hasNext()) {
				
				String s = in.nextLine();
				newLayer();
				
				if (lines.size() == 1 && 
					tokensMap.get(ParserTokens.NETWORK_NAME.toString()).length() == 0) {
					if (s.startsWith(ParserTokens.NETWORK_NAME.toString()) == false) {
						throw new ParserException("File " + file.getName() + 
												  " has an invalid template," + 
												  " network name not found");
					}
					tokensMap.put(ParserTokens.NETWORK_NAME.toString(), 
								  s.substring(ParserTokens.NETWORK_NAME.toString().length()));
				} else {
					/* Populate the Map tokens [KEY=TOKEN / VALUE=data from file] */
					entries = tokensMap.entrySet();
					String[] ss = s.split(" ");
					for (String sss : ss) {
						sss = sss.trim();
						for (Entry<String, String> entry : entries) {
							int idx = sss.indexOf(entry.getKey());
							if (idx != -1) {
								tokensMap.put(entry.getKey(), 
											  sss.substring(idx + entry.getKey().length()));
								break;
							}
						}
					}
					
					/* verify all mandatory tokens */
					for (Entry<String, String> entry : entries) {
						if (entry.getValue() == null && 
							entry.getKey().compareTo(ParserTokens.WEIGHTS.toString()) != 0) {
							throw new ParserException("File " + file.getName() + 
									  " has an invalid template," + 
									  " incomplete layer, expected token: " + 
									  entry.getKey());							
						}
					}
				}
			}
			
		} catch (FileNotFoundException e) {
			throw new ParserException("Failed to parse file " + e.getMessage());
		}
		
		return;
	}
	
	private double[][] parseOptionalWeights(String weights,	int totNeurons,
											int inputs) {
		int wi = 0;
		double[][] retWeights;
			
		weights = weights.replace("[", "").replace("]", "");
		String[] as = weights.split(",");
		
		if ((double)(as.length / (double)inputs) == (double)totNeurons) {
			/* optional weights without threshold */
			retWeights = new double[totNeurons][inputs];
			for(int n=0; n < totNeurons; n++) {
				for (int i=0; i < inputs; i++, wi++)
					retWeights[n][i] =  Double.parseDouble(as[wi]);			
			}
		} else {
			/* optional weights with threshold */
			retWeights = new double[totNeurons][inputs + 1];
			for(int n=0; n < totNeurons; n++) {
				for (int i=0; i <= inputs; i++, wi++)
					retWeights[n][i] =  Double.parseDouble(as[wi]);
			}
		}
		
		return retWeights;
	}
	
	/**
	 * Retrieve the Neural Network object created on the schema loaded from
	 * the file specified at the parser creation time.
	 *
	 * @return	the neural network object. 
	 */
	public NeuralNetwork getNN() {

		NeuralNetwork myNN = null;
		double[][] weights;
		int inputs;
		int maxNeurons;
		Layer layer;
		
		for (Map<String, String> line : lines) {

			if (myNN == null) {
				/* create the network
				 * the first line is used to hold the name of the network */
				myNN = new NeuralNetwork(line.get(ParserTokens.NAME.toString()));
				continue;
			}
			
			/* create the layer, the neurons, weights and push all of them 
			 * into the layer */
			layer = new Layer(line.get(ParserTokens.NAME.toString()), 
							  line.get(ParserTokens.ACTIVATION_FUNCTION.toString()));
			inputs = Integer.parseInt(line.get(ParserTokens.INPUT_UNITS.toString()));
			maxNeurons = Integer.parseInt(line.get(ParserTokens.OUTPUT_UNITS.toString()));

			if(line.get(ParserTokens.WEIGHTS.toString()) == null) {
				/* generate randomically weights and threshold for the neuron */
				weights = new double[maxNeurons][inputs + 1];
				for (int n=0; n < maxNeurons; n++)
					for (int i=0; i < weights[n].length; i++)
						weights[n][i] = Math.random();
			} else
				weights = parseOptionalWeights(line.get(ParserTokens.WEIGHTS.toString()),
											   maxNeurons, inputs); 
			
			for (int i=0; i < maxNeurons; i++) {
				layer.addNeuron(new Neuron(new double[inputs], weights[i]));
			}
			
			/* add layer into the network */
			myNN.addLayer(layer);
		}
		return myNN;
	}
	
	/**
	 * Print the status of the parser.
	 * If the parser has loaded the file successfully is ready to give the 
	 * Neural Netwok by the method getNN. 
	 *
	 * @return	parser status. 
	 */
	@Override
	public String toString() {
		
		StringBuilder sb = new StringBuilder();

		if (lines.isEmpty()) {
			sb.append("The parser is empty, please load a valid network file");
		} else {
			sb.append("File " + file.getName() + " has a valid template format");
			sb.append("\n");
			sb.append("Call getNN() method to obtain a network object based on it");
		}
		
		return sb.toString();
	}	
}
