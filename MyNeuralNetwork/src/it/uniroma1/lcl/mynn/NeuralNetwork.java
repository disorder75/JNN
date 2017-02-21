package it.uniroma1.lcl.mynn;

import java.lang.reflect.InvocationTargetException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedList;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

/**
 * Implementation of the neural network based IReteNeurale structure.
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 * 
 */
public class NeuralNetwork implements IReteNeurale {

	private String networkName;
	private LinkedList<Layer> layers = new LinkedList<Layer>();
	private IUpgradeFunction uf = null;
	private double lr = 0.2;
	
	/**
	 * Create the network object 
	 *
	 * @param	networkName  name of the network.
	 */
	public NeuralNetwork(String networkName) {
		this.networkName = networkName;
	}
	
	/**
	 * Add a neurons layer to the current network object. 
	 *
	 * @param	l	layer object to add.
	 */
	public void addLayer(Layer l) {
		layers.add(l);
	}	

	/**
	 * Return the layers sum. 
	 *
	 * @return	number of layers available into the network.
	 */
	public int getLayerCount() {
		return layers.size();
	}	

	/**
	 * Return the network name. 
	 *
	 * @return	network name
	 */
	@Override
	public String getNome() {
		return this.networkName;
	}

	/**
	 * Return the names of all layers.
	 *
	 * @return	array with all names.
	 */
	public String[] getLayersName() {
		String[] layersName = new String[layers.size()];
		for (int i=0; i < layers.size(); i++) {
			layersName[i] = layers.get(i).getName();
		}
		return layersName;
	}
	
	private static long getDateDiff(Date date1, Date date2, TimeUnit timeUnit) {
	    long diffInMillies = date2.getTime() - date1.getTime();
	    return timeUnit.convert(diffInMillies,TimeUnit.MILLISECONDS);
	}
	
	/**
	 * Bind the function to calculate the weigths of the neurons. 
	 *
	 * @param	uf	UpgradeFuction object.
	 * @see		IUpgradeFunction interface.
	 * @see		UfMultiLayer class.
	 * @see		UfPercettrone class.
	 * @see		UfSingleLayer class.
	 */
	public void setUpgradeFunction(IUpgradeFunction uf) {
		this.uf = uf;
	}

	public void setNextLayerInputs(double[] inputs, int i) {
		if (i <= this.layers.size()-1) {
			this.layers.get(i).setLayerInputs(inputs);
		}
	}

	
	private double[] process_j7(double[] values, int layerIndex) throws ActivateFunctionException {
		
		double[] processedValues;
		
		/* exit condition */
		if (layerIndex >= layers.size() || values.length == 0 || values == null)
			return values;
		
		/* base step */
		Layer currentLayer = layers.get(layerIndex);
		for (int i=0; i < currentLayer.getOutputUnits(); i++) {
			//System.out.println("layerInput java7 " + Arrays.toString(values));
			double out = currentLayer.activateNeuron(i, values);
			//System.out.println("neuronOutput java7 " + out);
			currentLayer.setLayerNeuronOutput(i, out);
		}
		
		/* recursive step */
		processedValues = process(currentLayer.getProcessedLayer(), ++layerIndex);
		return processedValues;
	}

	private double[] process(double[] values, int layerIndex) throws ActivateFunctionException {
		
		/* Iterative cycle on the layers/neurons instead recursive step.
		 * Access to the data by the streams.
		 * */
		
		AtomicInteger lId = new AtomicInteger(0); 
		
		layers.forEach(l -> {
			l.getNeurons().forEach(n -> {
				try {
					double[] in = l.getLayerInputs();
					//System.out.println("layerInput java8 " + Arrays.toString(in));
					double out = l.activateNeuron(n, in);
					//System.out.println("neuronOutput java8 " + out);
					l.setLayerNeuronOutput(n, out);
				} catch (Exception e) {
					e.printStackTrace();
				}
			});
		
			double[]processedValues = l.getProcessedLayer();
			this.setNextLayerInputs(processedValues, lId.incrementAndGet());
		});
		
		double[]processedValues = layers.get(layers.size()-1).getProcessedLayer();
		return processedValues;
	}

	/**
	 * Process the input data and store the layer processed output.
	 *
	 * @param	values	input data for the network (or current layer).
	 */
	@Override
	public double[] process(double[] values) {
		
		double ret[] = null;
		
		/* inputs value for the network */
		this.layers.get(0).setLayerInputs(values);
		
		try {
			ret = process(values, 0);
			//System.out.println("Network Output " + Arrays.toString(ret));
		} catch (ActivateFunctionException e) {
			e.printStackTrace();
		}
	
		return ret;
	}
	
	private double trainIstanza(double[] values, double[] output, int lIndex) 
			throws IllegalAccessException, IllegalArgumentException, InvocationTargetException, 
				   NoSuchMethodException, ActivateFunctionException {

		double errors = 0;

		/* exit condition */
		if (values.length == 0 || output.length == 0 || lIndex < 0)
			return errors;
				
		/* step base */
		try {
			this.uf.upgrade(layers, lIndex,	values,	output, lr);
		} catch (UpgradeFunctionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
		/* recursive call */
		double inputs[];
		if (lIndex == (layers.size()-1)) {
			inputs = values;
		} else {
			inputs = layers.get(lIndex).getProcessedLayer();
		}
		
		trainIstanza(inputs, output, --lIndex);

		return errors;
	}
	
	/**
	 * Train the network in order to produce the processed output near to the
	 * values of the inputs as much as possible.
	 * Neurons network will modified in according with the "upgrading formula". 
	 * On inrreversible errors an exception will be raised.
	 * 
	 * @param	values	inputs data to elaborate.
	 * @param	output	expected output (training set).
	 * @return	sum of errors.
	 */
	
	@Override
	public double trainIstanza(double[] values, double[] output) {

		double errors = 0;
	
		//System.out.println("trainIstanza values = " + Arrays.toString(values) +
			//	" output = " + Arrays.toString(output));
		
		/* Process the network with current inputs */
		double[] networkOutput = this.process(values);
		for (int i=0; i < output.length; i++) {
			errors += Math.abs(output[i] - networkOutput[i]);
			//System.out.println("network " + getNome() + " input: " + 
			//					 values[i] + " output: " + networkOutput[i] + 
			//					 " expected: " + output[i] + " error distance " 
			//					 + errors);
		}
		
		if (errors < 0.01) {
			/* this set value is already trained */
			return 0;
		}

		/* check the upgrade formula to apply for the training method */
		if (getLayerCount() == 1 && 
			layers.getFirst().getActivationFunction().toLowerCase().compareTo("step") == 0) {
			setUpgradeFunction(new UfPercettrone());
		} else if (getLayerCount() == 1)
			setUpgradeFunction(new UfSingleLayer());
		else {
			setUpgradeFunction(new UfMultiLayer());
		}
		
		/* walk the network recursively in order to upgrade weights / threshold */
		try {
			trainIstanza(values, output, layers.size()-1);
		} catch (IllegalAccessException | IllegalArgumentException
				| InvocationTargetException | NoSuchMethodException e) {
			/* reflection errors */
			e.printStackTrace();
		} catch (ActivateFunctionException e) {
			/* neuron errors */
			e.printStackTrace();
		}
		
		return errors;
	}
	
	/**
	 *  Train the network in according to a set of inputs values and a set
	 *  of outputs values.
	 *
	 * @param	inputs	inputs data to elaborate.
	 * @param	outputs	expected output (training set).
	 */
	@Override
	public void train(double[][] inputs, double[][] outputs) {
		//FOR (x,y) IN insieme di addestramento  
		//errori← trainIstanza( x, y ) 
		//somma_errori = somma_errori + errori 
		//END FOR 
		
		double errorsThreshold;

		Date date1 = new Date();		
		System.out.println("start network " + this.getNome() + " training at " + 
						   date1.getTime());
		
		do {
			errorsThreshold = 0;
			for (int x=0; x < inputs.length; x++) {
				errorsThreshold += trainIstanza(inputs[x], outputs[x]);		
			}
			//System.out.println("errorsThreshold " + errorsThreshold);			
		} while (errorsThreshold > 0.01); 
		
		Date date2 = new Date();
		System.out.println("training completed in " + 
						   getDateDiff(date1,date2,TimeUnit.MINUTES) + 
						   " minutes");
	}
	
	/**
	 *  Print the network schema in according with the files processed 
	 *  by the parser.
	 *
	 * @return	the network layout
	 */
	@Override
	public String toString() {
		
		StringBuilder sb = new StringBuilder();

		/* network name */
		sb.append(ParserTokens.NETWORK_NAME.toString() + getNome());
		sb.append("\n");
		
		if (layers.size() > 0) {
			for (Layer l : layers) {
				/* start layer */
				sb.append(ParserTokens.START_LAYER.toString());
				sb.append(" ");
				/* name */
				sb.append(ParserTokens.NETWORK_NAME.toString() + l.getName());
				sb.append(" ");
				/* activation function */
				sb.append(ParserTokens.ACTIVATION_FUNCTION.toString() + 
						  l.getActivationFunction());
				sb.append(" ");
				/* inputs units*/
				sb.append(ParserTokens.INPUT_UNITS.toString() + 
						  l.getNeuronInputUnits());
				sb.append(" ");
				/* outputUnits units*/
				sb.append(ParserTokens.OUTPUT_UNITS.toString() + 
						  l.getOutputUnits());
				sb.append(" ");
				/* outputUnits units*/
				ArrayList<double[]> wt = l.getWeightsList();
				sb.append(ParserTokens.WEIGHTS.toString() + "[");
				for (int i=0; i < wt.size(); i++) {
					double[] w = wt.get(i);
					sb.append("[");

			        DecimalFormat format = new DecimalFormat("#.#", new DecimalFormatSymbols(Locale.US));
			        format.setDecimalSeparatorAlwaysShown(false);
					
					for (int ii = 0; ii < w.length; ii++) {
						if (ii == w.length-1)
							sb.append("" + format.format(w[ii]));
						else
							sb.append("" + format.format(w[ii]) + ",");
					}
					
					if (i == wt.size()-1)
						sb.append("]");
					else
						sb.append("],");
				}
				sb.append("]");
				sb.append(" ");
				/* end layer */
				sb.append(ParserTokens.END_LAYER.toString());
				sb.append("\n");				
			}
		}
		
		return sb.toString().trim();
	}
}
