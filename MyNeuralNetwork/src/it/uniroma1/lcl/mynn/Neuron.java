package it.uniroma1.lcl.mynn;

import java.util.Arrays;
import java.util.HashMap;

/**
 * Representation of a Neuron, the basic element of a neural network.
 * The neuron has a variable channel inputs, for each input there is an 
 * associated weight and each neuron is able to produce an output in according 
 * with a formula requested by the layer.
 * For the morphology of a neuron and its properties see MyNN documentation.
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 * 
 */
public class Neuron {

	private double threshold;
	private double[] inputs;
	private double[] weights;
	private double weightSum;
	private HashMap<Integer, Double> bFactor;
	
	/**
	 * Create the neuron object 
	 *
	 * @param	inputs  number of channel inputs.
	 * @param	weights  weights associated to the inputs.
	 */
	public Neuron (double[] inputs, double[] weights) {

		this.inputs = inputs;
		this.weights = Arrays.copyOf(weights, inputs.length);
		
		/* if the threshold isn't present into the weights list, generate
		 * a random one */
		if (inputs.length == weights.length)
			this.threshold = Math.random();
		else
			this.threshold = weights[weights.length - 1];
		
		bFactor = new HashMap<Integer, Double>();
	}

	/**
	 * Set the weights for the current Neuron object.
	 * The previously values will be overwritten.
	 *
	 * @param	weigths  the new values.
	 */
	public void setWeights(double[] weigths) {
		this.weights = weigths;
	}

	/**
	 * Set the partial derivate value in according to the documentation.
	 * This value is used during the upgrading function for the weigths
	 * into the multi layer network.
	 * Each channel has a bFactor value.
	 *
	 * @param	weigthIndex 
	 * @param	bFactor  partial derivate value
	 */
	public void setBfactor(double bFactor, int weigthIndex) {
		this.bFactor.put(weigthIndex, bFactor);
	}

	/**
	 * Set a weight valye for the current Neuron object.
	 *
	 * @param	weigth	value
	 * @param	i  channel index
	 */
	public void setWeight(double weigth, int i) {
		this.weights[i] = weigth;
	}
	
	/**
	 * Retrieve the partial derivate value calculated previously.
	 * This method is usefull during the upgrading function for the weigths into
	 * the multi layer network.
	 * Each channel has a bFactor value.
	 * 
	 * @param	weigthIndex 
	 * @return	the stored bFactor
	 */
	public double getBfactor(int weigthIndex) {
		return this.bFactor.get(weigthIndex);
	}

	/**
	 * Retrieve the last transfert function value.
	 *  
	 * @return	transfer function value.
	 */
	public double getLastTransferFunctionResult() {
		return this.weightSum;
	}

	/**
	 * Retrieve the channels inputs number.
	 *  
	 * @return	total inputs counter. 
	 */
	public int getInputsCount() {
		return inputs.length;
	}

	/**
	 * Retrieve the weight value associated to a specific input.
	 *
	 * @param	i	input index.
	 * @return	weight value. 
	 */
	public double getWeight(int i) {
		return weights[i];
	}

	/**
	 * Retrieve the input value (if is already stored) for a specific channel.
	 *
	 * @param	i	input index.
	 * @return	input value. 
	 */
	public double getInput(int i) {
		return inputs[i];
	}

	/**
	 * Retrieve the Neuron threshold
	 *
	 * @return	threshold value. 
	 */
	public double getThreshold() {
		return threshold;
	}

	/**
	 * Retrieve all the weights values from the current Neuron object.
	 *
	 * @return	array of weigths. 
	 */
	public double[] getWeights() {
		return Arrays.copyOf(weights, weights.length);
	}

	/**
	 * Retrieve all the weights values plus the threshold at the end from the
	 * current Neuron object. 
	 *
	 * @return	array of weigths. 
	 */
	public double[] getWeightsAndThreshold() {
		double[] wt = Arrays.copyOf(weights, weights.length+1);
		wt[wt.length-1] = threshold;
		return wt;
	}

	/**
	 * Set the threshold for the current Neuron object.
	 * The old value will be overwritten.
	 *
	 * @param	threshold	the new value. 
	 */
	public void setThreshold(double threshold) {
		this.threshold = threshold;
		return;
	}

	/**
	 * Calculate and store the transfer function value.
	 * The transfer function is the balanced sum of all channel inputs for the
	 * associated weights value, example:
	 * 			transferFunction = E(Input-i * Weigth-i)
	 * 
	 * @param	inputsValues neuron inputs, 
	 * @return	transfer function value.
	 */
	public double transferFunction(double[] inputsValues) throws ActivateFunctionException {

		double weightSum = 0;
		
		/* inputs values must be in according with number of inputs of the
		 * neuron */
		if (inputsValues.length != inputs.length)
			throw new ActivateFunctionException("Input values mismatch with" + 
												" expected numbers from" + 
												" current neuron [" + 
												inputsValues.length + " vs " + 
												inputs.length + "]");

		/* generate random values for the weights */
		if (weights == null || weights.length == 0) {
			weights = new double[inputsValues.length];
			for (int i=0; i < weights.length; i++) { 
				weights[i] = Math.random();
			}
		}
		
		inputs = Arrays.copyOf(inputsValues, inputsValues.length);
		
		for (int i=0; i < weights.length; i++) {
			weightSum += weights[i] * inputs[i];
		}		
		return weightSum;
	}
	
	/**
	 * Calculate the "step" function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output. 
	 */
	public double step(double x) {
		return x < 0 ? 0 : 1;
	}

	/**
	 * Calculate the "identity" function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output. 
	 */
	public double identity(double x) { 
		return x; 
	}

	/**
	 * Calculate the "identity" defivate F1 function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output.
	 */
	public double identity_f1(double x) { 
		return 1;
	}

	/**
	 * Calculate the "logistic" function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output.
	 */
	public double logistic(double x) { 
		double exp = -x;
		double ret = 1 / (1 + Math.pow(Math.E, exp));
		return ret; 
	}
	
	/**
	 * Calculate the "logistic" defivate F1 function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output.
	 */
	public double logistic_f1(double x) { 
		return logistic(x) * (1 - logistic(x));
	}

	/**
	 * Calculate the "tanh" function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output.
	 */
	public double tanh(double x) { 
		return Math.tanh(x);
	}

	/**
	 * Calculate the "tanh" defivate F1 function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output.
	 */
	public double tanh_f1(double x) { 
		return 1 - Math.pow(tanh(x), 2);
	}

	/**
	 * Calculate the "relu" function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output.
	 */
	public double relu(double x) { 
		return x < 0 ? 0 : x;
	}
	
	/**
	 * Calculate the "tanh" defivate F1 function.
	 * Invoked by the activation function on the current Neuron object.
	 *
	 * @param	x	input.
	 * @return	output.
	 */
	public double relu_f1(double x) { 
		return x < 0 ? 0 : 1;
	}

}
