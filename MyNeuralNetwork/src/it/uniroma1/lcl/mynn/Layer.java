package it.uniroma1.lcl.mynn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

import com.sun.org.apache.bcel.internal.generic.RETURN;

/**
 * Representation of a layer from the neural network scheme.
 * A layer object keep inside a set of neurons of the same family.
 * Through its methods is able to activate all the neurons and store 
 * the processed output in order to give it to another layer.
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 * 
 */

public class Layer {

	private String name;
	private String activationFunction;
	private LinkedList<Neuron> neurons;
	private double[] inputs;
	private ArrayList<Double> processedOutput;
	
	/**
	 * Create a layer object able to keep inside a set of neurons 
	 * of the same family. This means that all the neurons will use the
	 * same activation function and they will have the same number of inputs
	 * connections. 
	 *
	 * @param	name  layer name.
	 * @param	activationFunction  name of the function used by the neurons.
	 */
	public Layer(String name, String activationFunction) {
		this.name = name;
		this.activationFunction = activationFunction;
		neurons = new LinkedList<Neuron>();
		processedOutput = new ArrayList<Double>();
	}
	
	/**
	 * Returns the name of the activation function used by the current object
	 * layer.
	 *
	 * @return	activation function name.
	 */
	public String getActivationFunction() {
		return activationFunction;
	}

	/**
	 * Set the activation function name.
	 * The previously value will be overwritten.
	 * 
	 * @param	activationFunction  name of the function used by neurons.		
	 */
	public void setActivationFunction(String activationFunction) {
		this.activationFunction = activationFunction;
		return;
	}

	/**
	 * Returns the name of the current object layer 
	 * object.
	 *
	 * @return	object layer name.
	 */
	public String getName() {
		return name;
	}

	/**
	 * Returns the number of connection inputs for the neurons. 
	 *
	 * @return	number of expected inputs or zero if no neurons are kept by the
	 * 			current object layer.
	 */
	public int getNeuronInputUnits() {
		/* Neurons of the same layer have the save numbers of input units, just
		 * pick one */
		return this.neurons.stream().findAny().get().getInputsCount();
		
	}
	public int getNeuronInputUnits_j7() {
		/* Neurons of the same layer have the save numbers of input units, just
		 * check the first one */
		int units = 0;
		if (neurons.size() > 0) {
			Neuron n = neurons.getFirst();
			units = n.getInputsCount();
		}
		return units;
	}
	
	/**
	 * Returns the weights of all neurons mantained by the current object layer. 
	 *
	 * @return	list of array, each element of the list is an array with all
	 * 			the weights of one neuron. 
	 */
	public ArrayList<double[]> getWeightsList() {
		return (ArrayList<double[]>) this.neurons.
										 stream().
										 map(n -> n.getWeightsAndThreshold()).
										 collect(Collectors.toList());
	}

	public ArrayList<double[]> getWeightsList_j7() {

		ArrayList<double[]> weights = null;
		if (neurons.size() > 0) {
			weights = new ArrayList<double[]>();			
			for (Neuron n : neurons) {
				weights.add(n.getWeightsAndThreshold());
			}
		} 
		return weights;
	}

	/**
	 * Returns the number of neurons mantained by the current object layer. 
	 *
	 * @return	number of neurons inside the layer.	
	 *  
	 */
	public int getOutputUnits() {
		return neurons.size();
	}

	/**
	 * Push a neuron into the current object layer. 
	 *
	 * @param	n	Neuron object to push into the layer
	 * @see		Neuron class
	 *  
	 */
	public void addNeuron(Neuron n) {
		neurons.add(n);
		processedOutput.add(0.0);
	}
	
	/**
	 * Return the processed output, in according with the activation function,
	 * of the current object layer. 
	 *
	 * @return	array of data representing the processed output of all the 
	 * 		 	neurons.
	 *  
	 */
	public double[] getProcessedLayer() {
		double[] ret = new double[processedOutput.size()];
		for (int i=0; i < processedOutput.size(); i++) {
			ret[i] = processedOutput.get(i);
		}
		return ret;
	}

	/**
	 * Returns the element at the specified position in this list.
	 *
	 * @return	Neuron object.
	 * @see		Neuron class.
	 * 
	 */
	public Neuron getNeuron(int i) {
		return neurons.get(i);
	}	

	/**
	 * Returns all the neurons mantained by the current object layer.
	 *
	 * @return	Neuron objects list.
	 * @see		Neuron class.
	 * 
	 */
	public LinkedList<Neuron> getNeurons() {
		return new LinkedList<>(neurons);
	}

	public double[] getLayerInputs() {
		return this.inputs;
	}
	
	/**
	 * Store the value representing the output of an neuron at the 
	 * specified position.
	 * The previously value, if exists, will be overwritten.
	 *
	 * @param	i position element where the value will be stored.
	 * @param	output value representing the neuron output.
	 * 
	 */
	public void setLayerNeuronOutput(int i, double output) {
		processedOutput.set(i, output);
		return;
	}

	/**
	 * Store the value representing the output of an neuron at the 
	 * specified position.
	 * The previously value, if exists, will be overwritten.
	 *
	 * @param	neuron that has generated the output.
	 * @param	output value representing the neuron output.
	 * 
	 */
	public void setLayerNeuronOutput(Neuron neuron, double output) throws ActivateFunctionException {
		int i = 0;
		for (Neuron n : this.neurons) {
			if (n.equals(neuron) == true) {
				setLayerNeuronOutput(i, output);
				return;
			}
			i++;
		}
		throw new ActivateFunctionException("invalid neuron, impossible to store output");
	}
	
	public void setLayerInputs(double[] inputs) {
		this.inputs = inputs;
	}
	
	/**
	 * Activate a neuron from the current object layer. 
	 * The activation function invoked will the depend from the activation
	 * function name stored into the layer.
	 * The invokation is dinamically resolved with reflection strategy, on errors
	 * an exception will be raised.
	 *
	 * @param	i	neuron index to activate.
	 * @param	values inputs of the neuron.
	 * @return	neuron output value.
	 * @see		ActivateFunctionException class for error management.
	 * 
	 */
	public double activateNeuron(int i, double[] values) throws ActivateFunctionException {

		double neuronOutput = 0;
		Neuron n = getNeuron(i);

		Method m;
		try {
			m = Neuron.class.getMethod(getActivationFunction().toLowerCase(), double.class);
		} catch (NoSuchMethodException e) {
			throw new ActivateFunctionException(getActivationFunction() + " method not found into the object class");
		} catch (SecurityException e) {
			throw new ActivateFunctionException(getActivationFunction() + " access denied");
		}
		
		//System.out.println("calculating transferOuput from values " + Arrays.toString(values) + " threshold " + n.getThreshold());
		double transferOuput = n.transferFunction(values) + n.getThreshold();
		//System.out.println("transferOuput " + transferOuput);
		
		try {
			neuronOutput = (double) m.invoke(n, transferOuput);
		} catch (IllegalAccessException e) {
			throw new ActivateFunctionException(getActivationFunction() + " access denied");
		} catch (IllegalArgumentException e) {
			throw new ActivateFunctionException(getActivationFunction() + " invalid function arguments");
		} catch (InvocationTargetException e) {
			throw new ActivateFunctionException(getActivationFunction() + " invocation failed");
		}
		// follow do the same work but cycle on all methods, more slow.
		/*
		for (Method m : Neuron.class.getMethods()) {
			if (m.getName().toLowerCase().compareTo(getActivationFunction().toLowerCase()) == 0) {
				double transferOuput = n.transferFunction(values) + n.getThreshold();
				neuronOutput = (double) m.invoke(n, transferOuput);
				processedOutput.set(i, neuronOutput);
				break;
			}
		}*/
		return neuronOutput;
	}
	
	/**
	 * Activate a neuron from the current object layer. 
	 * The activation function invoked will the depend from the activation
	 * function name stored into the layer.
	 * The invokation is dinamically resolved with reflection strategy, on errors
	 * an exception will be raised.
	 *
	 * @param	neuron to activate.
	 * @param	values inputs of the neuron.
	 * @return	neuron output value.
	 * @see		ActivateFunctionException class for error management.
	 * 
	 */
	public double activateNeuron(Neuron neuron, double[] values) throws ActivateFunctionException {
		int i = 0;
		for (Neuron n : this.neurons) {
			if (n.equals(neuron) == true) {
				return activateNeuron(i, values);
			}
			i++;
		}
		throw new ActivateFunctionException("invalid neuron to activate!");
	}	
	
	/**
	 * Activate a neuron from the current object layer. 
	 * The activation function invoked will be the derivate F1 related to the 
	 * activation function name stored into the layer.
	 * The invokation is dinamically resolved with reflection strategy, on 
	 * errors an exception will be raised.
	 *
	 * @param	i	neuron index to activate.
	 * @param	values inputs of the neuron.
	 * @return	neuron output value.
	 * @see		ActivateFunctionException class for error management.
	 * 
	 */
	public double activateNeuronF1(int i, double[] values) throws ActivateFunctionException {

		String af = new String(getActivationFunction());
		this.setActivationFunction(new String(getActivationFunction() + "_F1"));
		/* calculate the derivate */
		double neuronOutput = activateNeuron(i, values);
		this.setActivationFunction(af);		
		return neuronOutput;
	}
}
