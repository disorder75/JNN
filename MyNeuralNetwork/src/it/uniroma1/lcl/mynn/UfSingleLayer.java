package it.uniroma1.lcl.mynn;

import java.util.LinkedList;

/**
 * Upgrading weights algorithm for a generic single layer neural network.
 * The algorithm is explained into MyNN documentation.
 *  
 * @author      Nunzio Castelli
 * @since       1.0
 * 
 */
public class UfSingleLayer implements IUpgradeFunction {

	/**
	 * Update the weights of the neurons into the current layer in according
	 * to the generic single layer algorithm.
	 * On error an exception will be raised. 
	 * 
	 * @param	layers	the network layers.
	 * @param	i		current working layer. 
	 * @param	input	inputs for the current layer.
	 * @param	output 	the expected output for the current layer. 
	 * @param	lr		the learning rate constant.
	 * @return	null	reserved for future use	
	 */
	@Override
	public double[] upgrade(LinkedList<Layer> layers, int i, double[] input, 
							double[] output, double lr) 
									throws UpgradeFunctionException {

		/*
		 * Wi = Wi + n(Oi - Yi) * F1(E(Wi*Xi) + O) * Xi
		 * O = O + n(Oi - Yi) * F1(E(Wi*Xi) + O) * Xi
		 * 
		 * */
		
		Layer currentLayer = layers.get(i);
		double[] process = currentLayer.getProcessedLayer();
		//System.out.println("Single layer processed af " + Arrays.toString(process));
		
		for (int neuronIndex=0; neuronIndex < currentLayer.getOutputUnits(); neuronIndex++) {
			Neuron n = currentLayer.getNeuron(neuronIndex);

			/* check data integrity */
			if (n.getInputsCount() != input.length)
				new UpgradeFunctionException("Expected Neuron input mismatch with "
											 + "input set " + n.getInputsCount() + 
											 " vs " + input.length);
			
			for (int weightIndex=0; weightIndex < n.getInputsCount(); weightIndex++) {

				/* upgrade weight */
				double w = n.getWeight(weightIndex);
				double f1;
				
				try {
					f1 = currentLayer.activateNeuronF1(neuronIndex, input);
				} catch (ActivateFunctionException e) {
					throw new UpgradeFunctionException(
							"Failed to activate the derivate F1 function " +
							 "on the neuron id " + neuronIndex + " of the " + 
							 "single layer " + currentLayer.getName());				
				} 
				
				w = w + (lr * (output[neuronIndex] - process[neuronIndex])) * f1 * input[weightIndex];
				n.setWeight(w, weightIndex);

				/* upgrade threshold */
				double threshold = n.getThreshold();

				try {
					f1 = currentLayer.activateNeuronF1(neuronIndex, input);
				} catch (ActivateFunctionException e) {
					throw new UpgradeFunctionException(
							"Threshold: failed to activate the derivate F1 function " +
							 "on the neuron id " + neuronIndex + " of the " + 
							 "single layer " + currentLayer.getName());				
				}
				
				threshold = threshold + (lr*(output[neuronIndex] - process[neuronIndex])) * f1;
				n.setThreshold(threshold);
			}	
		}
		
		return null;
	}

}
