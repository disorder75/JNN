package it.uniroma1.lcl.mynn;

import java.util.LinkedList;

/**
 * Upgrading weights algorithm for the percettrone neural network.
 * The algorithm is explained into MyNN documentation.
 *  
 * @author      Nunzio Castelli
 * @since       1.0
 * 
 */
public class UfPercettrone implements IUpgradeFunction {

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
	 * @throws ActivateFunctionException 
	 * @throws UpgradeFunctionException 
	 */
	@Override
	public double[] upgrade(LinkedList<Layer> layers, int i, double[] input, 
							double[] output, double lr) throws UpgradeFunctionException 
									 {

		/*
		 * Wi = Wi + n(Oi - Yi) * Xi
		 * O = O + n(Oi - Yi)
		 * 
		 * */
		
		Layer currentLayer = layers.get(i);
		double[] process = currentLayer.getProcessedLayer();

		for (int ni=0; ni < currentLayer.getOutputUnits(); ni++) {
			Neuron n = currentLayer.getNeuron(ni);

			/* check data integrity */
			if (n.getInputsCount() != input.length)
				throw new UpgradeFunctionException("Expected Neuron input mismatch with "
											 + "input set " + n.getInputsCount() + 
											 " vs " + input.length);

			for (int weightIndex=0; weightIndex < n.getInputsCount(); weightIndex++) {

				/* upgrade weight */
				double w = n.getWeight(weightIndex);
				
				w = w + (lr*(output[ni] - process [ni])) * input[weightIndex];
				//System.out.println("upgrading neuron nr." + i + " from " + n.getWeight(i) + " to " + w);
				n.setWeight(w, weightIndex);
				
				/* upgrade threshold */
				double threshold = n.getThreshold();
				threshold = threshold + (lr*(output[ni] - process[ni]));
				//System.out.println("upgrading threshold nr." + i + " from " + n.getThreshold() + " to " + threshold);
				n.setThreshold(threshold);
			}
		}
		
		return null;
	}

}
