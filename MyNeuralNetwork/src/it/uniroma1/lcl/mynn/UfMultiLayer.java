package it.uniroma1.lcl.mynn;

import java.util.LinkedList;

/**
 * Upgrading weights algorithm for the multi-layer neural network.
 * The algorithm is explained into MyNN documentation.
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 * 
 */
public class UfMultiLayer implements IUpgradeFunction {

	/**
	 * Update the weights of the neurons into the current layer in according
	 * to the multi layer algorithm.
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
		 * Last Layer
		 * 
		 * Wi = Wi + n(Oi - Yi) * F1(E(Wi*Hi) + O) * Hi
		 * O = O + n(Oi - Yi) * F1(E(Wi*Hi) + O) * Hi
		 * B = (Oi - Yi) * F1(E(Wi*Hi) + O) 
		 * conseguence:
		 * Wi = Wi + n * B * Hi
		 * 
		 * Generic Layer
		 * 
		 * Wki = Wki + n * Bk * Hk
		 * Bk = B(k-1) * W(k-1)i * F1(E(Wk*Hk))
		 * 
		 * */
		
		Layer currentLayer = layers.get(i);
		double[] processedOutputs = currentLayer.getProcessedLayer();
		boolean outputLayer = false;
		double f1;
		double bi;
		
		if (i == layers.size()-1) {
			outputLayer = true;
		}
		
		for (int neuronIndex=0; neuronIndex < currentLayer.getOutputUnits(); neuronIndex++) {

			Neuron n = currentLayer.getNeuron(neuronIndex);
			
			for (int weightIndex=0; weightIndex < n.getInputsCount(); weightIndex++) {
				if (outputLayer == true) {
					double[] outputPrevLayer = layers.get(i-1).getProcessedLayer();
					
					try {
						f1 = currentLayer.activateNeuronF1(neuronIndex, outputPrevLayer);
					} catch (ActivateFunctionException e) {
						throw new UpgradeFunctionException(
								"Failed to activate the derivate F1 function " +
								 "on the neuron id " + neuronIndex + " of the " + 
								 "output layer " + currentLayer.getName());
					}
					
					bi = (output[neuronIndex] - processedOutputs[neuronIndex]) * f1;
					n.setBfactor(bi, weightIndex);
					/* upgrade weight */
					double w = n.getWeight(weightIndex) + lr * bi * n.getInput(weightIndex);
					n.setWeight(w, weightIndex);
					/* upgrade threshold */
					double threshold = n.getThreshold() + lr * bi;
					n.setThreshold(threshold);
				} else {

					/* hidden layers */
					Layer nextLayer = layers.get(i+1);
					double bk = 0;
					for (Neuron nn : nextLayer.getNeurons()) {
						for (int wi=0; wi < nn.getInputsCount(); wi++) {
							bk += nn.getBfactor(wi) * nn.getWeight(wi);
						}
					}
					
					try {
						f1 = currentLayer.activateNeuronF1(neuronIndex, input);
					} catch (ActivateFunctionException e) {
						throw new UpgradeFunctionException(
								"Failed to activate the derivate F1 function " +
								 "on the neuron id " + neuronIndex + " of the " + 
								 "hidden layer " + currentLayer.getName());
					}
					
					bk = bk * f1;
					double w = n.getWeight(weightIndex) + lr * bk * n.getInput(weightIndex);
					n.setWeight(w, weightIndex);
					
					double threshold = n.getThreshold() + lr * bk;
					n.setThreshold(threshold);
				}
			}	
		}
		
		return null;
	}	
}
