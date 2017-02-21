package it.uniroma1.lcl.mynn;

import java.util.LinkedList;

/**
 * The function to implement the algorithm to update the neuron's weights
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 * @see			UfPercettrone class for the implementation.
 * @see			UfSingleLayer class for the implementation.
 * @see			UfMultiLayer class for the implementation. 
 */

@FunctionalInterface
public interface IUpgradeFunction {
	double[] upgrade(LinkedList<Layer> layers, int i, double[] input, 
					 double[] output, double lr) 
							 throws UpgradeFunctionException;
}
