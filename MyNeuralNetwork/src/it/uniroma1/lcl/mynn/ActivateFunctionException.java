package it.uniroma1.lcl.mynn;

/**
 * The exception class to raise and catch errors during the activation of a 
 * neuron. 
 * 
 * @author      Nunzio Castelli 
 * @since       1.0
 */
public class ActivateFunctionException extends Exception {

	private static final long serialVersionUID = 1L;
	
	/**
	 * Exception halted by the ActivationFunction  
	 *
	 * @param  errMsg error description developed inside the parser
	 * @see    Parser class
	 */
	public ActivateFunctionException(String errMsg) {
		super(errMsg);
	}
}
