package it.uniroma1.lcl.mynn;

/**
 * The exception class to raise and catch errors during the upgrading function
 * for the weights of the neurons into the layers.
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 */
public class UpgradeFunctionException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	/**
	 * Exception halted by the ActivationFunction  
	 *
	 * @param  errMsg error description developed inside the parser
	 * @see    Parser class
	 */
	public UpgradeFunctionException(String errMsg) {
		super(errMsg);
	}

}
