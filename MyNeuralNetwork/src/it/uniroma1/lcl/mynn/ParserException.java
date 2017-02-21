package it.uniroma1.lcl.mynn;

/**
 * The exception class to raise and catch errors during file parsing process.
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 */
public class ParserException extends Exception {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Exception halted by the Parser  
	 *
	 * @param  errMsg error description developed inside the parser
	 * @see    Parser class
	 */
	public ParserException(String errMsg) {
		super(errMsg);
	}

}
