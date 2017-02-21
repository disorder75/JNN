package it.uniroma1.lcl.mynn;

/**
 * The tokens used by parser during the file processing.
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 */
public enum ParserTokens {

	NETWORK_NAME("nome="),
	NAME("nome="), 
	START_LAYER("layer={"), 
	END_LAYER("}"),
	ACTIVATION_FUNCTION("activationFunction="), 
	INPUT_UNITS("inputUnits="),
	OUTPUT_UNITS("outputUnits="), 
	WEIGHTS("weights=");
	
	private final String currentToken;
	
	ParserTokens(String tok) {
		this.currentToken = tok;
	}
	
	@Override
	public String toString() {
		return this.currentToken;
	}
	
}
