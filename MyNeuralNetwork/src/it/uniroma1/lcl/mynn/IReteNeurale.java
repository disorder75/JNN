package it.uniroma1.lcl.mynn;

/**
 * Neural network interface
 * 
 * @author      Nunzio Castelli
 * @since       1.0
 * @see			NeuralNetwork class for the implementation.
 */

public interface IReteNeurale {
	
	public double[] process(double[] values);
	
	public double trainIstanza(double[] values, double output[]);
	
	public void train(double[][] inputs, double[][] outputs);
	
	public String getNome();
	
	public static IReteNeurale carica(String filename) throws ParserException {
		
		// change and complete 
		/*
		if (filename.contains("And"))
			return new DummyReteAnd();
		if (filename.contains("Xor"))
			return new DummyReteXor();
		if (filename.contains("Squared"))		
			return new DummyReteSquared();
		if (filename.contains("Sum"))		
			return new DummyReteSum();
		if (filename.contains("Percettrone"))		
			return new DummyRetePercettrone();		
		return new DummyReteOr();
		*/
		
		Parser p = new Parser(filename);
		NeuralNetwork myNN = p.getNN();
		
		return myNN;		
	}	
}
