public class Formulas {
	
	public static double sigmoid(double x,boolean deriv) {
		if(!deriv) {
			return 1/(1+Math.pow(Math.exp(1),-1*x));
		}
		return sigmoid(x,false)*(1-sigmoid(x,false));
	}
	
	public static double relu(double x,boolean deriv) {
		if(!deriv) {
			if(x > 0) {
				return x;
			}
			else {
				return 0.01*x;
			}
		}
		if(x > 0) {
				return 1;
		}
		else {
			return 0.01;
		}
	}
	
	
	
	
	
	
	
	

	
}