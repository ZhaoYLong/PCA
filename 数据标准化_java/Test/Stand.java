package Test;

public class Stand {
	double[] list = new double[100];
	double sun=0;//方差和
	double v2;//方差
	
	double son=0;//定义均差和
	double v1;//均差
	
	double v ;//初始化平均值
	double sum=0;//c初始化数之和
	
	
	public void random() {
		for(int i=0;i<100;i++) {
			double val = (double)(Math.random()*100+1);
			list[i]=val;
			System.out.print(" " + list[i]);
		}
		System.out.println("*****************");
	}
	public void fan() {
		for(int i=0;i<100;i++) {
			double val = (double)(Math.random()*100+1);
			list[i]=val;
			System.out.print(" " + list[i]);
		}
		System.out.println("*****************");
		//double v ;//初始化平均值
		//double sum=0;//c初始化数之和
		for(int j=0;j<list.length;j++) {
			sum+=list[j];
		}
		v=sum/list.length;
		System.out.println("平均值="+v);
	//double son=0;//定义均差和
	//double v1;//均差

	for(int i=0;i<list.length;i++) {
		son+=Math.abs(list[i]-v);
	}
	v1=son/list.length;
	System.out.println("均差="+v1);
	
	//double sun=0;//方差和
	//double v2;//方差
	for(int j=0;j<list.length;j++) {
		sun+=(list[j]-v)*(list[j]-v);
	}
	v2=sun/list.length;
	System.out.println("方差="+v2);
		
		for(int i=0;i<list.length;i++) {
			list[i]=(list[i]-v1)/v2;
			System.out.print(" "+list[i]);
			//System.out.println("*****************");
		}
		/**
		 * for(int j=0;j<list.length;j++){
		 * 		System.out.print(list[i] + "\t");
		 * 		if(j%5==0)
		 * 		System.out.println();
		 * }
		 */
	}
	
	/**
	 * Softmax方法如下：
	 */
	public void softmax() {
		for(int i=0;i<100;i++) {
			double val = (double)(Math.random()*100+1);
			list[i]=val;
			System.out.print(" " + list[i]);
		}
		System.out.println("*****************");
		
		double[] y = new double[100];
		for(int c=0;c<y.length;c++) {
			for(int i=0;i<list.length;i++) {
				y[c]=(list[i]-v1)/(v2*1);
				//System.out.print("  "+ y[c]);
				list[i]=1/(1+Math.exp(-y[c]));
			}
		}
		
		for(int i=0;i<list.length;i++) {
			System.out.print("  "+ list[i]);
		}
		
	}

}
