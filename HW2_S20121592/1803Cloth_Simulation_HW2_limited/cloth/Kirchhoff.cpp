#include "rkf45.h"

#define NEQN 2

double work[3+6*NEQN+10];
int iwork[10], neqn = NEQN;

void ODE_I(double *t, double *y, double *yp)
{
	yp[0] = -4.0*y[0] + 3.0*y[1] + 6.0;
	yp[1] = -2.4*y[0] + 1.6*y[1] + 3.6;
}

double ExactI1(double t)
{
	return -3.375*exp(-2*t) + 1.875*exp(-0.4*t) + 1.5;
}

double ExactI2(double t)
{
	return -2.25*exp(-2.0*t) + 2.25*exp(-0.4*t);
}

double abserr(double src, double dest)
{
	return ( src > dest ) ? src-dest : dest-src;
}

int main(void)
{
	double y[2] = { 0.0, 0.0 };

	double err = 0.00000000001;

	double t = 0.0, tinit = 0.0;

	int iflag = +1;

	printf("=====================================================================================\n");

	printf("%3s   %15s   %15s\t\t%15s\t%15s\n","t","w1","w2","|I1(t)-w1|","|I2(t)-w2|");

	printf("-------------------------------------------------------------------------------------\n");
	
	printf("%3f   %15.10f   %15.10f\t%15.10f\t%15.10f\n",t,0.0,0.0,0.0,0.0);
	for( t = 0.1; t < 0.6 ; t += 0.1 )
	{
		rkf45_(ODE_I,&neqn,y,&tinit,&t,&err,&err,&iflag,work,iwork);
		
		printf("%3f   %15.10f   %15.10f\t%10.10E   %10.10E\n",t,
			y[0],y[1],abserr(ExactI1(t),y[0]),abserr(ExactI2(t),y[1]));
	}

	printf("=====================================================================================\n");
	
	return 0;
}
