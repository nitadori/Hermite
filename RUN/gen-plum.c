/*****************************************************************************
  File Name      : "gen-plum.c"
  Contents       : Plummer initial data generation for n-body tasks
  		 : with CM correction
  Coded by       : Peter Berczik
  Last redaction : 28.09.2006 13:49
*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>

/* Some "good" functions and constants... */
#define SIG(x)   ( ((x)<0) ? (-1):(1) )
#define ABS(x)   ( ((x)<0) ? (-x):(x) )

#define MAX(a,b) ( ((a)>(b)) ? (a):(b) )
#define MIN(a,b) ( ((a)<(b)) ? (a):(b) )

#define SQR(x)   ( (x)*(x) )

#define Pi       3.141592653589793238462643

#define KB       1024
#define N_MAX    4*1024*KB

/*
    1KB =    1024
    2KB =    2048
    4KB =    4096
    8KB =    8192
   16KB =   16384
   32KB =   32768
   64KB =   65536
  128KB =  131072
  256KB =  262144
  512KB =  524288
 1024KB = 1048576   ->   1MB
*/

int    i, k, CMCORR=1, N;

ldiv_t tmp_i;

double m[N_MAX], x[N_MAX][3], v[N_MAX][3],
       mcm, xcm[3], vcm[3];

double X,  Y,  Z,
       Vx, Vy, Vz,
       X1,X2,X3,X4,X5,X6,X7,
       R, Ve, V,
       M=1.0, eps=0.0,
       EK=0.0, EK_teor,
       conv,
       tmp;
        
FILE   *out;

/***************************************************************/

/***************************************************************/
/* RAND_MAX = 2147483647 */
/* my_rand  :  0.0 - 1.0 */

double my_rand(void)
{
return( (double)(rand()/(double)RAND_MAX) );
}
/***************************************************************/


/***************************************************************/
/***************************************************************/
/***************************************************************/
int main(int argc, char *argv[])
{

	if(argc == 2)
	{
		N = atoi(argv[1]);
	}
	else
	{
		printf("usage: ''gen-plum.exe 16'' (16KB particle) \n\n");
		printf("Warning!!! using the default number of stars: 16384 (16KB) \n");
		N = 16;
		//  exit(argc);
	}


	N = N*KB;


	/* VIR = 2*EK + EG ~ 0 => ETOT = EK + EG = -EK = EG/2 */

	/* G = M = R = 1 => E_tot = -3*Pi/64 */
	//  conv = 1.0; 
	//  EK_teor = 3.0*Pi/64.0;

	/* G = M = 1; R = 3*Pi/16 => E_tot = -0.25 */
	conv = 3.0*Pi/16.0;
	EK_teor = 0.25;



	/* INIT the rand() !!! */
	srand(19640916);                 /* it is just my birthday :-) */
	/* srand(time(NULL)); */ 


	i = 0;

	while(i < N)
	{

		X1 = my_rand(); X2 = my_rand(); X3 = my_rand();

		R = 1.0/sqrt( (pow(X1,-2.0/3.0) - 1.0) );

		if(R < 100.0)
		{

			Z = (1.0 - 2.0*X2)*R;
			X = sqrt(R*R - Z*Z) * cos(2.0*M_PI*X3);
			Y = sqrt(R*R - Z*Z) * sin(2.0*M_PI*X3);

			Ve = sqrt(2.0)*pow( (1.0 + R*R), -0.25 );


			X4 = 0.0; X5 = 0.0;

			while( 0.1*X5 >= X4*X4*pow( (1.0-X4*X4), 3.5) )
			{
				X4 = my_rand(); X5 = my_rand();
			} 


			V = Ve*X4;

			X6 = my_rand();
			X7 = my_rand();

			Vz = (1.0 - 2.0*X6)*V;
			Vx = sqrt(V*V - Vz*Vz) * cos(2.0*M_PI*X7);
			Vy = sqrt(V*V - Vz*Vz) * sin(2.0*M_PI*X7);

			X *= conv; Y *= conv; Z *= conv;    
			Vx /= sqrt(conv); Vy /= sqrt(conv); Vz /= sqrt(conv);


			m[i] = M/N;

			x[i][0] = X;
			x[i][1] = Y;
			x[i][2] = Z;

			v[i][0] = Vx;
			v[i][1] = Vy;
			v[i][2] = Vz;

			/*
			   tmp_i = ldiv(i, 256);
			   if(tmp_i.rem == 0) printf("i = %d \n", i);
			   */

			tmp_i = ldiv(i, N/64);

			if(tmp_i.rem == 0) 
			{
				printf(".");
				fflush(stdout);
			}

			i++;

		} /* if(r < 100.0) */

	} /* while(i < N) */





	if(CMCORR == 1)
	{

		mcm = 0.0;

		for(k=0;k<3;k++) 
		{
			xcm[k] = 0.0; vcm[k] = 0.0;
		} /* k */

		for(i=0; i<N; i++)
		{

			mcm += m[i];

			for(k=0;k<3;k++) 
			{
				xcm[k] += m[i] * x[i][k]; vcm[k] += m[i] * v[i][k];
			} /* k */

		}  /* i */


		for(k=0;k<3;k++) 
		{
			xcm[k] /= mcm; vcm[k] /= mcm;
		} /* k */


		for(i=0; i<N; i++)
		{

			for(k=0;k<3;k++) 
			{
				x[i][k] -= xcm[k]; v[i][k] -= vcm[k];
			} /* k */

		} /* i */

	} /* if(CMCORR == 1) */




#if 0
	out = fopen("data.inp","w");
#else
	{
		static char fname[256];
		snprintf(fname, 256, "pl%dk.dat", N/1024);
		out = fopen(fname, "w");
		assert(out);
	}
#endif

	fprintf(out,"%04d \n", 0);
	fprintf(out,"%06d \n", N);
	fprintf(out,"%.16E \n", 0.0);

	for(i=0; i<N; i++)
	{
#if 0
		fprintf(out,"%06d   %.16E   % .16E % .16E % .16E   % .16E % .16E % .16E \n", 
				i, m[i], x[i][0], x[i][1], x[i][2], v[i][0], v[i][1], v[i][2]);
#else
		const char *fmt = "%6d    %A    %A %A %A    %A %A %A    NAN\n";
		fprintf(out, fmt, i, m[i], x[i][0], x[i][1], x[i][2], v[i][0], v[i][1], v[i][2]);
#endif

		EK += 0.5*m[i] * (v[i][0]*v[i][0] + v[i][1]*v[i][1] + v[i][2]*v[i][2]);
	}



	printf("\n");
	printf("N = %d   EK = % .6E  EK_teor = % .6E  DE/EK = % .6E \n",
			N, EK, EK_teor, (EK-EK_teor)/EK_teor);



	fclose(out);

	return(0);

}
/***************************************************************/
/***************************************************************/
/***************************************************************/
