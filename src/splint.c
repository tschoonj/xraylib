#include <stdio.h>
#include <stdlib.h>

void splint(float xa[], float ya[], float y2a[], int n, float x, float *y)
{
	int klo, khi, k;
	float h, b, a;

	if (x > xa[n]) {
	  *y = ya[n];
	  return;
	}

	if (x < xa[1]) {
	  *y = ya[1];
	  return;
	}

	klo = 1;
	khi = n;
	while (khi-klo > 1) {
		k = (khi + klo) >> 1;//division by 2
		if (xa[k] > x) khi = k;
		else klo = k;
	}
	//determine position of x in xa

	h = xa[khi] - xa[klo];
	if (h == 0.0) {
	  *y = (ya[klo] + ya[khi])/2.0;
	  return;
	}
	a = (xa[khi] - x) / h;
	b = (x - xa[klo]) / h;
	*y = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo]
	     + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
}

void splintd(double xa[], double ya[], double y2a[], int n, double x, double *y)
{
	int klo, khi, k;
	double h, b, a;
 

	if (x > xa[n]) {
	  *y = ya[n];
	  return;
	}

	if (x < xa[1]) {
	  *y = ya[1];
	  return;
	}

	klo = 1;
	khi = n;
	while (khi-klo > 1) {
		k = (khi + klo) >> 1;//division by 2
		if (xa[k] > x) khi = k;
		else klo = k;
	}
	//determine position of x in xa

	h = xa[khi] - xa[klo];
	if (h == 0.0) {
	  *y = (ya[klo] + ya[khi])/2.0;
	  return;
	}
	a = (xa[khi] - x) / h;
	b = (x - xa[klo]) / h;
	*y = a*ya[klo] + b*ya[khi] + ((a*a*a-a)*y2a[klo]
	     + (b*b*b-b)*y2a[khi])*(h*h)/6.0;
}


