// New implementation of Symmetric Neural Network based on the idea from DeepSet and PointNet,
// ie, that any symmetric function f(x,y,...) can be represented in the form g(h(x) + h(y) + ...)
// where g and h are non-linear functions (implemented as deep neural networks)

// The entire network is composed of a single g-network and multiple h-networks.
// The g-network is just a regular NN.
// The h-networks are summed over M inputs, where M is the Multiplicity

// TO-DO:
// * Sort inputs to target function

#include <iostream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>
#include "feedforward-NN.h"

using namespace std;

extern NNET *create_NN(int, int *);
extern void free_NN(NNET *, int *);
extern void forward_prop_sigmoid(NNET *, int, double *);
extern void forward_prop_ReLU(NNET *, int, double *);
extern void forward_prop_softplus(NNET *, int, double *);
extern void back_prop(NNET *, double *);
extern void back_prop_ReLU(NNET *, double *);
extern double calc_error(NNET *, double *, double *);
extern void re_randomize(NNET *, int, int *);
extern double sigmoid(double);

// The input vector is of dimension N × M, where
// M = number of input elements, which I also call 'multiplicity':
#define M		5
// N = dimension of the embedding / encoding of each input element:
#define N		2

double X[M][N];
double Y[N];

double random01()		// create random number in standard interval, eg. [-1,1]
	{
	return rand() * 2.0 / (float) RAND_MAX - 1.0;
	}

// ***** To test the NN, we use as target function a sum of N-dimensional Gaussian
// functions with random centers.  The sum would consists of N such functions.
// The formula of the target function is:
//		f(x) = (1/N) ∑ exp-(kr)²
// where r = ‖ x-c ‖ where c is the center of the Gaussian function
// (k is the 'narrowness' of the Gaussian, for which 10.0 may be a nice value)
// Also, this function should be invariant under permutations.

// Centers of N Gaussian functions:
double c[N][M][N];

double target_func(double x[M][N], double y[N])		// dim X = M × N, dim Y = N
	{
	// **** Sort input elements
	

	#define k2 4.0				// k² where k = 10.0
	for (int i = 0; i < N; ++i)		// calculate each component of y
		{
		y[i] = 0.0;
		for (int n = 0; n < N; ++n)		// for each Gaussian function; there are N of them
			{
			double r2 = 0.0;			// r² = ‖ x-c ‖²
			for (int m = 0; m < M; ++m)
				r2 += pow(x[m][i] - c[n][m][i], 2);
			y[i] += exp(- k2 * r2);		// add one Gaussian to y[i]
			}
		y[i] /= N;
		}
	}

// To test convergence, we record the sum of squared errors for the last M and last M..2M
// trials, then compare their ratio.

// Success: time 5:58, topology = {2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1} (13 layers)
//			ReLU units, learning rate 0.05, leakage 0.0
#define ForwardPropMethod	forward_prop_sigmoid
#define ErrorThreshold		0.02

int main(int argc, char **argv)
	{
	int neuronsPerLayer[] = {N, 10, 10, 10, N}; // first = input layer, last = output layer
	int numLayers = sizeof (neuronsPerLayer) / sizeof (int);

	NNET *Net_g = create_NN(numLayers, neuronsPerLayer);
	LAYER lastLayer_g = Net_g->layers[numLayers - 1];

	NNET *Net_h[M];
	LAYER lastLayer_h[M];
	for (int m = 0; m < M; ++m)
		{
		Net_h[m] = create_NN(numLayers, neuronsPerLayer);
		lastLayer_h[m] = Net_h[m]->layers[numLayers - 1];
		}

	// ***** We keep 2 arrays of errors, one is NEWER and one OLDER.
	// The 2 arrays join together as one cyclic array.
	// This enables us to compare the average errors of the NEW and OLD arrays,
	// and calculate a RATIO of the two.
	#define err_cycle	50			// how many errors to record for averaging
	double errors1[err_cycle], errors2[err_cycle]; // two arrays for recording errors
	double sum_err1 = 0.0, sum_err2 = 0.0; // sums of errors
	int tail = 0; // index for cyclic arrays (last-in, first-out)

	double training_err = 0.0;
	for (int i = 0; i < err_cycle; ++i) // clear errors to 0.0
		errors1[i] = errors2[i] = 0.0;

	srand(time(NULL));				// random seed

	// ***** First we generate the random centers of N Gaussian functions
	// Note that these centers are 'sorted' such that the resulting points all reside in the
	// 'symmetric' region.
	for (int n = 0; n < N; ++n)
		for (int m = 0; m < M; ++m)
			{
			for (int i = 0; i < N; ++i)
				c[n][m][i] = random01();
			sort(c[n][m], c[n][m] + N);
			}

	for (int m = 0; m < M; ++m)
		{
		for (int i = 0; i < N; ++i)
			printf("%f ", c[0][m][i]);
		printf("\n");
		}

	char status[1024], *s;				// string buffer for status message

	for (int l = 1; true; ++l)			// main loop
		{
		s = status + sprintf(status, "[%05d] ", l);

		// ***** Create M random X vectors (each of dim N)
		for (int m = 0; m < M; ++m)
			for (int i = 0; i < N; ++i)
				// X[k] = (rand() / (float) RAND_MAX);
				X[m][i] = random01();

		// ***** Forward propagation

		for (int m = 0; m < M; ++m)
			ForwardPropMethod(Net_h[m], N, X[m]);			// X[m] is updated

		// Bridge layer: add up X[m] to get Y
		for (int i = 0; i < N; ++i)
			{
			Y[i] = 0;
			for (int m = 0; m < M; ++m)
				{
				Y[i] += X[m][i];
				}
			}

		ForwardPropMethod(Net_g, N, Y);					// Y is updated

		// ***** Calculate target value

		double ideal[N];
		target_func(X, ideal);

		// ***** Calculate the RMS (root-mean-squared) error

		double error[N];
		double sum_e2 = 0.0;
		for (int n = 0; n < N; ++n)
			{
			double e = ideal[n] - lastLayer_g.neurons[n].output;
			error[n] = e;
			sum_e2 += e * e;
			}
		double RMS_error = sqrt(sum_e2 / N);

		// **** Calculate average error in a cyclic buffer of RMS errors

		training_err = RMS_error;		// ? dunno why I used 2 variable names
		// printf("RMS error = %lf  ", training_err);

		// **** Update error arrays cyclically
		// (This is easier to understand by referring to the next block of code)
		sum_err2 -= errors2[tail];		// minus what is to be kicked out
		sum_err2 += errors1[tail];		// add what is to be inserted
		sum_err1 -= errors1[tail];		// minus what is to be kicked out
		sum_err1 += training_err;		// add newly inserted
		// printf("sum1, sum2 = %lf %lf\n", sum_err1, sum_err2);

		double avg_err = (l < err_cycle) ? (sum_err1 / l) : (sum_err1 / err_cycle);
		if (avg_err < 2.0)
			s += sprintf(s, "average RMS error=%1.06lf, ", avg_err);
		else
			s += sprintf(s, "average RMS error=%e, ", avg_err);

		// record new error in cyclic arrays
		errors2[tail] = errors1[tail];		// errors2 = OLDER array
		errors1[tail] = training_err;		// errors1 = NEWER array
		++tail;
		if (tail == err_cycle)				// loop back in cycle
			tail = 0;

		// ***** Back-propagation
		back_prop(Net_g, error);

		// ***** Bridge between g_network and h_networks
		// This emulates the back-prop algorithm for 1 layer (see "g-and-h-networks.png")
		// and the error is simply copied to each h-network
		LAYER prevLayer = Net_g->layers[0];
		for (int n = 0; n < N; n++)		// for error in bridge layer
			{
			// local gradient ≡ 1 for the bridge layer, because there's no sigmoid function
			error[n] = prevLayer.neurons[n].grad;
			}

		// ***** Back-propagate the h-networks
		for (int m = 0; m < M; m++)
			back_prop(Net_h[m], error);

		// ***** Updated weights of the h-networks is now averaged:
		for (int l = 1; l < numLayers; ++l)		// for all layers except 0th which has no weights
			for (int n = 0; n < Net_h[0]->layers[l].numNeurons; n++)	// for each neuron
				for (int i = 0; i < Net_h[0]->layers[l - 1].numNeurons; i++) // for each weight
					{
					// Calculate average value
					double avg = 0.0f;
					for (int m = 0; m < M; ++m)		// for each multiplicity
						avg += Net_h[m]->layers[l].neurons[n].weights[i + 1];
					avg /= M;

					// Overwrite all weights with average value
					for (int m = 0; m < M; ++m)		// for each multiplicity
						Net_h[m]->layers[l].neurons[n].weights[i + 1] = avg;
					}

		// ***** Test the network
		if ((l % 200) == -1)	// 0 = enable this part, -1 = disable
			{
			// Testing set
			double test_err = 0.0;
			#define numTests 50
			for (int j = 0; j < numTests; ++j)
				{
				// Create random K vector
				for (int m = 0; m < M; ++m)
					for (int i = 0; i < N; ++i)
						X[m][i] = random01();
				// plot_tester(K[0], K[1]);

				ForwardPropMethod(Net_g, N, X[0]);

				// Desired value = K_star
				double single_err = 0.0;
				for (int k = 0; k < 1; ++k)
					{
					// double ideal = 1.0f - (0.5f - K[0]) * (0.5f - K[1]);
					double ideal = 0.0;
					// double ideal = K[k];				/* identity function */

					// Difference between actual outcome and desired value:
					double error = 0.0;  // ideal - lastLayer_h.neurons[k].output;

					single_err += fabs(error); // record sum of errors
					}
				test_err += single_err;
				}
			test_err /= ((double) numTests);
			if (test_err < 2.0)
				s += sprintf(s, "random test |e|=%1.06lf, ", test_err);
			else
				s += sprintf(s, "random test |e|=%e, ", test_err);
			if (test_err < ErrorThreshold)
				break;
			}

		// **** If no convergence for a long time...
		if (l > 50 && (isnan(avg_err) || avg_err > 10.0))
			{
			re_randomize(Net_h[0], numLayers, neuronsPerLayer);
			sum_err1 = 0.0; sum_err2 = 0.0;
			tail = 0;
			for (int j = 0; j < err_cycle; ++j) // clear errors to 0.0
				errors1[j] = errors2[j] = 0.0;
			l = 1;

			printf("\n****** Network re-randomized.\n");
			}

		// ***** periodically display the ratio of average errors
		if ((l % 1000) == 0)
			{
			double ratio = (sum_err2 - sum_err1) / sum_err1;
			if (ratio > 0)
				s += sprintf(s, "|e| ratio=%e\n", ratio);
			else
				s += sprintf(s, "|e| ratio=\x1b[31m%e\x1b[39;49m\n", ratio);
			//if (isnan(ratio))
			//	break;
			}

		if ((l % 1000) == 0) // display status periodically
			{
			s += sprintf(s, "average error=%e", avg_err);
			printf("%s\n", status);
			}

		// if (ratio - 0.5f < 0.0000001)	// ratio == 0.5 means stationary
		// if (test_err < 0.01)

		if (false)  // (userKey == 3)		// Re-start with new random weights
			{
			re_randomize(Net_h[0], numLayers, neuronsPerLayer);
			sum_err1 = 0.0; sum_err2 = 0.0;
			tail = 0;
			for (int j = 0; j < err_cycle; ++j) // clear errors to 0.0
				errors1[j] = errors2[j] = 0.0;
			l = 1;

			printf("\n****** Network re-randomized.\n");
			}
		}

	free_NN(Net_g, neuronsPerLayer);
	for (int m = 0; m < M; ++m)
		free_NN(Net_h[m], neuronsPerLayer);
	}
