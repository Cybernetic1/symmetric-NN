// New implementation of Symmetric Neural Network based on the idea from DeepSet and PointNet,
// ie, that any symmetric function f(x,y,...) can be represented in the form g(h(x) + h(y) + ...)
// where g and h are non-linear functions (implemented as deep neural networks)

// The entire network is composed of a single g-network and multiple h-networks.
// The g-network is just a regular NN.
// The h-networks are summed over M inputs, where M is the Multiplicity

// TO-DO:

#include <cmath>
#include <cassert>
#include <array>
#include <algorithm>
#include <random>
#include "feedforward-NN.h"

using namespace std;

extern NNET *create_NN(int, int *);
extern void free_NN(NNET *, int *);
extern void forward_prop_sigmoid(NNET *, int, array <double, N>);
extern void forward_prop_ReLU(NNET *, int, array <double, N>);
extern void forward_prop_softplus(NNET *, int, array <double, N>);
extern void back_prop(NNET *, double []);
extern void back_prop_ReLU(NNET *, double []);
extern void re_randomize(NNET *, int, int *);
extern double sigmoid(double);

// The input vector X (with multiplicity M) is an array of shape X[M][N]
// The C++ "array" structure is used because sort() cannot be applied on simple 2D arrays
array <array <double, N>, M> X = {};
array <double, N> Y;			// output vector of shape Y[N]

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

// For sorting elements of X in lexicographic order
bool compareX(array <double, N> x1, array <double, N> x2)
	{
	for (int i = 0; i < N; ++i)
		if (x1[i] < x2[i])
			return true;
		else if (x1[i] > x2[i])
			return false;
		else
			assert(false);
	assert(false);
	}

double target_func(array <array <double, N>, M> x, double y[N])
// dim X = M × N, dim Y = N
	{
	// **** Sort input elements
	sort(X.begin(), X.begin() + M, compareX);

	#define k2 10.0				// k² where k = 10.0
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

NNET *Net_g;			// g-network
LAYER lastLayer_g;
NNET *Net_h[M];			// h-network × M times
LAYER lastLayer_h[M];
int neuronsPerLayer[] = {N, 5, 8, 5, N}; // first = input layer, last = output layer
int numLayers = sizeof (neuronsPerLayer) / sizeof (int);

// ***** Forward propagation (g- and h-networks)
void forward_prop()
	{
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
	}

// ***** We keep 2 arrays of errors, one is NEWER and one OLDER.
// The 2 arrays join together as one cyclic array.
// This enables us to compare the average errors of the NEW and OLD arrays,
// and calculate a RATIO of the two.
double error[N];					// error of single pass
#define err_cycle	1000			// how many errors to record for averaging
double errors1[err_cycle], errors2[err_cycle]; // two arrays for recording errors
double sum_err1 = 0.0, sum_err2 = 0.0; // sums of errors
int tail = 0; // index for cyclic arrays (last-in, first-out)
double avg_err;
char status[1024], *s;				// string buffer for status message

void record_error()
	{
	// ***** Calculate target value
	double ideal[N];
	target_func(X, ideal);

	// ***** Calculate RMS (root-mean-squared) error
	double sum_e2 = 0.0;
	for (int n = 0; n < N; ++n)
		{
		double e = ideal[n] - lastLayer_g.neurons[n].output;
		error[n] = e;
		sum_e2 += e * e;
		}
	double RMS_err = sqrt(sum_e2 / N);
	// printf("RMS error = %lf  ", RMS_error);

	// **** Calculate average error in a cyclic buffer of RMS errors
	// Update error arrays cyclically
	// (This is easier to understand by referring to the next block of code)
	sum_err2 -= errors2[tail];		// minus what is to be kicked out
	sum_err2 += errors1[tail];		// add what is to be inserted
	sum_err1 -= errors1[tail];		// minus what is to be kicked out
	sum_err1 += RMS_err;		// add newly inserted
	// printf("sum1, sum2 = %lf %lf\n", sum_err1, sum_err2);

	avg_err = sum_err1 / err_cycle;	// ignore cases where a full cycle hasn't been filled
	if (avg_err < 2.0)
		s += sprintf(s, "avg RMS err=%1.06lf, ", avg_err);
	else
		s += sprintf(s, "avg RMS err=%e, ", avg_err);

	// record new error in cyclic arrays
	errors2[tail] = errors1[tail];		// errors2 = OLDER array
	errors1[tail] = RMS_err;			// errors1 = NEWER array
	++tail;
	if (tail == err_cycle)				// loop back in cycle
		tail = 0;
	}

// ***** Back-propagation for the entire network (g- and h-)
void backward_prop()
	{
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
	}

int main(int argc, char **argv)
	{
	// ***** Initialize g- and h-networks
	Net_g = create_NN(numLayers, neuronsPerLayer);
	lastLayer_g = Net_g->layers[numLayers - 1];

	for (int m = 0; m < M; ++m)
		{
		Net_h[m] = create_NN(numLayers, neuronsPerLayer);
		lastLayer_h[m] = Net_h[m]->layers[numLayers - 1];
		}

	for (int i = 0; i < err_cycle; ++i) // clear errors to 0.0
		errors1[i] = errors2[i] = 0.0;

	// ***** Generate random target function
	// ie, generate the random centers of N Gaussian functions
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

	for (int l = 1; true; ++l)			// ***** Main loop
		{
		s = status + sprintf(status, "[%05d] ", l);

		// ***** Create M random X vectors (each of dim N)
		for (int m = 0; m < M; ++m)
			for (int i = 0; i < N; ++i)
				X[m][i] = random01();

		forward_prop();

		record_error();

		backward_prop();
		
		// **** If no convergence for a long time, re-randomize network
		if (l > 100000 && (isnan(avg_err) || avg_err > 2.0))
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
		if ((l % 5000) == 0)
			{
			double ratio = (sum_err2 - sum_err1) / sum_err1;
			if (ratio > 0)
				s += sprintf(s, "err ratio=%1.05lf", ratio);
			else
				s += sprintf(s, "err ratio=\x1b[31m%1.05lf\x1b[39;49m", ratio);
			//if (isnan(ratio))
			//	break;
			}

		if ((l % 5000) == 0) // display status periodically
			{
			// s += sprintf(s, "average error=%e", avg_err);
			printf("%s\n", status);
			}
		}

	// ***** Free allocated memory
	free_NN(Net_g, neuronsPerLayer);
	for (int m = 0; m < M; ++m)
		free_NN(Net_h[m], neuronsPerLayer);
	}
