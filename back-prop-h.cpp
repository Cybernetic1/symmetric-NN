// ******** back-propagation for the h-networks ********
// There are M such h-networks, they share the same weights,
// but have different outputs and local gradients (stored as arrays inside the data structure)

#include <cmath>
#include <cassert>
#include <array>
#include "h-network.h"

using namespace std;

extern double randomWeight();
extern double sigmoid(double);
extern double rectifier(double);
extern double softplus(double);
extern double d_softplus(double);

#define Eta 0.01			// learning rate
#define BIASINPUT 1.0		// input for bias. It's always 1.
#define Steepness 3.0		// parameter of sigmoid function & its derivative
#define Leakage 0.1			// parameter of ReLU function & its derivative

#define LastAct	true		// If false, activation function DISABLED on output layer

//****************************create neural network*********************//
// GIVEN: how many layers, and how many neurons in each layer
NNET_h *create_NN_h(int numLayers, int *NN_topology)
	{
	NNET_h *net = (NNET_h *) malloc(sizeof (NNET_h));
	srand(time(NULL));
	net->numLayers = numLayers;

	assert(numLayers >= 3);

	net->layers = (LAYER_h *) malloc(numLayers * sizeof (LAYER_h));
	//construct input layer, no weights
	net->layers[0].numNeurons = NN_topology[0];
	net->layers[0].neurons = (NEURON_h *) malloc(NN_topology[0] * sizeof (NEURON_h));

	//construct hidden layers
	for (int l = 1; l < numLayers; ++l) //construct layers
		{
		net->layers[l].neurons = (NEURON_h *) malloc(NN_topology[l] * sizeof (NEURON_h));
		net->layers[l].numNeurons = NN_topology[l];
		for (int n = 0; n < NN_topology[l]; ++n) // construct each neuron in the layer
			{
			net->layers[l].neurons[n].weights =
					(double *) malloc((NN_topology[l - 1] + 1) * sizeof (double));
			for (int i = 1; i <= NN_topology[l - 1]; ++i)
				//when i = 0, it's bias weight (this can be ignored)
				net->layers[l].neurons[n].weights[i] = randomWeight();
			}
		}
	return net;
	}

void re_randomize_h(NNET_h *net, int numLayers, int *NN_topology)
	{
	srand(time(NULL));

	for (int l = 1; l < numLayers; ++l)							// for each layer
		for (int n = 0; n < NN_topology[l]; ++n)				// for each neuron
			for (int i = 0; i <= NN_topology[l - 1]; ++i)	// for each weight
				net->layers[l].neurons[n].weights[i] = randomWeight();
	}

void free_NN_h(NNET_h *net, int *NN_topology)
	{
	// for input layer
	free(net->layers[0].neurons);

	// for each hidden layer
	int numLayers = net->numLayers;
	for (int l = 1; l < numLayers; l++) // for each layer
		{
		for (int n = 0; n < NN_topology[l]; n++) // for each neuron in the layer
			{
			free(net->layers[l].neurons[n].weights);
			}
		free(net->layers[l].neurons);
		}

	// free all layers
	free(net->layers);

	// free the whole net
	free(net);
	}

//**************************** forward-propagation ***************************//

void forward_prop_sigmoid_h(NNET_h *net, int dim_X, array <array <double, N>, M> X)
	{
	for (int m = 0; m < M; ++m)		// for each multiplicity
		{
		// set the output of input layer
		for (int i = 0; i < dim_X; ++i)
			net->layers[0].neurons[i].output[m] = X[m][i];

		// calculate output from hidden layers to output layer
		for (int l = 1; l < net->numLayers; l++)
			{
			for (int n = 0; n < net->layers[l].numNeurons; n++)
				{
				double v = 0.0f; //induced local field for neurons
				// calculate v, which is the sum of the product of input and weights
				for (int k = 0; k <= net->layers[l - 1].numNeurons; k++)
					{
					if (k == 0)
						v += net->layers[l].neurons[n].weights[k] * BIASINPUT;
					else
						v += net->layers[l].neurons[n].weights[k] *
							net->layers[l - 1].neurons[k - 1].output[m];
					}

				// For the last layer, skip the sigmoid function
				// Note: this idea seems to destroy back-prop convergence

				if (!LastAct && l == net->numLayers - 1)
					{
					net->layers[l].neurons[n].output[m] = v;
					net->layers[l].neurons[n].grad[m] = 1.0f;
					}
				else
					{
					double output = sigmoid(v);
					net->layers[l].neurons[n].output[m] = output;
					net->layers[l].neurons[n].grad[m] = Steepness * output * (1.0 - output);
					}
				}
			}
		}
	}

// Same as above, except with soft_plus activation function
void forward_prop_softplus_h(NNET_h *net, int dim_X, array <array <double, N>, M> X)
	{
	for (int m = 0; m < M; ++m)		// for each multiplicity
		{
		// set the output of input layer
		for (int i = 0; i < dim_X; ++i)
			net->layers[0].neurons[i].output[m] = X[m][i];

		// calculate output from hidden layers to output layer
		for (int l = 1; l < net->numLayers; l++)
			{
			for (int n = 0; n < net->layers[l].numNeurons; n++)
				{
				double v = 0.0; // induced local field for neurons
				// calculate v, which is the sum of the product of input and weights
				for (int k = 0; k <= net->layers[l - 1].numNeurons; k++)
					{
					if (k == 0)
						v += net->layers[l].neurons[n].weights[k] * BIASINPUT;
					else
						v += net->layers[l].neurons[n].weights[k] *
							net->layers[l - 1].neurons[k - 1].output[m];
					}

				net->layers[l].neurons[n].output[m] = softplus(v);
				net->layers[l].neurons[n].grad[m] = d_softplus(v);
				}
			}
		}
	}

// Same as above, except with rectifier activation function
// ReLU = "rectified linear unit"
void forward_prop_ReLU_h(NNET_h *net, int dim_X, array <array <double, N>, M> X)
	{
	for (int m = 0; m < M; ++m)		// for each multiplicity
		{
		// set the output of input layer
		for (int i = 0; i < dim_X; ++i)
			net->layers[0].neurons[i].output[m] = X[m][i];

		// calculate output from hidden layers to output layer
		for (int l = 1; l < net->numLayers; l++)
			{
			for (int n = 0; n < net->layers[l].numNeurons; n++)
				{
				double v = 0.0; // induced local field for neurons
				// calculate v, which is the sum of the product of input and weights
				for (int k = 0; k <= net->layers[l - 1].numNeurons; k++)
					{
					if (k == 0)
						v += net->layers[l].neurons[n].weights[k] * BIASINPUT;
					else
						v += net->layers[l].neurons[n].weights[k] *
							net->layers[l - 1].neurons[k - 1].output[m];
					}

				net->layers[l].neurons[n].output[m] = rectifier(v);

				// This is to prepare for back-prop
				if (v < 0.0)
					net->layers[l].neurons[n].grad[m] = Leakage;
				// if (v > 1.0)
				//	net->layers[l].neurons[n].grad[m] = Leakage;
				else
					net->layers[l].neurons[n].grad[m] = 1.0;
				}
			}
		}
	}

//****************************** back-propagation ***************************//
// In the update formula, we need to adjust by "η ∙ input ∙ ∇", where η is the learning rate.
// The value of		∇_j = σ'(summed input) Σ_i W_ji ∇_i
// where σ is the sigmoid function, σ' is its derivative.  This formula is obtained directly
// from differentiating the error E with respect to the weights W.

// The meaning of del (∇) is the "local gradient".  At the output layer, ∇ is equal to
// the derivative σ'(summed inputs) times the error signal, while on hidden layers it is
// equal to the derivative times the weighted sum of the ∇'s from the "next" layers.
// From the algorithmic point of view, ∇ is derivative of the error with respect to the
// summed inputs (for that particular neuron).  It changes for every input instance because
// the error is dependent on the NN's raw input.  So, for each raw input instance, the
// "local gradient" keeps changing.

void back_prop_h(NNET_h *net, double *errors)
	{
	int numLayers = net->numLayers;
	LAYER_h lastLayer = net->layers[numLayers - 1];

	for (int m = 0; m < M; ++m)		// for each multiplicity
		{
		// calculate gradient for output layer
		for (int n = 0; n < lastLayer.numNeurons; ++n)
			{
			// double output = lastLayer.neurons[n].output;
			//for output layer, ∇ = sign(y)∙error
			// .grad has been prepared in forward-prop
			lastLayer.neurons[n].grad[m] *= errors[n];
			}

		// calculate gradient for hidden layers
		for (int l = numLayers - 2; l > 0; --l)		// for each hidden layer
			{
			for (int n = 0; n < net->layers[l].numNeurons; n++)		// for each neuron in layer
				{
				// double output = net->layers[l].neurons[n].output;
				double sum = 0.0f;
				LAYER_h prevLayer = net->layers[l + 1];
				for (int i = 0; i < prevLayer.numNeurons; i++)		// for each weight
					{
					sum += prevLayer.neurons[i].weights[n + 1]		// ignore weights[0] = bias
							* prevLayer.neurons[i].grad[m];
					}
				// .grad has been prepared in forward-prop
				net->layers[l].neurons[n].grad[m] *= sum;
				}
			}

		// update all weights
		for (int l = 1; l < numLayers; ++l)		// except for 0th layer which has no weights
			{
			for (int n = 0; n < net->layers[l].numNeurons; n++)		// for each neuron
				{
				net->layers[l].neurons[n].weights[0] += Eta *
						net->layers[l].neurons[n].grad[m] * 1.0;		// 1.0f = bias input
				for (int i = 0; i < net->layers[l - 1].numNeurons; i++)	// for each weight
					{
					double inputForThisNeuron = net->layers[l - 1].neurons[i].output[m];
					net->layers[l].neurons[n].weights[i + 1] += Eta *
							net->layers[l].neurons[n].grad[m] * inputForThisNeuron;
					}
				}
			}
		}
	}
