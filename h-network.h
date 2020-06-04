// The input vector is of dimension N × M, where
// M = number of input elements, which I also call 'multiplicity':
#define M		5
// N = dimension of the embedding / encoding of each input element:
#define N		2

// ***** There are M copies of h-networks
// They share the same set of weights
// They have different outputs and local gradients (stored as M-dimensional arrays)
// Otherwise the data structure is same as the standard one

typedef struct NEURON_h
	{
    double output[M];
    double *weights;
    double grad[M];		// "local gradient"
	} NEURON_h;

typedef struct LAYER_h
	{
    int numNeurons;
    NEURON_h *neurons;
	} LAYER_h;

typedef struct NNET_h
	{
    int numLayers;
    LAYER_h *layers;
	} NNET_h; //neural network
