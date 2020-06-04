
//**********************struct for NEURON**********************************//
typedef struct NEURON
	{
    double output;
    double *weights;
    double grad;		// "local gradient"
	} NEURON;

//**********************struct for LAYER***********************************//
typedef struct LAYER
	{
    int numNeurons;
    NEURON *neurons;
	} LAYER;

//*********************struct for NNET************************************//
typedef struct NNET
	{
    int numLayers;
    LAYER *layers;
	} NNET; //neural network

// The input vector is of dimension N × M, where
// M = number of input elements, which I also call 'multiplicity':
#define M		5
// N = dimension of the embedding / encoding of each input element:
#define N		2

