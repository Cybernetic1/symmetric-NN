// N = dimension of the embedding / encoding of each input element:
#define N		2

typedef struct NEURON
	{
    double output;
    double *weights;
    double grad;		// "local gradient"
	} NEURON;

typedef struct LAYER
	{
    int numNeurons;
    NEURON *neurons;
	} LAYER;

typedef struct NNET
	{
    int numLayers;
    LAYER *layers;
	} NNET; //neural network
