#include<iostream>
#include<vector>
#include<cmath>
#include<random>
#include<ctime>
using namespace std;

class ANN
{
private:

	int numLayers;
	int maxLength;
	vector<int> layerLengths;

	vector<vector<float>> biases;
	vector<vector<vector<float>>> weights;
	vector<vector<float>> activations;
	vector<vector<float>> errors;

public:

	//--------------------CONSTRUCTORS--------------------

	// get lengths of ANN layers, then sets weights and biases randomly for given "dimensions", also sets activations and errors to 0
	ANN(vector<int> &lengths)
	{
		default_random_engine generatorB(time(NULL));
		normal_distribution<float> distB(0,1);

		numLayers = lengths.size();
		layerLengths = lengths;

		maxLength = 0;
		for (int m = 0; m < numLayers; m++)
			maxLength = (layerLengths[m] > maxLength ? layerLengths[m] : maxLength);

		//set biases; biases exist for first layer but shouldn't, so don't access [0]
		for (int i = 0; i < numLayers; i++) {
			vector<float> row;
			for (int j = 0; j < lengths[i]; j++)
				row.push_back( distB(generatorB) );  //was 1
			biases.push_back(row);
		}

		//set all activations to 0
		for (int i = 0; i < numLayers; i++) {
			vector<float> row;
			for (int j = 0; j < lengths[i]; j++)
				row.push_back( 0 );
			activations.push_back(row);
		}

		//set errors to 0
		for (int i = 0; i < numLayers; i++) {
			vector<float> row;
			for (int j = 0; j < lengths[i]; j++)
				row.push_back(0);
			errors.push_back(row);
		}

		//set weights; weights[n][][] correspond to neurons in biases[n+1][]
		for (int i = 0; i < numLayers - 1; i++) {
			vector<vector<float>> row;

			for (int j = 0; j < lengths[i]; j++) {
				vector<float> subRow;
				for (int k = 0; i + 1 < numLayers && k < lengths[i + 1]; k++)
					subRow.push_back( distB(generatorB) );
				row.push_back(subRow);
			}
			weights.push_back(row);
		}
	}

	//set ANN based on 3D weight matrix and bias matrix
	ANN(vector<vector<vector<float>>> &w, vector<vector<float>> &b)
	{
		//check if dimensions of w and b are compatible
		if (w.size() != b.size() - 1) {
			cout << "Error: incompatible dimensions." << endl;
			return;
		}
		
		//further check for dimension compatibility
		for (int i = 0; i < w.size(); i++){
			if (w[i].size() != b[i].size()) {
				cout << "Error: incompatible dimensions." << endl;
				return;
			}

			for (int j = 0; j < w[i].size(); j++) {
				if (w[i][j].size() != b[i + 1].size()) {
					cout << "Error: incompatible dimensions." << endl;
					return;
				}
			}
		}

		//initialize ANN
		weights = w;
		biases = b;
		numLayers = b.size();

		for (int i = 0; i < b.size(); i++)
			layerLengths.push_back(b[i].size());

		maxLength = 0;
		for (int m = 0; m < numLayers; m++)
			maxLength = (layerLengths[m] > maxLength ? layerLengths[m] : maxLength);

		for (int i = 0; i < numLayers; i++) {
			vector<float> row;
			for (int j = 0; j < layerLengths[i]; j++)
				row.push_back(0);
			activations.push_back(row);
		}

		for (int i = 0; i < numLayers; i++) {
			vector<float> row;
			for (int j = 0; j < layerLengths[i]; j++)
				row.push_back(0);
			errors.push_back(row);
		}
	}

	//--------------------PRINT FUNCTIONS--------------------

	//print all errors, X for first (input) layer
	void printErrors()
	{
		cout << "Errors:" << endl;

		for (unsigned j = 0; j < maxLength; j++) {
			for (unsigned i = 0; i < errors.size(); i++) {
				if (i == 0 && j < errors[i].size())
					cout << "X ";
				else if (j >= errors[i].size())
					cout << "  ";
				else
					cout << errors[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	//print all activations
	void printActivations()
	{
		cout << "Activations:" << endl;

		for (unsigned j = 0; j < maxLength; j++) {
			for (unsigned i = 0; i < activations.size(); i++) {
				if (j >= activations[i].size())
					cout << "  ";
				else
					cout << activations[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	//print all biases; first layer is Xs
	void printBiases()
	{
		cout << "Biases:" << endl;

		for (unsigned j = 0; j < maxLength; j++){
			for (unsigned i = 0; i < biases.size(); i++) {
					if (i == 0 && j < biases[i].size())
						cout << "X ";
					else if (j >= biases[i].size())
						cout << "  ";
					else
						cout << biases[i][j] << " ";
			}
				cout << endl;
		}
		cout << endl;
	}

	//print all weights
	void printWeights()
	{
		cout << "Weights:" << endl;

		for (int i = 0; i < maxLength; i++) {
			for (int j = 0; j < numLayers - 1; j++) {
				for (int k = 0; k < layerLengths[j + 1]; k++)
					(i < layerLengths[j] ? cout << weights[j][i][k] << " " : cout << "  ");
				cout << "  ";
			}
			cout << endl;
		}
		cout << endl;
	}

	//print weights leading into layer [w] from [w - 1]
	void printWeightsForLayer(int w)
	{
		if (w <= 0 || numLayers <= w)
			return;

		for (int j = 0; j < layerLengths[w - 1]; j++){
			for (int k = 0; k < layerLengths[w]; k++)
				cout << weights[w - 1][j][k] << " ";
			cout << endl;
		}
		cout << endl;
	}

	//--------------------UPDATING FUNCTIONS--------------------

	//calculate activations for all layers
	void feedForward(vector<float> &x)
	{
		activations[0] = x;

		for (int i = 1; i < numLayers; i++) {
			for (int j = 0; j < activations[i].size(); j++) {
				float sum = 0;

				for (int k = 0; k < activations[i - 1].size(); k++)
					sum += weights[i - 1][k][j] * activations[i - 1][k];

				activations[i][j] = 1/(1 + exp(-(sum + biases[i][j])));
			}
		}
	}

	//find output vector for final layer
	void outputError(vector<float> &y)
	{
		int last = biases.size() - 1;

		for (int i = 0; i < biases[last].size(); i++)
			errors[last][i] = (activations[last][i] - y[i]) * (activations[last][i] * (1 - activations[last][i]));
	}

	//calculate errors of previous layers
	void backpropogate()
	{
		for (int i = errors.size() - 2; i > 0; i--) {
			for (int j = 0; j < errors[i].size(); j++) {
				float dot = 0;

				for (int k = 0; k < weights[i][j].size(); k++)
					dot += (weights[i][j][k] * errors[i + 1][k]);

				errors[i][j] = dot * (activations[i][j] * (1 - activations[i][j]));
			}
		}
	}


	//gradient descent
	void descent(float rate)
	{
		//adjust weights
		for (int i = 0; i < weights.size(); i++) {
			for (int j = 0; j < weights[i].size(); j++) {
				for (int k = 0; k < activations[i + 1].size(); k++) 
					weights[i][j][k] -= rate * activations[i][j] * errors[i+1][k];
			}
		}

		//adjust biases
		for (int i = 0; i < biases.size(); i++) {
			for (int j = 0; j < biases[i].size(); j++)
				biases[i][j] -= rate * errors[i][j];
		}
	}

	//calls all four steps at once for one training example
	void updateOne(vector<float> &in, vector<float> &out, float rate)
	{	
		feedForward(in);
		outputError(out);
		backpropogate();
		descent(rate);
	}

	//calls all four steps at once for a training example set
	void update(vector<vector<float>> &in, vector<vector<float>> &out, float rate, int n)
	{
		//see if there is one output for each input
		if (in.size() != out.size()) {
			cout << "Error: incompatible input/output size." << endl;
			return;
		}

		//make sure input and output lengths are correct
		for (int i = 0; i < in.size(); i++) {
			if (in[i].size() != biases[0].size()) {
				cout << "Error: invalid input length." << endl;
				return;
			}
			else if (out[i].size() != biases[biases.size() - 1].size()) {
				cout << "Error: invalid expected output length." << endl;
				return;
			}
		}

		//call update for each example alternating n times
		for (int a = 0; a < n; a++) {
			for (int i = 0; i < in.size(); i++)
				updateOne(in[i], out[i], rate);
		}
	}

	//calls all four steps at once for set of training pairs (x, y)
	void updatePair(vector<vector<vector<float>>> pair, float rate, int n)
	{
		//check to see if pairs are given and then if they match dimensions of input and output layers
		for (int i = 0; i < pair.size(); i++) {
			if (pair[i].size() != 2)
				return;

			if (pair[i][0].size() != biases[0].size() || pair[i][1].size() != biases[biases.size() - 1].size())
				return;
		}

		for (int a = 0; a < n; a++) {
			for (int i = 0; i < pair.size(); i++)
				updateOne(pair[i][0], pair[i][1], rate);
		}
	}
};

int main()
{
	/*vector<vector<float>> theBiases = { {0, 0, 0}, {0, 4}, {0, 0} };
	vector<vector<vector<float>>> theWeights = { { {1, 3},{-2, 2},{3, 3} }, { {1, -1},{-1, -2} } };

	vector<vector<float>> theB = { {0, 0, 0}, {0, 0}, {0, 0} };
	vector<vector<vector<float>>> theW = { { {100, -100},{-100, 100},{-100, 100} }, { {100, -100},{-100, 100} } };


	vector<int> len = {3, 2, 2};

	vector<float> input1 = {1, 0, 0};
	vector<float> output1 = {1, 0};

	vector<float> input2 = {0, 1, 0};
	vector<float> output2 = { 0, 1 };

	vector<float> input3 = {0, 0, 1};
	vector<float> output3 = {0, 1};

	vector<vector<float>> input = {input1, input2, input3};
	vector<vector<float>> output = {output1, output2, output3};

	ANN network1 = ANN(len);
	ANN network1 = ANN(theW, theB);
	network1.printBiases();
	network1.printWeights();
	network1.update(input, output, 10, 10000);
	network1.printBiases();
	network1.printWeights();
	network1.feedForward(input1);
	network1.printActivations();
	network1.printErrors();*/

	vector<int> len2 = {5, 4, 3};
	vector<vector<float>> inp = { {1, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 0, 1} };
	vector<vector<float>> out = { {1, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {0, 1, 0} };

	/*ANN network2 = ANN(len2);
	network2.update(inp, out, 10, 1000);
	vector<float> feed = {1, 0, 0, 0, 0};
	network2.feedForward(feed);
	network2.printActivations();
	network2.printErrors();*/

	vector<vector<vector<float>>> train = { {{1, 0, 0, 0, 0}, {1, 0, 1}}, {{0, 1, 0, 0, 0}, {1, 0, 1}}, {{0, 0, 1, 0, 0}, {1, 1, 1}}, {{0, 0, 0, 1, 0}, {0, 1, 1}}, {{0, 0, 0, 0, 1}, {0, 1, 0}} };


	ANN network3 = ANN(len2);
	network3.updatePair(train, 1, 1000);
	vector<float> feed = { 0, 0, 0, 0, 1 };
	network3.feedForward(feed);
	network3.printActivations();
	network3.printErrors();

	system("PAUSE");
}