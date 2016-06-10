#include<iostream>
#include<vector>
#include<cmath>
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

public:

	// get lengths of ANN layers, then sets weights and biases to 1 for given "dimensions"
	ANN(vector<int> &lengths)
	{
		numLayers = lengths.size();
		layerLengths = lengths;

		maxLength = 0;
		for (int m = 0; m < numLayers; m++)
			maxLength = (layerLengths[m] > maxLength ? layerLengths[m] : maxLength);

		//set biases; biases exist for first layer but shouldn't, so don't access [0]
		for (int i = 0; i < numLayers; i++) {
			vector<float> row;
			for (int j = 0; j < lengths[i]; j++)
				row.push_back(i*j);  //was 1
			biases.push_back(row);
		}

		//set all activations to 0
		for (int i = 0; i < numLayers; i++) {
			vector<float> row;
			for (int j = 0; j < lengths[i]; j++)
				row.push_back(0);
			activations.push_back(row);
		}

		//set weights; weights[n][][] correspond to neurons in biases[n+1][]
		for (int i = 0; i < numLayers; i++) {
			vector<vector<float>> row;

			for (int j = 0; j < lengths[i]; j++) {
				vector<float> subRow;
				for (int k = 0; i + 1 < numLayers && k < lengths[i + 1]; k++)
					subRow.push_back(1);
				row.push_back(subRow);
			}
			weights.push_back(row);
		}
	}

	//set ANN based on 3D weight matrix and bias matrix
	ANN(vector<vector<vector<float>>> &w, vector<vector<float>> &b)
	{
		//check if dimensions of w and b are compatible
		if (w.size() != b.size() - 1)
			return;
		
		for (int i = 0; i < w.size(); i++){
			if (w[i].size() != b[i].size())
				return;

			for (int j = 0; j < w[i].size(); j++) {
				if (w[i][j].size() != b[i + 1].size())
					return;
			}
		}

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
	}

	void printActivations()
	{
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

	//print all weights
	void printWeights()
	{
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

	//Sigmoid/sigma function
	float sigmoid(double z)
	{
		return 1 / (1 + exp(-z));
	}

	//Sigmoid derivative
	float sigmoidPrime(double z)
	{
		return sigmoid(z)*(1 - sigmoid(z));
	}

	void feedForward(vector<float> &input)
	{
		activations[0] = input;

		for (int i = 1; i < numLayers; i++) {
			for (int j = 0; j < activations[i].size(); j++) {
				float sum = 0;

				for (int k = 0; k < activations[i - 1].size(); k++)
					sum += weights[i - 1][k][j] * activations[i - 1][k];

				activations[i][j] = sigmoid(sum + biases[i][j]);
			}
		}
	}
};

int main()
{
	/*vector<int> len = {3, 2, 2};
	ANN network = ANN(len);
	network.printBiases();
	network.printWeights();
	network.printActivations();
	vector<float> in = {1, 0, .001f};
	network.feedForward(in);
	network.printActivations();

	cout << "----------------------------------------------------" << endl;

	vector<vector<float>> bB = { {0, 0, 0}, {1, 2} };
	vector<vector<vector<float>>> wW = { {{1, 2},{3, 4},{5, 6}} };

	ANN network2 = ANN(wW, bB);
	network2.printBiases();
	network2.printWeights();
	network2.printActivations();
	vector<float> in2 = {0, 3, .002f};
	network2.feedForward(in2);
	network2.printActivations();*/



	vector<vector<float>> bB3 = { { 0, 0, 0 },{ 0, 4 }, {0, 0} };
	vector<vector<vector<float>>> wW3 = { { { 1, 3 },{ -2, 2 },{ 3, 3} }, { {1, -1}, {-1, -2} } };

	ANN network3 = ANN(wW3, bB3);
	network3.printBiases();
	network3.printWeights();
	network3.printActivations();
	vector<float> in3 = { 1, 0, 0};
	network3.feedForward(in3);
	network3.printActivations();

	system("PAUSE");
}