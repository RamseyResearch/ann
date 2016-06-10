#include<iostream>
#include<vector>
using namespace std;

class ANN
{
private:

	int numLayers;
	int maxLength;
	vector<int> layerLengths;

	vector<vector<float>> biases;
	vector<vector<vector<float>>> weights;

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
				row.push_back(1);
			biases.push_back(row);
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
};

int main()
{
	vector<int> len = {4, 3, 5, 1};
	ANN network = ANN(len);
	network.printBiases();
	network.printWeights();

	vector<vector<float>> bB = { {0, 0, 0}, {1, 2} };
	vector<vector<vector<float>>> wW = { {{1, 2},{3, 4},{5, 6}} };

	ANN network2 = ANN(wW, bB);
	network2.printBiases();
	network2.printWeights();

	system("PAUSE");
}