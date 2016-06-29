#include<iostream>
#include<vector>
#include<cmath>
#include<random>
#include<ctime>
#include<fstream>
#include<string>
using namespace std;

vector<vector<vector<float>>> getTrainingData(string name);

//------------------------------COST CLASSES------------------------------

//abstract cost class
class Cost
{

public:

	virtual float cost(vector<vector<float>>& ex, vector<vector<float>>& acts) = 0;
	virtual float gradient(float aaa, float yyy) = 0;

};

//mean square error cost function
class SquareError : public Cost
{

public:

	float cost(vector<vector<float>>& ex, vector<vector<float>>& acts)
	{
		float total = 0;

		for (int i = 0; i < ex[1].size(); i++)
			total += pow(ex[1][i] - acts[acts.size() - 1][i], 2);

		return total / 2;
	}

	float gradient(float aaa, float yyy)
	{
		return aaa - yyy;
	}

};

//cross-entropy cost function
class CrossEntropy : public Cost
{

public:

	float cost(vector<vector<float>>& ex, vector<vector<float>>& acts)
	{
		float total = 0;

		for (int i = 0; i < ex[1].size(); i++)
			total += ex[1][i] * log(acts[acts.size() - 1][i]) + (1 - ex[1][i]) * log(1 - acts[acts.size() - 1][i]);

		return - (total);
	}

	float gradient(float aaa, float yyy)
	{

	}

};

//------------------------------THE ANN CLASS------------------------------

class ANN
{

private:

	//------------------------------ANN CONTENTS------------------------------
	int numLayers;
	int maxLength;
	vector<int> layerLengths;

	vector<vector<float>> biases;
	vector<vector<vector<float>>> weights;
	vector<vector<float>> activations;
	vector<vector<float>> errors;

	Cost* theCost;

public:

	//------------------------------CONSTRUCTORS------------------------------

	//set ANN based on layer lengths
	ANN(vector<int>& lengths)
	{
		theCost = new SquareError();

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
				row.push_back( distB(generatorB) );
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

	//set ANN based on weight and bias matrix
	ANN(vector<vector<vector<float>>>& w, vector<vector<float>>& b)
	{
		theCost = new SquareError();

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

	//set ANN based on file
	ANN(string file)
	{
		theCost = new SquareError();

		ifstream in(file);
		if (!in) {
			cout << "Error!!" << endl;
			return;
		}

		in >> numLayers;
		layerLengths.resize(numLayers);

		maxLength = 0;

		for (int i = 0; i < numLayers; i++) {
			in >> layerLengths[i];
			maxLength = (maxLength < layerLengths[i] ? layerLengths[i]: maxLength);
		}

		//init biases
		for (int i = 0; i < numLayers; i++) {
			vector<float> row;
			for (int j = 0; j < layerLengths[i]; j++)
				row.push_back(0);
			biases.push_back(row);
		}

		//init weights
		for (int i = 0; i < numLayers - 1; i++) {
			vector<vector<float>> row;

			for (int j = 0; j < layerLengths[i]; j++) {
				vector<float> subRow;
				for (int k = 0; i + 1 < numLayers && k < layerLengths[i + 1]; k++)
					subRow.push_back(0);
				row.push_back(subRow);
			}
			weights.push_back(row);
		}

		string temp;
		in >> temp;

		//get biases
		for (int i = 0; i < maxLength; i++) {
			for (int j = 0; j < numLayers; j++) {
				if (i < layerLengths[j]) {
					if (j == 0) {
						char c;
						in >> c;
						biases[j][i] = 0;
					}
					else
						in >> biases[j][i];
				}
			}
		}

		in >> temp;

		//get weights
		for (int i = 0; i < weights.size(); i++) {
			for (int j = 0; j < weights[i].size(); j++) {
				for (int k = 0; k < weights[i][j].size(); k++)
					in >> weights[i][j][k];
			}
		}

		in.close();

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

	//------------------------------PRINT FUNCTIONS------------------------------

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
	void printBiases(ostream& st)
	{
		st << "Biases:" << endl;

		for (unsigned j = 0; j < maxLength; j++){
			for (unsigned i = 0; i < biases.size(); i++) {
					if (i == 0 && j < biases[i].size())
						st << "X ";
					else if (j >= biases[i].size())
						st << "  ";
					else
						st << biases[i][j] << " ";
			}
				st << endl;
		}
		st << endl;
	}

	//print all weights; w[n][][] correspond to b[n+1][]
	void printWeights(ostream& st)
	{
		st << "Weights:" << endl;

		for (int i = 1; i <= weights.size(); i++) {
			for (int j = 0; j < layerLengths[i - 1]; j++) {
				for (int k = 0; k < layerLengths[i]; k++)
					st << weights[i - 1][j][k] << " ";
				st << endl;
			}
			st << endl;
		}		
			//printWeightsForLayer(i, st);
	}

	//------------------------------BACKPROPOGATION------------------------------

	//calculate activations for all layers
	void feedForward(vector<float>& x)
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

	//find error vector for final layer
	void outputError(vector<float>& y)
	{
		int last = biases.size() - 1;

		for (int i = 0; i < biases[last].size(); i++)
			errors[last][i] = theCost->gradient(activations[last][i],y[i]) * (activations[last][i] * (1 - activations[last][i]));
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

	//------------------------------UPDATING FUNCTIONS------------------------------

	//calls all four steps at once for one training example
	void updateOne(vector<float>& in, vector<float>& out, float rate)
	{	
		feedForward(in);
		outputError(out);
		backpropogate();
		descent(rate);
	}

	//calls all four steps at once for a training example set
	void update(vector<vector<float>>& in, vector<vector<float>>& out, float rate, int n)
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

	//calls all four steps at once for set of training pairs (x, y) and perc% of data is used to train
	void updatePair(vector<vector<vector<float>>>& pair, float rate, int n, float perc)
	{
		//check to see if pairs are given and then if they match dimensions of input and output layers
		for (int i = 0; i < pair.size(); i++) {
			if (pair[i].size() != 2)
				return;

			if (pair[i][0].size() != biases[0].size() || pair[i][1].size() != biases[biases.size() - 1].size())
				return;
		}

		for (int a = 0; a < n; a++) {
			for (int i = 0; i < (pair.size() * perc)/100; i++)
				updateOne(pair[i][0], pair[i][1], rate);
			
			//if (a % (n / 10) == n/10 - 1)
				cout << "Epoch " << a + 1<< " Cost: " << costPairs(pair) << endl;
		}

		cout << endl;
	}

	//------------------------------COST FUNCTION CALL------------------------------

	//mean squared error
	float costPairs(vector<vector<vector<float>>>& stuff)
	{
		float sum = 0;

		for (int i = 0; i < stuff.size(); i++) {
			feedForward(stuff[i][0]);

			sum += theCost->cost(stuff[i], activations);
		}

		return sum/(stuff.size());
	}

	//------------------------------FILE I/O------------------------------

	//output ANN to file
	void output(string fileName)
	{
		ofstream file;
		file.open(fileName);

		if (!file) {
			cout << "Error 404: No file" << endl;
			return;
		}

		file << biases.size()<<endl;

		for (int i = 0; i < biases.size(); i++)
			file << biases[i].size() << " ";

		file << endl;

		printBiases(file);
		printWeights(file);

		file.close();
	}

	//------------------------------OUTPUT CLASSIFICATION------------------------------

	//gives ratio of correct classification of last perc% of data
	void classify(vector<vector<vector<float>>>& data, int perc)
	{
		int size = data.size(), correct = 0, index;

		for (int i = size - size*(perc/100.0); i < size; i++)
		{
			feedForward(data[i][0]);

			if (activations.back()[0] > activations.back()[1])
				index = 0;
			else
				index = 1;

			if (data[i][1][index] >= .99)
				correct++;	
		}

		cout << "Ratio correct: " << correct << "/" << size * (perc/100.0) << endl;
	}

};

//------------------------------MAIN AND OTHER FUNCTIONS------------------------------

int main()
{
	vector<vector<vector<float>>> train;

	train = getTrainingData("train3.dat");

	vector<int> len = {5, 4, 2};

	ANN network = ANN(len);
	network.updatePair(train, .1, 150, 90);
	network.output("ANN.txt");

	network.classify(train, 10);

	system("PAUSE");
}

vector<vector<vector<float>>> getTrainingData(string name)
{
	vector<vector<vector<float>>> data;

	ifstream file;
	file.open(name);

	int numIn, numOut, lines;
	file >> numIn >> numOut;
	string dummy;

	for (int i = 0; i < numIn + numOut; i++)
		file >> dummy;

	file >> lines;

	for (int i = 0; i < lines; i ++) {
		vector<vector<float>> pair;
		vector<float> in, out;

		in.resize(numIn);
		out.resize(numOut);

		for (int j = 0; j < numIn; j++)
			file >> in[j];

		for (int k = 0; k < numOut; k++)
			file >> out[k];

		pair = {in, out};
		data.push_back(pair);
	}

	return data;
}