#include "ANN.h"
#include "Cost.h"


int maintrain(int argc, char *argv[])
{
	vector<vector<vector<float>>> train;

	train = getTrainingData("train4.dat");

	vector<int> len = {5, 4, 2};

	ANN network = ANN(len);
	network.updatePair(train, .1f, 150, 90);
	network.output("ANN.txt");

	network.classify(train, 10);

	system("PAUSE");

	return 0;
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
