#pragma once
#include<iostream>
#include<vector>
#include<cmath>
using namespace std;

//------------------------------NEURON CLASSES------------------------------

//abstract neuron
class Neuron
{
public:

	virtual float actFunc(float z) = 0;
	virtual float actFuncPrime(float z) = 0;

};

//sigmoid neuron
class Sigmoid : public Neuron
{
public:

	float actFunc(float z)
	{
		return 1 / (1 + exp(-z));
	}

	float actFuncPrime(float z)
	{
		return actFunc(z) * (1 - actFunc(z));
	}

};

class Max : public Neuron
{
public:

	float actFunc(float z)
	{
		return (z < 0 ? 0 : z);
	}

	float actFuncPrime(float z)
	{
		return (z < 0 ? 0.0f : 1.0f);
	}
};

//------------------------------COST CLASSES------------------------------

//abstract cost
class Cost
{

public:

	virtual float cost(vector<vector<float>>& ex, vector<vector<float>>& acts) = 0;
	virtual float gradient(float aaa, float yyy)
	{
		return aaa - yyy;
	}

};

//mean square error cost function
class SquareError : public Cost
{

public:

	float cost(vector<vector<float>>& ex, vector<vector<float>>& acts)
	{
		float total = 0;

		for (unsigned i = 0; i < ex[1].size(); i++)
			total += pow(ex[1][i] - acts[acts.size() - 1][i], 2);

		return total / 2;
	}
};

//cross-entropy cost function
class CrossEntropy : public Cost
{

public:

	float cost(vector<vector<float>>& ex, vector<vector<float>>& acts)
	{
		float total = 0;

		for (unsigned i = 0; i < ex[1].size(); i++)
			total += ex[1][i] * log(acts[acts.size() - 1][i]) + (1 - ex[1][i]) * log(1 - acts[acts.size() - 1][i]);

		return -total;
	}

	float gradient(float aaa, float yyy)
	{
		return ((-yyy / aaa) + ((1 - yyy)/(1 - aaa)));
	}
};