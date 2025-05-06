package main

import (
	"fmt"
	"math"
	"math/rand"
)

// training loop:
//
// POLICY NETWORK
// 1. feed current position (x) into policy network
// 2. 3 hidden layer neurons:
//      ->tanh((x * weight) + bias)
//      ->tanh((x * weight) + bias)
//      ->tanh((x * weight) + bias)
// 3. pass 3 values into 3 output neurons to get actions
//      ->((x1 * weight) + (x2 * weight) + (x3 * weight)) + bias
// 4. get action probability distribution vector from softmax function
//      ->softmax(output1, output2, output3) will create a vector
//      ->example result: [0.1, 0.3, 0.6]
// 5. Choose an action (randomly for exploration)
// 6. Apply the action to the environment
// 7. Compute the reward
//      ->+1 if on target
//      ->+0 if not
//      LATER: ->-1 for each step
// 8. Log the transition data
//
// VALUE NETWORK
// 1. pass x into values network
// 2. estimate the "value" of the given position based on reward
// 3. Log the transition data for advantage calculation
//
// repeat until target is reached or max steps is reached
//
// Take the episode data and:
// 1. Compute the return
// 2. Compute the advantage
// 3. Store it batch training

const (
	maxSteps    = 20
	maxPosition = 100
	minPosition = 0
	target      = 50
)

type TransitionData struct {
	PlayerPosition    int
	Action            string
	ActionProbability float64
	Reward            int
	ValuePredicted    float64
	TargetPosition    int
	Done              bool
}

type Environment struct {
	PlayerPosition int
	TargetPosition int
	MaxSteps       int
}

type Policy struct {
	HiddenNeurons []Neuron
	OutputNeurons []Neuron
}

type Values struct {
	Weights []float64
}

type Neuron struct {
	Weights []float64
	Bias    float64
}

func main() {
	//training cycle:
	//->run an episode that should accept advantage data to alter it's weights.
	//->gather log file data and calculate advantage
	//->pass into next episode
	//->repeat

	//reinforcement should be at the start
	env := Environment{}
	env.Reset()

	p := Policy{}
	p.InitializeHiddenNeurons(15)
	p.InitializeOutputNeurons(3, len(p.HiddenNeurons))

	outputs := p.OutputLayer(p.HiddenLayer(env.PlayerPosition))
	probabilityDistribution := Softmax(outputs)

	action := StochasticSample(probabilityDistribution, rand.Intn(100))
	if action == 2 {
		fmt.Println("error getting action...")
	}
	fmt.Println(action)
	//apply action from probabilityDistribution

	//compute reward. should be -1 by default and then check if we are on the target and reward 50 points if we are.

}

func StochasticSample(vector []float64, randomNumber int) (action int) {
	var samples []int
	for i, value := range vector {
		integer := int(value * 100)
		for range integer {
			samples = append(samples, i)
		}
	}
	switch samples[randomNumber] {
	case 1:
		return -1
	case 2:
		return 1
	case 0:
		return 0
	}
	return 2
}

func Softmax(vector []float64) (probabilityDistribution []float64) {
	expoSum := 0.0
	for _, value := range vector {
		expoSum += math.Exp(value)
	}
	for _, value := range vector {
		probabilityDistribution = append(probabilityDistribution, (math.Exp(value) / expoSum))
	}
	return probabilityDistribution
}

func (p *Policy) InitializeHiddenNeurons(count int) {
	for range count {
		neuron := Neuron{}
		neuron.Bias = rand.Float64()
		randomnumber := rand.Float64()
		neuron.Weights = append(neuron.Weights, randomnumber)
		p.HiddenNeurons = append(p.HiddenNeurons, neuron)
	}
}

func (p *Policy) InitializeOutputNeurons(ncount, wcount int) {
	for range ncount {
		neuron := Neuron{}
		neuron.Bias = rand.Float64()
		fmt.Println(neuron.Bias)
		for range wcount {
			randomnumber := rand.Float64()
			neuron.Weights = append(neuron.Weights, randomnumber)
		}
		p.OutputNeurons = append(p.OutputNeurons, neuron)
	}
}

func (env *Environment) Move(position int) {

}

func (env *Environment) Reset() {
	env.MaxSteps = maxSteps
	env.PlayerPosition = rand.Intn(100)
	env.TargetPosition = target
}

func (p *Policy) HiddenLayer(input int) (vector []float64) {
	fmt.Println("input : ", input)
	for _, neuron := range p.HiddenNeurons {
		fmt.Println("float64(input): ", float64(input))
		fmt.Println("neuron.Weights[0]: ", neuron.Weights[0])
		logit := (float64(input) * neuron.Weights[0]) + neuron.Bias
		fmt.Println("logit: ", logit)
		fmt.Println("TanH(logit): ", TanH(logit))
		vector = append(vector, TanH(logit))
	}
	return vector
}

func (p *Policy) OutputLayer(hiddenVector []float64) (outputs []float64) {
	for _, neuron := range p.OutputNeurons {
		var summedWeights float64
		for i, value := range hiddenVector {
			summedWeights += (value * neuron.Weights[i])
		}
		outputs = append(outputs, TanH(summedWeights+neuron.Bias))
	}
	return outputs
}

func TanH(logit float64) float64 {
	fmt.Println("-------------------------------------------")
	fmt.Println(logit)
	fmt.Println(math.Sinh(logit))
	fmt.Println(math.Cosh(logit))
	fmt.Println(math.Sinh(logit) / math.Cosh(logit))
	return math.Sinh(logit) / math.Cosh(logit)
}
