package main

import (
	"encoding/json"
	"fmt"
	"github.com/mikevidotto/ff/ff"
	"log"
	"math"
	"math/rand"
	"os"
	"reflect"
)

// training loop:
// VALUES NETWORK
// 1. estimate value of current state (x)
//
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
// 1. pass updated state values network
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
	maxSteps    = 50
	maxPosition = 100
	minPosition = 0
	target      = 50
)

type TransitionData struct {
	Steps []StepData
}

type StepData struct {
	StateValue float64 //result of V(x)

	StatePosition     int     //current position: x
	Action            int     // left: -1, stay: 0 or right: 1
	ActionProbability float64 //result of P(x)
	Reward            int     //-1 for each step, +50 for reaching target
	Done              bool    //true if x = target

	StepIndex        int //increments each step
	DistanceToTarget int //target - current position
}

type Environment struct {
	PlayerPosition int
	TargetPosition int
}

type PPO struct {
	Policy PolicyNetwork
	Values ValuesNetwork
}

type NeuralNetwork interface {
	LoadNeurons() error
	SaveNeurons() error
}

type PolicyNetwork struct {
	HiddenNeurons []Neuron
	OutputNeurons []Neuron
}

type ValuesNetwork struct {
	HiddenNeurons []Neuron
	OutputNeuron  Neuron
}

type Neuron struct {
	Weights []float64
	Bias    float64
}

func main() {
	//training cycle:
	//1. initialize episode
	//  ->load the network
	//  ->use advantage to alter weights.
	//  ->reset environment
	//2. get action from policy network
	//3. apply action to environment
	//4. calculate reward
	//5. run values network
	//6. gather log file data and calculate advantage
	//7. save updated network
	//8. repeat

	//reinforcement should be at the start
	//  ->load environment
	//  ->load the network

	Episode := TransitionData{}
	step := StepData{}

	env := Environment{}
	env.Reset()

	//START EPISODE with 20 STEPS MAX
	for range maxSteps {
		if !step.Done {

		//env, err := env.Load()
		//if err != nil {
		//	log.Fatal(err)
		//}
		step.Initialize()
		step.StatePosition = env.PlayerPosition
		step.DistanceToTarget = env.TargetPosition - step.StatePosition

		ppo := PPO{}
		ppo, err := ppo.Load()
		//run values network for current state
		estimatedvalue := ppo.Values.OutputLayer(ppo.Values.HiddenLayer(env.PlayerPosition))
		step.StateValue = estimatedvalue

		//run policy network

		outputs := ppo.Policy.OutputLayer(ppo.Policy.HiddenLayer(env.PlayerPosition))

		//outputs := p.OutputLayer(p.HiddenLayer(env.PlayerPosition))
		probabilityDistribution := Softmax(outputs)

		action, probability := StochasticSample(probabilityDistribution)
		if action == 2 {
			fmt.Println("error getting action...")
		}

		step.Action = action
		step.ActionProbability = probability

		if env.PlayerPosition == target {
			step.Done = true
		}

		//apply action from probabilityDistribution
		if env.PlayerPosition+action > 100 {
			env.PlayerPosition = 100
		} else if env.PlayerPosition+action < 0 {
			env.PlayerPosition = 0
		} else {
			env.PlayerPosition += action
		}

		//compute reward. should be -1 by default and then check if we are on the target and reward 50 points if we are.
		step.Reward += -1
		if env.PlayerPosition == target {
			step.Reward += 50
		}

		//compute advantage and save data or something

		storedPPO, err := ppo.Load()
		if err != nil {
			log.Fatal(err)
		}
		if !reflect.DeepEqual(ppo, storedPPO) {
			err = ppo.Save()
			if err != nil {
				log.Fatal(err)
			}
		}

		//storedEnv, err := env.Load()
		//if err != nil {
		//	log.Fatal(err)
		//}
		//if !reflect.DeepEqual(env, storedEnv) {
		//	fmt.Println("Saving because changes have been made to the Environment.")
		//	err = env.Save()
		//	if err != nil {
		//		log.Fatal(err)
		//	}
		//} else {
		//	fmt.Println("No changes made to the Environment, therefore we won't update init file.")
		//}

		Episode.Steps = append(Episode.Steps, step)
		step.StepIndex++
        }
	}

    //log/store the episode data to calculate the advantage and return


	for _, step := range Episode.Steps {
		fmt.Printf("step: %2d, x: %2d, %4d units to target, V(%d): %4f action: %2d, action prob: %4f, reward: %2d, done:  %4t\n", step.StepIndex, step.StatePosition, step.DistanceToTarget, step.StatePosition, step.StateValue, step.Action, step.ActionProbability, step.Reward, step.Done)
		//fmt.Printf("Step %d \nx position: %d: %d units to target(50) \nstate value: %f \nAction: %d \nActionProbability: %f\nReward: %d\nDone: %t\n-----------------------------------------------\n", step.StepIndex, step.StatePosition, step.DistanceToTarget, step.StateValue, step.Action, step.ActionProbability, step.Reward, step.Done)
	}
}

func (data *StepData) Initialize() {

	//data.StatePosition = 0
	//data.DistanceToTarget = 0
	//data.StateValue = 0.0

	//data.Action = 0
	//data.ActionProbability = 0.0

	//data.Reward = 0

	//data.StepIndex = 0
	//data.Done = false
}

//func (env *Environment) Load() (Environment, error) {
//	var savedEnvironment Environment
//
//	if !ff.FileExists("./init/env/latest.txt") {
//		//reset to default values
//		savedEnvironment.Reset()
//		return savedEnvironment, nil
//	}
//	file, err := os.Open("./init/env/latest.txt")
//	if err != nil {
//		return Environment{}, err
//	}
//	defer file.Close()
//
//	decoder := json.NewDecoder(file)
//	err = decoder.Decode(&savedEnvironment)
//	if err != nil {
//		return Environment{}, err
//	}
//
//	return savedEnvironment, nil
//}
//
//func (env *Environment) Save() error {
//	bytes, err := json.Marshal(env)
//	if err != nil {
//		return err
//	}
//	//create backup of current network data
//	if ff.FileExists("./init/env/latest.txt") {
//		data, err := os.ReadFile("./init/env/latest.txt")
//		if err != nil {
//			return err
//		}
//		file, err := os.CreateTemp("./history/env/", "*.txt")
//		if err != nil {
//			return err
//		}
//		_, err = file.Write(data)
//		if err != nil {
//			return err
//		}
//	}
//	err = os.WriteFile("./init/env/latest.txt", bytes, 0666)
//	if err != nil {
//		return err
//	}
//
//	return nil
//}

func (p *PPO) Save() error {
	bytes, err := json.Marshal(p)
	if err != nil {
		return err
	}
	//create backup of current network data
	if ff.FileExists("./init/ppo/latest.txt") {
		data, err := os.ReadFile("./init/ppo/latest.txt")
		if err != nil {
			return err
		}
		file, err := os.CreateTemp("./history/ppo/", "*.txt")
		if err != nil {
			return err
		}
		_, err = file.Write(data)
		if err != nil {
			return err
		}
	}
	err = os.WriteFile("./init/ppo/latest.txt", bytes, 0666)
	if err != nil {
		return err
	}

	return nil
}

func (p *PPO) Load() (PPO, error) {
	var savedPPO PPO
	if !ff.FileExists("./init/ppo/latest.txt") {
		savedPPO.Policy.InitializeHiddenNeurons(3)
		savedPPO.Policy.InitializeOutputNeurons(3, 3)
		savedPPO.Values.InitializeHiddenNeurons(3)
		savedPPO.Values.InitializeOutputNeurons(3)
		return savedPPO, nil
	}
	file, err := os.Open("./init/ppo/latest.txt")
	if err != nil {
		return PPO{}, err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	err = decoder.Decode(&savedPPO)
	if err != nil {
		return PPO{}, err
	}

	return savedPPO, nil
}

func StochasticSample(vector []float64) (action int, probability float64) {
	testsum := 0
	var samples []int
	for i, value := range vector {
		integer := int(value * 100)
		for range integer {
			samples = append(samples, i)
		}
		testsum += integer
	}
	randomNumber := rand.Intn(len(samples) - 1)
	switch samples[randomNumber] {
	case 0:
		return 0, vector[0]
	case 1:
		return -1, vector[1]
	case 2:
		return 1, vector[2]
	}
	return 2, 2
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

func (p *PolicyNetwork) InitializeHiddenNeurons(count int) {
	for range count {
		neuron := Neuron{}
		neuron.Bias = rand.Float64()
		randomnumber := rand.Float64()
		neuron.Weights = append(neuron.Weights, randomnumber)
		p.HiddenNeurons = append(p.HiddenNeurons, neuron)
	}
}

func (p *PolicyNetwork) InitializeOutputNeurons(ncount, wcount int) {
	for range ncount {
		neuron := Neuron{}
		neuron.Bias = rand.Float64()
		for range wcount {
			randomnumber := rand.Float64()
			neuron.Weights = append(neuron.Weights, randomnumber)
		}
		p.OutputNeurons = append(p.OutputNeurons, neuron)
	}
}

func (p *ValuesNetwork) InitializeHiddenNeurons(count int) {
	for range count {
		neuron := Neuron{}
		neuron.Bias = rand.Float64()
		randomnumber := rand.Float64()
		neuron.Weights = append(neuron.Weights, randomnumber)
		p.HiddenNeurons = append(p.HiddenNeurons, neuron)
	}
}

func (v *ValuesNetwork) InitializeOutputNeurons(wcount int) {
	v.OutputNeuron.Bias = rand.Float64()
	for range wcount {
		randomnumber := rand.Float64()
		v.OutputNeuron.Weights = append(v.OutputNeuron.Weights, randomnumber)
	}
}

func (env *Environment) Reset() {
	env.PlayerPosition = rand.Intn(100)
	env.TargetPosition = target
}

func (v *ValuesNetwork) HiddenLayer(input int) (vector []float64) {
	for _, neuron := range v.HiddenNeurons {
		logit := (float64(input) * neuron.Weights[0]) + neuron.Bias
		vector = append(vector, logit)
	}

	return vector
}

func (v *ValuesNetwork) OutputLayer(hiddenVector []float64) float64 {
	var summedWeights float64
	for i, value := range hiddenVector {
		summedWeights += (value * v.OutputNeuron.Weights[i])
	}
	return summedWeights + v.OutputNeuron.Bias
}

func (p *PolicyNetwork) HiddenLayer(input int) (vector []float64) {
	for _, neuron := range p.HiddenNeurons {
		logit := (float64(input) * neuron.Weights[0]) + neuron.Bias
		vector = append(vector, TanH(logit))
	}
	return vector
}

func (p *PolicyNetwork) OutputLayer(hiddenVector []float64) (outputs []float64) {
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
	return math.Sinh(logit) / math.Cosh(logit)
}
