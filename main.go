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

	State             int     //current position: x
	Action            int     // left: -1, stay: 0 or right: 1
	ActionProbability float64 //result of P(x)
	LogProbability    float64
	NextState         int  // current position + action: x + action (-1, 0, 1)
	Reward            int  //-1 for each step, +50 for reaching target
	Done              bool //true if x = target

	StepIndex        int //increments each step
	DistanceToTarget int //target - current position

	Return    float64
	Advantage float64
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
	CreateDirectories()
	//training cycle:
	//1. initialize episode
	//  ->load the network's neurons
	//  ->use advantage to alter weights.
	//  ->reset environment
	//2. get action from policy network
	//3. apply action to environment
	//4. calculate reward
	//5. run values network
	//6. calculate return and advantage
	//  ->return: At = reward + discountvalue(0.99) * ((reward2 + discountvalue(0.99)) * reward3 + discountvalue(0.99)...
	//  ->advantage = return(At) - estimated value of state at t( V(t) )
	//6. get log value of... something. I think the policy network results?
	//7. save updated network
	//8. repeat

	//initialize policy
	//run episode, collect transition data for ppo
	//run training
	//update policy

	//NEXT STEPS!!!!!!!!!!!!!!!!!!!
	// think about the workflow of the application.
	// we want to be able to generate an episode, and then train off that episode's transition data.
	// when we run the application, it should have options:
	// 1. run episode
	// 2. run training
	//
	//
	// training.
	//run each step of your stored transition data through the policy network
	//step 1: x=5 -> x=6, action=1(right)
	//->PolicyNetwork(5) results:
	//  left   stay  right
	//->(0.32, 0.32, 0.36)
	//get the logprobability of the same action for that step
	//log(0.36)
	//get the ratio of both log probabilities
	//->not sure how yet... (division?) if x=4, y=8 then x/y = 4/8 = 1/2, therefore 1:2 ratio
	//multiply ratio by the advantage
	//clip the result
	//take negative clipped value to get loss

	//then take the loss for each step, sum them up and get the average.
	//feed the average into an optimizer that adjusts the weights of the policy network.
	//fun stuff!

	step := StepData{}

	env := Environment{}
	env.Reset()

	ppo := PPO{}
	ppo, err := ppo.Load()
	if err != nil {
		log.Fatal(err)
	}

	EpisodeData := TransitionData{}

	//START EPISODE
	EpisodeData = EpisodeData.RunEpisode(ppo, 50, env, step)

	//function: log/store the episode data using TransitionData{}
	EpisodeData.SaveData()
	//we have the ppo data inside the latest.txt file.
	//->function: populate a temp ppo and return it with the adjusted values
	//->assign current ppo to this function before saving so that the old ppo is backed up before it is adjusted
	var losses []float64
	for _, step := range EpisodeData.Steps {
		outputs := ppo.Policy.OutputLayer(ppo.Policy.HiddenLayer(step.State))
		probabilityDistribution := Softmax(outputs)
		probability := GetNewProbability(probabilityDistribution, step.Action)

		newlogprob := math.Log(probability)
		ratio := math.Exp(step.LogProbability / newlogprob)
		clippedratio := ClipValue(ratio)
		unclippedobjective := ratio * step.Advantage
		clippedobjective := clippedratio * step.Advantage
		surrogate_loss := MinimumValue(clippedobjective, unclippedobjective)
		losses = append(losses, (surrogate_loss * -1))
	}

	var sumloss float64
	for _, loss := range losses {
		sumloss += loss
	}
	lossaverage := sumloss / float64(len(EpisodeData.Steps))

	fmt.Println("loss average: ", lossaverage)

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

	for _, step := range EpisodeData.Steps {
		fmt.Printf("step: %2d, x: %3d -> %3d, %4d units to target, V(%d): %4f action: %2d, action prob: %4f, reward: %2d, done:  %4t\n", step.StepIndex, step.State, step.NextState, step.DistanceToTarget, step.State, step.StateValue, step.Action, step.ActionProbability, step.Reward, step.Done)

		//fmt.Printf("Step %d \nx position: %d: %d units to target(50) \nstate value: %f \nAction: %d \nActionProbability: %f\nReward: %d\nDone: %t\n-----------------------------------------------\n", step.StepIndex, step.State, step.DistanceToTarget, step.StateValue, step.Action, step.ActionProbability, step.Reward, step.Done)
	}
}

func MinimumValue(value1, value2 float64) float64 {
	var minimum float64
	if value1 < value2 {
		minimum = value1
	} else {
		minimum = value2
	}
	return minimum
}

func ClipValue(value float64) float64 {
	var clipped float64
	epsilon := 0.2
	min := 1 - epsilon
	max := 1 + epsilon

	if value > max {
		clipped = max
	} else if value < min {
		clipped = min
	} else {
		clipped = value
	}
	return clipped
}

func GetNewProbability(probabilityDistribution []float64, savedAction int) (probability float64) {

	switch savedAction {
	case -1:
		probability = probabilityDistribution[0]
	case 0:
		probability = probabilityDistribution[1]
	case 2:
		probability = probabilityDistribution[2]
	}
	return probability
}

func (data *TransitionData) RunEpisode(ppo PPO, numSteps int, env Environment, step StepData) TransitionData {
	for range maxSteps {
		if !step.Done {

			step.Reward = -1 //automatically -1 reward for each step taken

			step.State = env.PlayerPosition
			step.DistanceToTarget = target - env.PlayerPosition

			//run values network for current state
			step.StateValue = ppo.Values.OutputLayer(ppo.Values.HiddenLayer(env.PlayerPosition))

			//run policy network

			outputs := ppo.Policy.OutputLayer(ppo.Policy.HiddenLayer(env.PlayerPosition))

			//outputs := p.OutputLayer(p.HiddenLayer(env.PlayerPosition))
			probabilityDistribution := Softmax(outputs)

			action, probability := StochasticSample(probabilityDistribution)
			if action == 2 {
				log.Println("error getting action...")
			}

			step.Action = action
			step.ActionProbability = probability
			step.LogProbability = math.Log(probability)

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

			step.NextState = env.PlayerPosition
			if step.NextState == target {
				step.Reward += 50
				step.Done = true
			} else {

				//compute reward. should be -1 if we are further from the target and 0 if we are closer to then check if we are on the target and reward 50 points if we are.
				margin := (env.TargetPosition - env.PlayerPosition)
				if margin < 0 {
					if step.DistanceToTarget < 0 {
						if (margin * -1) == (step.DistanceToTarget * -1) {
							step.Reward += 0
						} else if (margin * -1) > (step.DistanceToTarget * -1) {
							step.Reward += -1
						} else {
							step.Reward += 2
						}

					} else {
						if (margin * -1) == step.DistanceToTarget {
							step.Reward += 0
						} else if (margin * -1) > step.DistanceToTarget {
							step.Reward += -1
						} else {
							step.Reward += 2
						}
					}
				} else {
					if step.DistanceToTarget < 0 {
						if margin == (step.DistanceToTarget * -1) {
							step.Reward += 0
						} else if margin > (step.DistanceToTarget * -1) {
							step.Reward += -1
						} else {
							step.Reward += 2
						}

					} else {
						if margin == step.DistanceToTarget {
							step.Reward += 0
						} else if margin > step.DistanceToTarget {
							step.Reward += -1
						} else {
							step.Reward += 2
						}

					}
				}
			}

			data.Steps = append(data.Steps, step)
			step.StepIndex++
		}
	}
	//function:calculate the advantage and return
	returnvalues := Returns(data.Steps, ppo)
	advantagevalues := Advantage(data.Steps, ppo, returnvalues)

	for i, step := range data.Steps {
		step.Return = returnvalues[i]
		step.Advantage = advantagevalues[i]
	}

	*data = data.AddReturnsAdvantages(returnvalues, advantagevalues)

	return *data

}

func (td *TransitionData) AddReturnsAdvantages(returns, advantages []float64) (updatedData TransitionData) {
	for i, step := range td.Steps {
		tempstep := step
		newstep := StepData{
			StateValue:        tempstep.StateValue,        //result of V(x)
			State:             tempstep.State,             //current position: x
			Action:            tempstep.Action,            // left: -1, stay: 0 or right: 1
			ActionProbability: tempstep.ActionProbability, //result of P(x)
			LogProbability:    tempstep.LogProbability,
			NextState:         tempstep.NextState,        // current position + action: x + action (-1, 0, 1)
			Reward:            tempstep.Reward,           //-1 for each step, +50 for reaching target
			Done:              tempstep.Done,             //true if x = target
			StepIndex:         tempstep.StepIndex,        //increments each step
			DistanceToTarget:  tempstep.DistanceToTarget, //target - current position
			Return:            returns[i],
			Advantage:         advantages[i],
		}

		updatedData.Steps = append(updatedData.Steps, newstep)
	}

	return updatedData
}

func CreateDirectories() {
	transition := "./transitions"
	transitionHistory := "./history/transitions"
	init := "./init"
	ppo := "./init/ppo"
	CreateDirIfNotExist(transition)
	CreateDirIfNotExist(transitionHistory)
	CreateDirIfNotExist(init)
	CreateDirIfNotExist(ppo)
}

func CreateDirIfNotExist(dir string) {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err = os.MkdirAll(dir, 0666)
		if err != nil {
			log.Fatal(err)
		}
	}

}

func Returns(steps []StepData, ppo PPO) (finalreturns []float64) {
	var returns []float64
	gamma := 0.99
	returnvalue := 0.0
	returnsum := 0.0
	for t := len(steps) - 1; t >= 0; t-- {
		if t == len(steps)-1 {
			returnvalue = float64(steps[t].Reward)
			returnsum = returnvalue
			returns = append(returns, returnsum)
		} else {
			returnvalue = float64(steps[t].Reward) + (gamma * returnsum)
			returnsum = returnvalue
			returns = append(returns, returnsum)
		}
	}

	for j := len(returns) - 1; j >= 0; j-- {
		finalreturns = append(finalreturns, returns[j])
	}

	return finalreturns
}

func Advantage(steps []StepData, ppo PPO, returns []float64) (advantages []float64) {
	var advantage float64
	for i, step := range steps {
		advantage = returns[i] - step.StateValue
		advantages = append(advantages, advantage)
	}

	return advantages
}

func (td *TransitionData) SaveData() error {
	bytes, err := json.Marshal(td)
	if err != nil {
		return err
	}
	if ff.FileExists("./transitions/latest_transition.txt") {
		data, err := os.ReadFile("./transitions/latest_transition.txt")
		if err != nil {
			return err
		}
		file, err := os.CreateTemp("./history/transitions/", "*.txt")
		if err != nil {
			return err
		}
		_, err = file.Write(data)
		if err != nil {
			return err
		}
	}
	err = os.WriteFile("./transitions/latest_transition.txt", bytes, 0666)
	if err != nil {
		return err
	}

	return nil
}

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
