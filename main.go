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
	maxSteps     = 50
	maxPosition  = 100
	minPosition  = 0
	target       = 50
	learningrate = 0.01
)

type TransitionData struct {
	Steps []StepData
}

type StepData struct {
	StateValue float64 //result of V(x)

	State                   int     //current position: x
	Action                  int     // left: -1, stay: 0 or right: 1
	ActionProbability       float64 //result of P(x)
	ProbabilityDistribution []float64
	YVector                 []float64
	LogProbability          float64
	HiddenValues            []float64
	OutputLogits            []float64
	NextState               int  // current position + action: x + action (-1, 0, 1)
	Reward                  int  //-1 for each step, +50 for reaching target
	Done                    bool //true if x = target

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
	//TODO
	//
	//flags:
	//->e: run an episode with the latest policy, backup the old episode data, and store the latest episode data
	//->t: run a training session for the current network, backup the old policy data, and store the latest version of the policy

	CreateDirectories()

	ppo, err := LoadPPO()
	if err != nil {
		log.Fatal(err)
	}

	//START EPISODE
	EpisodeData := RunEpisode(ppo, maxSteps)

	//function: log/store the episode data using TransitionData{}
	EpisodeData.SaveData()
	//we have the ppo data inside the latest.txt file.
	//->function: populate a temp ppo and return it with the adjusted values
	//->assign current ppo to this function before saving to back up old ppo before it is adjusted
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
		surrogate_loss := min(clippedobjective, unclippedobjective)
		losses = append(losses, (surrogate_loss * -1))
	}

	var sumloss float64
	for _, loss := range losses {
		sumloss += loss
	}
	lossaverage := sumloss / float64(len(EpisodeData.Steps))

	fmt.Println("loss average: ", lossaverage)

	ogradients := GetOutputGradients(EpisodeData, ppo)
	hgradients := GetHiddenGradients(EpisodeData, ppo)

	updatedPPO := BackProp(hgradients, ogradients, EpisodeData.Steps, ppo)

	fmt.Println("new ppo: ", updatedPPO)

	storedPPO, err := LoadPPO()
	if err != nil {
		log.Fatal(err)
	}
	if !reflect.DeepEqual(ppo, storedPPO) {
		err = ppo.Save()
		if err != nil {
			log.Fatal(err)
		}
	}

	//for _, step := range EpisodeData.Steps {
	//	fmt.Printf("step: %2d, x: %3d -> %3d, %4d units to target, V(%d): %4f action: %2d, action prob: %4f, reward: %2d, done:  %4t\n", step.StepIndex, step.State, step.NextState, step.DistanceToTarget, step.State, step.StateValue, step.Action, step.ActionProbability, step.Reward, step.Done)

	//	//fmt.Printf("Step %d \nx position: %d: %d units to target(50) \nstate value: %f \nAction: %d \nActionProbability: %f\nReward: %d\nDone: %t\n-----------------------------------------------\n", step.StepIndex, step.State, step.DistanceToTarget, step.StateValue, step.Action, step.ActionProbability, step.Reward, step.Done)
	//}
}

func GetOutputGradients(data TransitionData, ppo PPO) []float64 {
	var gradients [][]float64
	//for each step we want:
	//->the loss w.r.t. each weight in the output layer. There are 3 output neurons with 3 weights each, for a total of 9 weights
	//
	//So the steps would be:
	//1. iterate over the steps
	//2. iterate over each neuron to get the value of each hidden layer output that goes into calculating the loss w.r.t. each weight
	//3. add each value for that step into an array.
	//4. append each step's array to an array.
	//RESULT: An array of length:50 for each step, that contains an array of nine values for each weight.
	//5. take each element of each array and add

	for _, step := range data.Steps {
		var stepvalues []float64
		for j := range ppo.Policy.OutputNeurons {
			for k := range ppo.Policy.HiddenNeurons {
				//fmt.Printf("step.HiddenValues[%d]: %.3f\n", k, step.HiddenValues[k])
				losswrtweight := (step.ProbabilityDistribution[j] - step.YVector[j]) * step.HiddenValues[k]
				stepvalues = append(stepvalues, losswrtweight)
			}
		}
		gradients = append(gradients, stepvalues)
	}

	var sums [9]float64
	//sum up all values
	for _, step := range gradients {
		for j, value := range step {
			sums[j] += value
			//   fmt.Printf("sums: %.3f\n", sums[j])
		}
		//fmt.Printf("%d: %.3f\n", i, step)
	}

	var averages []float64
	for _, sum := range sums {
		averages = append(averages, sum/maxSteps)
		fmt.Printf("averages: %.3f\n", averages)
	}

	return averages
}

func GetHiddenGradients(data TransitionData, ppo PPO) []float64 {

	var gradients [][]float64

	for _, step := range data.Steps {
		var stepvalues []float64
		for i := range ppo.Policy.HiddenNeurons {
			var hiddenweightsum float64
			for j, oneuron := range ppo.Policy.OutputNeurons {
				losswrtweight := (step.ProbabilityDistribution[j] - step.YVector[j]) * oneuron.Weights[i] * (1 - step.HiddenValues[i]) * step.StateValue
				hiddenweightsum += losswrtweight
			}
			stepvalues = append(stepvalues, hiddenweightsum)
		}
		gradients = append(gradients, stepvalues)
	}

	var sums [3]float64
	for _, step := range gradients {
		fmt.Println("len(step):", len(step))
		for j, value := range step {
			sums[j] += value
		}
	}

	var averages []float64
	for _, sum := range sums {
		averages = append(averages, sum/maxSteps)
	}
	return averages
}

func BackProp(haverages, oaverages []float64, steps []StepData, ppo PPO) (updatedPPO PPO) {
	updatedPPO = ppo
	for i, neurons := range ppo.Policy.HiddenNeurons {
		for j, weight := range neurons.Weights {
			newweight := weight - (haverages[j] * learningrate)
			fmt.Println("current hidden weight: ", weight)
			updatedPPO.Policy.HiddenNeurons[i].Weights[j] = newweight
			fmt.Println("new hidden weight: ", updatedPPO.Policy.HiddenNeurons[i].Weights[j])
		}
	}
	for i, neurons := range ppo.Policy.OutputNeurons {
		for j, weight := range neurons.Weights {
			newweight := weight - (oaverages[j] * learningrate)
			fmt.Println("current output weight: ", weight)
			updatedPPO.Policy.OutputNeurons[i].Weights[j] = newweight
			fmt.Println("new output weight: ", updatedPPO.Policy.OutputNeurons[i].Weights[j])
		}
	}
	return updatedPPO
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

func RunEpisode(ppo PPO, numSteps int) (data TransitionData) {
	step := StepData{}
	env := Environment{}
	env.Reset()
	for range maxSteps {
		if !step.Done {

			step.Reward = -1 //automatically -1 reward for each step taken

			step.State = env.PlayerPosition
			step.DistanceToTarget = target - env.PlayerPosition

			//run values network for current state
			step.StateValue = ppo.Values.OutputLayer(ppo.Values.HiddenLayer(env.PlayerPosition))

			//run policy network

			fmt.Printf("env.PlayerPosition: %d\n", env.PlayerPosition)
			step.HiddenValues = ppo.Policy.HiddenLayer(env.PlayerPosition)
			for _, value := range step.HiddenValues {
				fmt.Printf("HiddenValues: %.3f", value)
			}
			step.OutputLogits = ppo.Policy.OutputLayer(step.HiddenValues)

			step.ProbabilityDistribution = Softmax(step.OutputLogits)

			action, probability := StochasticSample(step.ProbabilityDistribution)
			if action == 2 {
				log.Println("error getting action...")
			}

			if step.YVector == nil {
				switch action {
				case -1:
					step.YVector = []float64{1, 0, 0}
				case 0:
					step.YVector = []float64{0, 1, 0}
				case 1:
					step.YVector = []float64{0, 0, 1}
				}
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

	data = data.AddReturnsAdvantages(returnvalues, advantagevalues)

	return data

}

//type StepData struct {
//	StateValue float64 //result of V(x)
//
//	State                   int     //current position: x
//	Action                  int     // left: -1, stay: 0 or right: 1
//	ActionProbability       float64 //result of P(x)
//	ProbabilityDistribution []float64
//	YVector                 []float64
//	LogProbability          float64
//	HiddenValues            []float64
//	OutputLogits            []float64
//	NextState               int  // current position + action: x + action (-1, 0, 1)
//	Reward                  int  //-1 for each step, +50 for reaching target
//	Done                    bool //true if x = target
//
//	StepIndex        int //increments each step
//	DistanceToTarget int //target - current position
//
//	Return    float64
//	Advantage float64
//}

func (td *TransitionData) AddReturnsAdvantages(returns, advantages []float64) (updatedData TransitionData) {
	for i, step := range td.Steps {
		tempstep := step
		newstep := StepData{
			StateValue: tempstep.StateValue, //result of V(x)

			State:                   tempstep.State,             //current position: x
			Action:                  tempstep.Action,            // left: -1, stay: 0 or right: 1
			ActionProbability:       tempstep.ActionProbability, //result of P(x)
			ProbabilityDistribution: tempstep.ProbabilityDistribution,
			YVector:                 tempstep.YVector,
			LogProbability:          tempstep.LogProbability,
			HiddenValues:            tempstep.HiddenValues,
			OutputLogits:            tempstep.OutputLogits,
			NextState:               tempstep.NextState, // current position + action: x + action (-1, 0, 1)
			Reward:                  tempstep.Reward,    //-1 for each step, +50 for reaching target
			Done:                    tempstep.Done,      //true if x = target

			StepIndex:        tempstep.StepIndex,        //increments each step
			DistanceToTarget: tempstep.DistanceToTarget, //target - current position

			Return:    returns[i],
			Advantage: advantages[i],
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

func LoadPPO() (PPO, error) {
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
		//fmt.Printf("(float64(input): %.3f\nneuron.Weights[0]: %.3f\nneuron.Bias: %.3f\nLOGIT: %.3f\ntanH(%.3f): %.3f\n", float64(input), neuron.Weights[0], neuron.Bias, logit, logit, math.Tanh(logit))
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
